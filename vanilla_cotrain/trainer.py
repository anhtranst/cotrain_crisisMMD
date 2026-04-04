"""Vanilla Co-Training pipeline (Blum & Mitchell, 1998).

Iterative co-training where two models teach each other by selecting their
most confident predictions as hard pseudo-labels for the other model.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from lg_cotrain.data_loading import (
    ImageDataset,
    MultimodalDataset,
    TweetDataset,
    build_label_encoder,
    detect_classes,
    load_tsv,
    split_labeled_set,
)
from lg_cotrain.evaluate import compute_ece, compute_metrics
from lg_cotrain.model import BertClassifier, ImageClassifier, create_fresh_model
from lg_cotrain.utils import (
    EarlyStopping,
    get_device,
    set_seed,
    setup_logging,
)

from .config import VanillaCoTrainConfig

logger = logging.getLogger("vanilla_cotrain")


class VanillaCoTrainer:
    """Iterative vanilla co-training pipeline."""

    def __init__(self, config: VanillaCoTrainConfig):
        self.config = config
        self.device = get_device(config.device)

        # Text tokenizer (needed for text_only and text_image)
        if config.modality in ("text_only", "text_image"):
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        else:
            self.tokenizer = None

        # Image processor (needed for image_only and text_image)
        if config.modality in ("image_only", "text_image"):
            from transformers import CLIPImageProcessor
            self.image_processor = CLIPImageProcessor.from_pretrained(
                config.image_model_name
            )
        else:
            self.image_processor = None

    # ------------------------------------------------------------------
    # View-aware model creation
    # ------------------------------------------------------------------

    def _create_model_for_view(self, view: str):
        """Create a fresh model for the given view.

        For text_image: view "A" -> BertClassifier, view "B" -> ImageClassifier.
        For text_only / image_only: both views use the same model type.
        """
        cfg = self.config
        if cfg.modality == "text_image":
            if view == "A":
                return BertClassifier(cfg.model_name, cfg.num_labels).to(self.device)
            else:
                return ImageClassifier(cfg.image_model_name, cfg.num_labels).to(self.device)
        elif cfg.modality == "image_only":
            return ImageClassifier(cfg.image_model_name, cfg.num_labels).to(self.device)
        else:  # text_only
            return BertClassifier(cfg.model_name, cfg.num_labels).to(self.device)

    # ------------------------------------------------------------------
    # View-aware dataset creation
    # ------------------------------------------------------------------

    def _make_dataset_for_view(self, df, view: str, label_column="class_label"):
        """Create a dataset appropriate for the given view.

        For text_image:
            view "A" -> TweetDataset (text features)
            view "B" -> ImageDataset (image features)
        For text_only: TweetDataset for both views.
        For image_only: ImageDataset for both views.
        """
        labels = self._encode_labels(df[label_column])
        cfg = self.config

        if cfg.modality == "text_image":
            if view == "A":
                return TweetDataset(
                    texts=df["tweet_text"].tolist(),
                    labels=labels,
                    tokenizer=self.tokenizer,
                    max_length=cfg.max_seq_length,
                )
            else:
                return ImageDataset(
                    image_paths=df["image_path"].tolist(),
                    labels=labels,
                    project_root=cfg.project_root,
                    image_processor=self.image_processor,
                )
        elif cfg.modality == "image_only":
            return ImageDataset(
                image_paths=df["image_path"].tolist(),
                labels=labels,
                project_root=cfg.project_root,
                image_processor=self.image_processor,
            )
        else:  # text_only
            return TweetDataset(
                texts=df["tweet_text"].tolist(),
                labels=labels,
                tokenizer=self.tokenizer,
                max_length=cfg.max_seq_length,
            )

    def _encode_labels(self, series):
        return [self.label2id[lbl] for lbl in series]

    # ------------------------------------------------------------------
    # View-aware forward / predict
    # ------------------------------------------------------------------

    def _forward_batch(self, model, batch, view: str):
        """Run model forward on a batch, returning logits. View-aware."""
        cfg = self.config
        if cfg.modality == "text_image":
            if view == "A":
                return model(
                    batch["input_ids"].to(self.device),
                    batch["attention_mask"].to(self.device),
                )
            else:
                return model(batch["pixel_values"].to(self.device))
        elif cfg.modality == "image_only":
            return model(batch["pixel_values"].to(self.device))
        else:  # text_only
            return model(
                batch["input_ids"].to(self.device),
                batch["attention_mask"].to(self.device),
            )

    def _predict_proba_batch(self, model, batch, view: str):
        """Run model.predict_proba on a batch. View-aware."""
        cfg = self.config
        if cfg.modality == "text_image":
            if view == "A":
                return model.predict_proba(
                    batch["input_ids"].to(self.device),
                    batch["attention_mask"].to(self.device),
                )
            else:
                return model.predict_proba(batch["pixel_values"].to(self.device))
        elif cfg.modality == "image_only":
            return model.predict_proba(batch["pixel_values"].to(self.device))
        else:  # text_only
            return model.predict_proba(
                batch["input_ids"].to(self.device),
                batch["attention_mask"].to(self.device),
            )

    # ------------------------------------------------------------------
    # Sample selection: top-k per class
    # ------------------------------------------------------------------

    def _select_top_k_per_class(
        self, probs: np.ndarray, pred_labels: np.ndarray
    ) -> np.ndarray:
        """Select top-k most confident samples per class.

        Args:
            probs: Softmax probability matrix, shape (N, num_classes).
            pred_labels: Predicted class ids, shape (N,).

        Returns:
            Array of selected indices into the input arrays.
        """
        k = self.config.samples_per_class
        selected = []
        confidences = probs.max(axis=1)

        for cls_id in range(self.config.num_labels):
            cls_mask = pred_labels == cls_id
            cls_indices = np.where(cls_mask)[0]
            if len(cls_indices) == 0:
                continue
            cls_confs = confidences[cls_indices]
            # Top-k by confidence within this class
            top_k_local = min(k, len(cls_indices))
            top_k_idx = np.argsort(cls_confs)[-top_k_local:]
            selected.extend(cls_indices[top_k_idx])

        return np.array(selected, dtype=int)

    # ------------------------------------------------------------------
    # Training loop for a single model
    # ------------------------------------------------------------------

    def _train_model(self, model, train_loader, num_epochs, view: str):
        """Train a model for a given number of epochs. Returns the model."""
        cfg = self.config
        optimizer = AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        total_steps = num_epochs * len(train_loader)
        warmup_steps = int(total_steps * cfg.warmup_ratio)
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                logits = self._forward_batch(model, batch, view)
                loss = F.cross_entropy(logits, batch["labels"].to(self.device))
                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()

        return model

    # ------------------------------------------------------------------
    # Predict on unlabeled pool
    # ------------------------------------------------------------------

    def _predict_on_pool(self, model, pool_loader, view: str):
        """Predict softmax probabilities on the unlabeled pool.

        Returns:
            probs: np.ndarray of shape (N, num_classes)
            pred_labels: np.ndarray of shape (N,)
        """
        model.eval()
        all_probs = []
        with torch.no_grad():
            for batch in pool_loader:
                probs = self._predict_proba_batch(model, batch, view)
                all_probs.append(probs.cpu().numpy())
        all_probs = np.concatenate(all_probs, axis=0)
        pred_labels = all_probs.argmax(axis=1)
        return all_probs, pred_labels

    # ------------------------------------------------------------------
    # Ensemble prediction (cross-modal aware)
    # ------------------------------------------------------------------

    def _ensemble_predict(self, model_a, model_b, df, label_column="class_label"):
        """Ensemble prediction: average softmax from both models, then argmax.

        For text_image: feeds text to model A, images to model B.
        For single-modality: feeds same data to both models.

        Returns:
            preds: np.ndarray of shape (N,)
            labels: np.ndarray of shape (N,)
            probs: np.ndarray of shape (N, num_classes)
        """
        model_a.eval()
        model_b.eval()

        ds_a = self._make_dataset_for_view(df, "A", label_column)
        ds_b = self._make_dataset_for_view(df, "B", label_column)
        loader_a = DataLoader(ds_a, batch_size=self.config.batch_size, shuffle=False)
        loader_b = DataLoader(ds_b, batch_size=self.config.batch_size, shuffle=False)

        all_probs_a = []
        all_probs_b = []
        all_labels = []

        with torch.no_grad():
            for batch_a, batch_b in zip(loader_a, loader_b):
                probs_a = self._predict_proba_batch(model_a, batch_a, "A")
                probs_b = self._predict_proba_batch(model_b, batch_b, "B")
                all_probs_a.append(probs_a.cpu().numpy())
                all_probs_b.append(probs_b.cpu().numpy())
                all_labels.append(batch_a["labels"].numpy())

        all_probs_a = np.concatenate(all_probs_a, axis=0)
        all_probs_b = np.concatenate(all_probs_b, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        avg_probs = (all_probs_a + all_probs_b) / 2.0
        preds = avg_probs.argmax(axis=1)

        return preds, all_labels, avg_probs

    # ------------------------------------------------------------------
    # Save predictions
    # ------------------------------------------------------------------

    def _save_predictions(self, df, preds, probs, output_dir, filename):
        """Save per-sample predictions to a TSV file."""
        out_df = df.copy().reset_index(drop=True)
        out_df["predicted_label"] = [self.id2label[p] for p in preds]
        out_df["confidence"] = probs.max(axis=1).round(4)

        for class_id in range(probs.shape[1]):
            label_name = self.id2label[class_id]
            out_df[f"prob_{label_name}"] = probs[:, class_id].round(6)

        out_path = Path(output_dir) / filename
        out_df.to_csv(out_path, sep="\t", index=False)
        logger.info(f"Predictions saved to {out_path} ({len(out_df)} samples)")

    # ------------------------------------------------------------------
    # Main algorithm
    # ------------------------------------------------------------------

    def run(self) -> Dict:
        """Run the vanilla co-training pipeline and return metrics."""
        cfg = self.config
        if cfg.device and cfg.device.startswith("cuda:"):
            gpu_idx = cfg.device.split(":")[1]
            log_filename = f"experiment_gpu{gpu_idx}.log"
        else:
            log_filename = "experiment.log"
        setup_logging(cfg.log_dir, log_filename=log_filename)
        set_seed(cfg.seed_set)
        logger.info(
            f"Starting Vanilla Co-Training: task={cfg.task}, modality={cfg.modality}, "
            f"budget={cfg.budget}, seed_set={cfg.seed_set}"
        )

        # Load data
        df_labeled = load_tsv(cfg.labeled_path, modality=cfg.modality)
        df_unlabeled = load_tsv(cfg.unlabeled_path, modality=cfg.modality)
        df_dev = load_tsv(cfg.dev_path, modality=cfg.modality)
        df_test = load_tsv(cfg.test_path, modality=cfg.modality)

        # Detect classes and build label encoder
        detected_classes = detect_classes(df_labeled, df_unlabeled, df_dev, df_test)
        cfg.num_labels = len(detected_classes)
        self.label2id, self.id2label = build_label_encoder(labels=detected_classes)
        logger.info(f"Detected {cfg.num_labels} classes: {detected_classes}")

        # Split labeled set into D_l1, D_l2
        df_l1, df_l2 = split_labeled_set(df_labeled, seed=cfg.seed_set)

        # Optionally cap unlabeled set size
        if cfg.max_unlabeled_samples is not None and len(df_unlabeled) > cfg.max_unlabeled_samples:
            df_unlabeled = df_unlabeled.sample(
                n=cfg.max_unlabeled_samples, random_state=cfg.seed_set
            ).reset_index(drop=True)
            logger.info(f"Capped unlabeled set to {cfg.max_unlabeled_samples} samples")

        # Initialize growing labeled sets and shrinking unlabeled pool
        labeled_A = df_l1.copy().reset_index(drop=True)
        labeled_B = df_l2.copy().reset_index(drop=True)
        unlabeled_pool = df_unlabeled.copy().reset_index(drop=True)

        logger.info(
            f"Initial: labeled_A={len(labeled_A)}, labeled_B={len(labeled_B)}, "
            f"unlabeled_pool={len(unlabeled_pool)}"
        )

        # Track per-iteration stats
        iteration_log = []
        total_added_to_A = 0
        total_added_to_B = 0

        # ========================
        # Iterative Co-Training
        # ========================
        for iteration in range(1, cfg.num_iterations + 1):
            logger.info(f"=== Iteration {iteration}/{cfg.num_iterations} ===")

            if len(unlabeled_pool) == 0:
                logger.info("Unlabeled pool exhausted. Stopping iterations.")
                break

            # Create fresh models
            model_a = self._create_model_for_view("A")
            model_b = self._create_model_for_view("B")

            # Train model A on labeled_A
            ds_a = self._make_dataset_for_view(labeled_A, "A")
            loader_a = DataLoader(ds_a, batch_size=cfg.batch_size, shuffle=True)
            self._train_model(model_a, loader_a, cfg.train_epochs, "A")

            # Train model B on labeled_B
            ds_b = self._make_dataset_for_view(labeled_B, "B")
            loader_b = DataLoader(ds_b, batch_size=cfg.batch_size, shuffle=True)
            self._train_model(model_b, loader_b, cfg.train_epochs, "B")

            # Model A predicts on unlabeled pool
            pool_ds_a = self._make_dataset_for_view(unlabeled_pool, "A")
            pool_loader_a = DataLoader(pool_ds_a, batch_size=cfg.batch_size, shuffle=False)
            probs_a, preds_a = self._predict_on_pool(model_a, pool_loader_a, "A")

            # Model B predicts on unlabeled pool
            pool_ds_b = self._make_dataset_for_view(unlabeled_pool, "B")
            pool_loader_b = DataLoader(pool_ds_b, batch_size=cfg.batch_size, shuffle=False)
            probs_b, preds_b = self._predict_on_pool(model_b, pool_loader_b, "B")

            # Select top-k per class from each model
            selected_by_a = self._select_top_k_per_class(probs_a, preds_a)
            selected_by_b = self._select_top_k_per_class(probs_b, preds_b)

            added_to_B = len(selected_by_a)
            added_to_A = len(selected_by_b)

            if added_to_A == 0 and added_to_B == 0:
                logger.info("No samples selected by either model. Stopping iterations.")
                break

            # Add model A's confident predictions to labeled_B (with predicted labels)
            if added_to_B > 0:
                new_for_B = unlabeled_pool.iloc[selected_by_a].copy()
                new_for_B["class_label"] = [self.id2label[p] for p in preds_a[selected_by_a]]
                labeled_B = pd.concat([labeled_B, new_for_B], ignore_index=True)

            # Add model B's confident predictions to labeled_A (with predicted labels)
            if added_to_A > 0:
                new_for_A = unlabeled_pool.iloc[selected_by_b].copy()
                new_for_A["class_label"] = [self.id2label[p] for p in preds_b[selected_by_b]]
                labeled_A = pd.concat([labeled_A, new_for_A], ignore_index=True)

            # Remove all selected samples from unlabeled pool
            all_selected = np.union1d(selected_by_a, selected_by_b)
            unlabeled_pool = unlabeled_pool.drop(
                unlabeled_pool.index[all_selected]
            ).reset_index(drop=True)

            total_added_to_A += added_to_A
            total_added_to_B += added_to_B

            # Evaluate ensemble on dev
            dev_preds, dev_labels, _ = self._ensemble_predict(model_a, model_b, df_dev)
            dev_metrics = compute_metrics(dev_labels, dev_preds)

            iter_stats = {
                "iteration": iteration,
                "added_to_A": added_to_A,
                "added_to_B": added_to_B,
                "labeled_A_size": len(labeled_A),
                "labeled_B_size": len(labeled_B),
                "unlabeled_remaining": len(unlabeled_pool),
                "dev_macro_f1": dev_metrics["macro_f1"],
                "dev_error_rate": dev_metrics["error_rate"],
            }
            iteration_log.append(iter_stats)

            logger.info(
                f"Iteration {iteration}: added_to_A={added_to_A}, added_to_B={added_to_B}, "
                f"pool_remaining={len(unlabeled_pool)}, "
                f"dev_macro_f1={dev_metrics['macro_f1']:.4f}"
            )

            # Free GPU memory between iterations
            del model_a, model_b
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        num_iterations_completed = len(iteration_log)

        # ========================
        # Final Fine-Tuning
        # ========================
        logger.info("=== Final Fine-Tuning ===")
        model_a = self._create_model_for_view("A")
        model_b = self._create_model_for_view("B")

        # Fine-tune model A on labeled_A with early stopping
        es_a = EarlyStopping(patience=cfg.finetune_patience)
        opt_a = AdamW(model_a.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        ds_ft_a = self._make_dataset_for_view(labeled_A, "A")
        loader_ft_a = DataLoader(ds_ft_a, batch_size=cfg.batch_size, shuffle=True)
        total_ft_steps = cfg.finetune_max_epochs * len(loader_ft_a)
        warmup_ft = int(total_ft_steps * cfg.warmup_ratio)
        from transformers import get_linear_schedule_with_warmup
        sched_a = get_linear_schedule_with_warmup(
            opt_a, num_warmup_steps=warmup_ft, num_training_steps=total_ft_steps
        )

        # Fine-tune model B on labeled_B with early stopping
        es_b = EarlyStopping(patience=cfg.finetune_patience)
        opt_b = AdamW(model_b.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        ds_ft_b = self._make_dataset_for_view(labeled_B, "B")
        loader_ft_b = DataLoader(ds_ft_b, batch_size=cfg.batch_size, shuffle=True)
        total_ft_steps_b = cfg.finetune_max_epochs * len(loader_ft_b)
        warmup_ft_b = int(total_ft_steps_b * cfg.warmup_ratio)
        sched_b = get_linear_schedule_with_warmup(
            opt_b, num_warmup_steps=warmup_ft_b, num_training_steps=total_ft_steps_b
        )

        for epoch in range(cfg.finetune_max_epochs):
            # Train model A
            model_a.train()
            for batch in loader_ft_a:
                opt_a.zero_grad()
                logits = self._forward_batch(model_a, batch, "A")
                loss = F.cross_entropy(logits, batch["labels"].to(self.device))
                loss.backward()
                opt_a.step()
                sched_a.step()

            # Train model B
            model_b.train()
            for batch in loader_ft_b:
                opt_b.zero_grad()
                logits = self._forward_batch(model_b, batch, "B")
                loss = F.cross_entropy(logits, batch["labels"].to(self.device))
                loss.backward()
                opt_b.step()
                sched_b.step()

            # Evaluate ensemble on dev
            dev_preds, dev_labels, _ = self._ensemble_predict(model_a, model_b, df_dev)
            dev_metrics = compute_metrics(dev_labels, dev_preds)
            dev_f1 = dev_metrics["macro_f1"]

            stop_a = es_a.step(dev_f1, model_a)
            stop_b = es_b.step(dev_f1, model_b)

            logger.info(
                f"Fine-tune epoch {epoch+1}: dev_macro_f1={dev_f1:.4f}, "
                f"dev_err={dev_metrics['error_rate']:.2f}%, "
                f"es_counter_a={es_a.counter}, es_counter_b={es_b.counter}"
            )

            if stop_a and stop_b:
                logger.info(f"Early stopping at fine-tune epoch {epoch+1}")
                break

        # Restore best models
        es_a.restore_best(model_a)
        es_b.restore_best(model_b)

        # ========================
        # Final Evaluation
        # ========================
        logger.info("=== Final Evaluation ===")
        test_preds, test_labels, test_probs = self._ensemble_predict(
            model_a, model_b, df_test
        )
        test_metrics = compute_metrics(test_labels, test_preds)
        test_ece = compute_ece(test_labels, test_probs)

        dev_preds, dev_labels, dev_probs = self._ensemble_predict(
            model_a, model_b, df_dev
        )
        dev_metrics = compute_metrics(dev_labels, dev_preds)
        dev_ece = compute_ece(dev_labels, dev_probs)

        results = {
            "task": cfg.task,
            "modality": cfg.modality,
            "method": cfg.method,
            "budget": cfg.budget,
            "seed_set": cfg.seed_set,
            "test_error_rate": test_metrics["error_rate"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_weighted_f1": test_metrics["weighted_f1"],
            "test_macro_precision": test_metrics["macro_precision"],
            "test_macro_recall": test_metrics["macro_recall"],
            "test_weighted_precision": test_metrics["weighted_precision"],
            "test_weighted_recall": test_metrics["weighted_recall"],
            "test_ece": test_ece,
            "test_per_class_f1": test_metrics["per_class_f1"],
            "dev_error_rate": dev_metrics["error_rate"],
            "dev_macro_f1": dev_metrics["macro_f1"],
            "dev_weighted_f1": dev_metrics["weighted_f1"],
            "dev_ece": dev_ece,
            "num_iterations_completed": num_iterations_completed,
            "samples_added_to_A": total_added_to_A,
            "samples_added_to_B": total_added_to_B,
            "final_labeled_A_size": len(labeled_A),
            "final_labeled_B_size": len(labeled_B),
            "unlabeled_remaining": len(unlabeled_pool),
            "samples_per_class": cfg.samples_per_class,
            "per_iteration_log": iteration_log,
        }

        # Save results
        output_path = Path(cfg.output_dir) / "metrics.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
        logger.info(
            f"Test error rate: {test_metrics['error_rate']:.2f}%, "
            f"Test macro-F1: {test_metrics['macro_f1']:.4f}, "
            f"Test ECE: {test_ece:.4f}"
        )

        # Save per-sample predictions
        self._save_predictions(
            df_test, test_preds, test_probs, cfg.output_dir, "test_predictions.tsv"
        )
        self._save_predictions(
            df_dev, dev_preds, dev_probs, cfg.output_dir, "dev_predictions.tsv"
        )

        return results
