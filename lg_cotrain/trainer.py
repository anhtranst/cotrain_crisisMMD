"""3-phase LG-CoTrain pipeline: weight generation, co-training, fine-tuning."""

import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .config import LGCoTrainConfig
from .data_loading import (
    ImageDataset,
    MultimodalDataset,
    TweetDataset,
    build_d_lg,
    build_label_encoder,
    detect_classes,
    load_pseudo_labels,
    load_tsv,
    split_labeled_set,
)
from .evaluate import compute_ece, compute_metrics, ensemble_predict
from .model import create_fresh_model
from .utils import (
    EarlyStopping,
    EarlyStoppingWithDelta,
    PerClassEarlyStopping,
    compute_class_weights,
    compute_imbalance_ratio,
    get_device,
    set_seed,
    setup_logging,
)
from .weight_tracker import WeightTracker

logger = logging.getLogger("lg_cotrain")


class LGCoTrainer:
    """Orchestrates the 3-phase LG-CoTrain pipeline."""

    def __init__(self, config: LGCoTrainConfig):
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

    def _make_dataset(self, df, label_column="class_label"):
        """Create a dataset from a DataFrame, dispatched by modality."""
        labels = self._encode_labels(df[label_column])
        modality = self.config.modality

        if modality == "image_only":
            return ImageDataset(
                image_paths=df["image_path"].tolist(),
                labels=labels,
                project_root=self.config.project_root,
                image_processor=self.image_processor,
            )
        elif modality == "text_image":
            return MultimodalDataset(
                texts=df["tweet_text"].tolist(),
                image_paths=df["image_path"].tolist(),
                labels=labels,
                tokenizer=self.tokenizer,
                image_processor=self.image_processor,
                project_root=self.config.project_root,
                max_length=self.config.max_seq_length,
            )
        else:  # text_only
            return TweetDataset(
                texts=df["tweet_text"].tolist(),
                labels=labels,
                tokenizer=self.tokenizer,
                max_length=self.config.max_seq_length,
            )

    def _encode_labels(self, series):
        return [self.label2id[lbl] for lbl in series]

    def _forward_batch(self, model, batch):
        """Run model forward on a batch, returning logits. Modality-aware."""
        modality = self.config.modality
        if modality == "image_only":
            return model(batch["pixel_values"].to(self.device))
        elif modality == "text_image":
            return model(
                batch["input_ids"].to(self.device),
                batch["attention_mask"].to(self.device),
                batch["pixel_values"].to(self.device),
            )
        else:  # text_only
            return model(
                batch["input_ids"].to(self.device),
                batch["attention_mask"].to(self.device),
            )

    def _predict_proba_batch(self, model, batch):
        """Run model.predict_proba on a batch. Modality-aware."""
        modality = self.config.modality
        if modality == "image_only":
            return model.predict_proba(batch["pixel_values"].to(self.device))
        elif modality == "text_image":
            return model.predict_proba(
                batch["input_ids"].to(self.device),
                batch["attention_mask"].to(self.device),
                batch["pixel_values"].to(self.device),
            )
        else:  # text_only
            return model.predict_proba(
                batch["input_ids"].to(self.device),
                batch["attention_mask"].to(self.device),
            )

    def _collect_probs(self, model, loader, pseudo_label_ids) -> np.ndarray:
        """Collect p(pseudo_label | x; theta) for each sample in D_LG.

        Args:
            model: The model to evaluate.
            loader: Non-shuffled DataLoader over D_LG.
            pseudo_label_ids: Array of pseudo-label class indices per sample.

        Returns:
            Array of shape (num_samples,) with per-sample probabilities.
        """
        model.eval()
        all_probs = []
        all_indices = []

        with torch.no_grad():
            for batch in loader:
                sample_idx = batch["sample_idx"].numpy()
                probs = self._predict_proba_batch(model, batch).cpu().numpy()
                all_probs.append(probs)
                all_indices.append(sample_idx)

        all_probs = np.concatenate(all_probs, axis=0)  # (N, num_classes)
        all_indices = np.concatenate(all_indices, axis=0)

        # Extract p(pseudo_label) for each sample
        result = np.zeros(len(pseudo_label_ids))
        for i, idx in enumerate(all_indices):
            result[idx] = all_probs[i, pseudo_label_ids[idx]]

        return result

    def run(self) -> Dict:
        """Run the full 3-phase pipeline and return metrics."""
        cfg = self.config
        setup_logging(cfg.output_dir)
        set_seed(cfg.seed_set)
        logger.info(f"Starting LG-CoTrain: task={cfg.task}, modality={cfg.modality}, budget={cfg.budget}, seed_set={cfg.seed_set}")

        # Validate phase1_seed_strategy
        if cfg.phase1_seed_strategy not in ("last", "best"):
            raise ValueError(
                f"Unknown phase1_seed_strategy: {cfg.phase1_seed_strategy!r}. "
                "Valid: 'last', 'best'"
            )

        # Load data
        df_labeled = load_tsv(cfg.labeled_path, modality=cfg.modality)
        df_unlabeled = load_tsv(cfg.unlabeled_path, modality=cfg.modality)
        df_pseudo = load_pseudo_labels(cfg.pseudo_label_path)
        df_dev = load_tsv(cfg.dev_path, modality=cfg.modality)
        df_test = load_tsv(cfg.test_path, modality=cfg.modality)

        # Detect classes and build label encoder
        detected_classes = detect_classes(df_labeled, df_unlabeled, df_dev, df_test)
        cfg.num_labels = len(detected_classes)
        self.label2id, self.id2label = build_label_encoder(labels=detected_classes)
        logger.info(f"Detected {cfg.num_labels} classes: {detected_classes}")

        # Split labeled set
        df_l1, df_l2 = split_labeled_set(df_labeled, seed=cfg.seed_set)

        # Optionally cap unlabeled set size (for fast smoke tests)
        if cfg.max_unlabeled_samples is not None and len(df_unlabeled) > cfg.max_unlabeled_samples:
            df_unlabeled = df_unlabeled.sample(n=cfg.max_unlabeled_samples, random_state=cfg.seed_set).reset_index(drop=True)
            logger.info(f"Capped unlabeled set to {cfg.max_unlabeled_samples} samples")

        # Build D_LG
        df_dlg = build_d_lg(df_unlabeled, df_pseudo, modality=cfg.modality)
        pseudo_label_ids = np.array(self._encode_labels(df_dlg["predicted_label"]))
        num_dlg = len(df_dlg)
        logger.info(f"D_l1: {len(df_l1)}, D_l2: {len(df_l2)}, D_LG: {num_dlg}")

        # Build datasets
        ds_l1 = self._make_dataset(df_l1)
        ds_l2 = self._make_dataset(df_l2)
        ds_dlg = self._make_dataset(df_dlg, label_column="predicted_label")
        ds_dev = self._make_dataset(df_dev)
        ds_test = self._make_dataset(df_test)

        # DataLoaders
        loader_l1 = DataLoader(ds_l1, batch_size=cfg.batch_size, shuffle=True)
        loader_l2 = DataLoader(ds_l2, batch_size=cfg.batch_size, shuffle=True)
        loader_dlg_train = DataLoader(ds_dlg, batch_size=cfg.batch_size, shuffle=True)
        loader_dlg_eval = DataLoader(ds_dlg, batch_size=cfg.batch_size, shuffle=False)
        loader_dev = DataLoader(ds_dev, batch_size=cfg.batch_size, shuffle=False)
        loader_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False)

        # ========================
        # Phase 1: Weight Generation
        # ========================
        logger.info("=== Phase 1: Weight Generation ===")
        model1 = create_fresh_model(cfg).to(self.device)
        model2 = create_fresh_model(cfg).to(self.device)
        opt1 = AdamW(model1.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        opt2 = AdamW(model2.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        tracker1 = WeightTracker(num_dlg)
        tracker2 = WeightTracker(num_dlg)

        # Track best Phase 1 epoch by ensemble dev macro-F1 (used when phase1_seed_strategy="best")
        best_phase1_epoch = cfg.weight_gen_epochs - 1  # default: last epoch
        best_phase1_f1 = -1.0

        for epoch in range(cfg.weight_gen_epochs):
            # Train model1 on D_l1
            model1.train()
            for batch in loader_l1:
                opt1.zero_grad()
                logits = self._forward_batch(model1, batch)
                loss = F.cross_entropy(logits, batch["labels"].to(self.device))
                loss.backward()
                opt1.step()

            # Train model2 on D_l2
            model2.train()
            for batch in loader_l2:
                opt2.zero_grad()
                logits = self._forward_batch(model2, batch)
                loss = F.cross_entropy(logits, batch["labels"].to(self.device))
                loss.backward()
                opt2.step()

            # Record probabilities over D_LG
            probs1 = self._collect_probs(model1, loader_dlg_eval, pseudo_label_ids)
            probs2 = self._collect_probs(model2, loader_dlg_eval, pseudo_label_ids)
            tracker1.record_epoch(probs1)
            tracker2.record_epoch(probs2)

            # Evaluate ensemble on dev for "best" seeding strategy
            if cfg.phase1_seed_strategy == "best":
                dev_preds_p1, dev_labels_p1, _ = ensemble_predict(
                    model1, model2, loader_dev, self.device, modality=cfg.modality
                )
                dev_metrics_p1 = compute_metrics(dev_labels_p1, dev_preds_p1)
                p1_dev_f1 = dev_metrics_p1["macro_f1"]
                if p1_dev_f1 > best_phase1_f1:
                    best_phase1_f1 = p1_dev_f1
                    best_phase1_epoch = epoch
                logger.info(
                    f"Phase 1 epoch {epoch+1}/{cfg.weight_gen_epochs}: "
                    f"mean_prob1={probs1.mean():.4f}, mean_prob2={probs2.mean():.4f}, "
                    f"dev_macro_f1={p1_dev_f1:.4f}"
                )
            else:
                logger.info(
                    f"Phase 1 epoch {epoch+1}/{cfg.weight_gen_epochs}: "
                    f"mean_prob1={probs1.mean():.4f}, mean_prob2={probs2.mean():.4f}"
                )

        # ========================
        # Phase 2: Co-Training
        # ========================
        logger.info("=== Phase 2: Co-Training ===")
        model1 = create_fresh_model(cfg).to(self.device)
        model2 = create_fresh_model(cfg).to(self.device)
        opt1 = AdamW(model1.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        opt2 = AdamW(model2.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        # Linear LR scheduler with warmup spanning Phase 2 + Phase 3.
        num_batches_dlg = len(loader_dlg_train)
        num_batches_l1 = len(loader_l1)
        num_batches_l2 = len(loader_l2)
        total_steps_1 = num_batches_dlg * cfg.cotrain_epochs + num_batches_l1 * cfg.finetune_max_epochs
        total_steps_2 = num_batches_dlg * cfg.cotrain_epochs + num_batches_l2 * cfg.finetune_max_epochs
        warmup_steps_1 = int(total_steps_1 * cfg.warmup_ratio)
        warmup_steps_2 = int(total_steps_2 * cfg.warmup_ratio)
        scheduler1 = get_linear_schedule_with_warmup(opt1, warmup_steps_1, total_steps_1)
        scheduler2 = get_linear_schedule_with_warmup(opt2, warmup_steps_2, total_steps_2)

        logger.info(
            f"LR scheduler: total_steps_1={total_steps_1}, warmup={warmup_steps_1}; "
            f"total_steps_2={total_steps_2}, warmup={warmup_steps_2}"
        )

        # Seed Phase 2 trackers based on phase1_seed_strategy.
        if cfg.phase1_seed_strategy == "best":
            logger.info(
                f"Phase 1 best epoch: {best_phase1_epoch + 1}/{cfg.weight_gen_epochs} "
                f"(dev_macro_f1={best_phase1_f1:.4f})"
            )
            cotrain_tracker1 = WeightTracker.seed_from_epoch(tracker1, best_phase1_epoch)
            cotrain_tracker2 = WeightTracker.seed_from_epoch(tracker2, best_phase1_epoch)
        else:
            # Default "last": per Algorithm 1, seed with final Phase 1 epoch.
            cotrain_tracker1 = WeightTracker.seed_from_last_epoch(tracker1)
            cotrain_tracker2 = WeightTracker.seed_from_last_epoch(tracker2)

        # Initial lambdas from seeded trackers (1 epoch -> variability=0, lambda1=lambda2=confidence)
        lambda1 = cotrain_tracker1.compute_lambda_optimistic()
        lambda2 = cotrain_tracker2.compute_lambda_conservative()
        logger.info(
            f"Phase 1 done -> seeded Phase 2 trackers. "
            f"lambda1: mean={lambda1.mean():.4f}, range=[{lambda1.min():.4f}, {lambda1.max():.4f}]"
        )
        logger.info(
            f"Phase 1 done -> seeded Phase 2 trackers. "
            f"lambda2: mean={lambda2.mean():.4f}, range=[{lambda2.min():.4f}, {lambda2.max():.4f}]"
        )

        for epoch in range(cfg.cotrain_epochs):
            model1.train()
            model2.train()
            epoch_loss1 = 0.0
            epoch_loss2 = 0.0
            n_batches = 0

            for batch in loader_dlg_train:
                labels = batch["labels"].to(self.device)
                sample_idx = batch["sample_idx"].numpy()

                # Get per-sample weights
                w2 = torch.tensor(
                    lambda2[sample_idx], dtype=torch.float32, device=self.device
                )
                w1 = torch.tensor(
                    lambda1[sample_idx], dtype=torch.float32, device=self.device
                )

                # Model 1 loss (uses theta2's weights = lambda2)
                logits1 = self._forward_batch(model1, batch)
                per_sample_loss1 = F.cross_entropy(logits1, labels, reduction="none")
                loss1 = (w2 * per_sample_loss1).mean()

                opt1.zero_grad()
                loss1.backward()
                opt1.step()
                scheduler1.step()

                # Model 2 loss (uses theta1's weights = lambda1)
                logits2 = self._forward_batch(model2, batch)
                per_sample_loss2 = F.cross_entropy(logits2, labels, reduction="none")
                loss2 = (w1 * per_sample_loss2).mean()

                opt2.zero_grad()
                loss2.backward()
                opt2.step()
                scheduler2.step()

                epoch_loss1 += loss1.item()
                epoch_loss2 += loss2.item()
                n_batches += 1

            # Update probabilities and recompute lambdas
            probs1 = self._collect_probs(model1, loader_dlg_eval, pseudo_label_ids)
            probs2 = self._collect_probs(model2, loader_dlg_eval, pseudo_label_ids)
            cotrain_tracker1.record_epoch(probs1)
            cotrain_tracker2.record_epoch(probs2)
            lambda1 = cotrain_tracker1.compute_lambda_optimistic()
            lambda2 = cotrain_tracker2.compute_lambda_conservative()

            # Evaluate ensemble on dev
            dev_preds, dev_labels, _ = ensemble_predict(model1, model2, loader_dev, self.device, modality=cfg.modality)
            dev_metrics = compute_metrics(dev_labels, dev_preds)

            logger.info(
                f"Phase 2 epoch {epoch+1}/{cfg.cotrain_epochs}: "
                f"loss1={epoch_loss1/n_batches:.4f}, loss2={epoch_loss2/n_batches:.4f}, "
                f"dev_macro_f1={dev_metrics['macro_f1']:.4f}, dev_err={dev_metrics['error_rate']:.2f}%"
            )

        # ========================
        # Phase 3: Fine-Tuning
        # ========================
        logger.info("=== Phase 3: Fine-Tuning ===")
        strategy = cfg.stopping_strategy

        if strategy == "baseline":
            es1 = EarlyStopping(patience=cfg.finetune_patience)
            es2 = EarlyStopping(patience=cfg.finetune_patience)

        elif strategy == "no_early_stopping":
            # Run all finetune_max_epochs; patience is set too high to ever fire.
            # restore_best() still returns the best-ever checkpoint across all epochs.
            es1 = EarlyStopping(patience=cfg.finetune_max_epochs)
            es2 = EarlyStopping(patience=cfg.finetune_max_epochs)

        elif strategy == "per_class_patience":
            es1 = PerClassEarlyStopping(patience=cfg.finetune_patience,
                                        num_classes=cfg.num_labels)
            es2 = PerClassEarlyStopping(patience=cfg.finetune_patience,
                                        num_classes=cfg.num_labels)

        elif strategy == "weighted_macro_f1":
            class_weights = compute_class_weights(df_labeled["class_label"], self.label2id)
            es1 = EarlyStopping(patience=cfg.finetune_patience)
            es2 = EarlyStopping(patience=cfg.finetune_patience)

        elif strategy == "balanced_dev":
            _counts = df_dev["class_label"].value_counts()
            _min_count = int(_counts.min())
            df_dev_balanced = (
                df_dev.groupby("class_label", group_keys=False)
                .sample(n=_min_count, random_state=cfg.seed_set)
            )
            ds_dev_balanced = self._make_dataset(df_dev_balanced)
            loader_dev_balanced = DataLoader(ds_dev_balanced,
                                             batch_size=cfg.batch_size, shuffle=False)
            es1 = EarlyStopping(patience=cfg.finetune_patience)
            es2 = EarlyStopping(patience=cfg.finetune_patience)

        elif strategy == "scaled_threshold":
            imbalance_ratio = compute_imbalance_ratio(df_labeled["class_label"])
            es1 = EarlyStoppingWithDelta(patience=cfg.finetune_patience,
                                         imbalance_ratio=imbalance_ratio)
            es2 = EarlyStoppingWithDelta(patience=cfg.finetune_patience,
                                         imbalance_ratio=imbalance_ratio)

        else:
            raise ValueError(
                f"Unknown stopping_strategy: {strategy!r}. "
                "Valid: 'baseline', 'no_early_stopping', 'per_class_patience', "
                "'weighted_macro_f1', 'balanced_dev', 'scaled_threshold'"
            )

        for epoch in range(cfg.finetune_max_epochs):
            # Fine-tune model1 on D_l1
            model1.train()
            for batch in loader_l1:
                opt1.zero_grad()
                logits = self._forward_batch(model1, batch)
                loss = F.cross_entropy(logits, batch["labels"].to(self.device))
                loss.backward()
                opt1.step()
                scheduler1.step()

            # Fine-tune model2 on D_l2
            model2.train()
            for batch in loader_l2:
                opt2.zero_grad()
                logits = self._forward_batch(model2, batch)
                loss = F.cross_entropy(logits, batch["labels"].to(self.device))
                loss.backward()
                opt2.step()
                scheduler2.step()

            # Evaluate ensemble on dev — always on full dev set for logging
            dev_preds, dev_labels, _ = ensemble_predict(model1, model2, loader_dev, self.device, modality=cfg.modality)
            dev_metrics = compute_metrics(dev_labels, dev_preds)
            dev_f1 = dev_metrics["macro_f1"]

            # Compute stopping signal based on strategy
            if strategy == "per_class_patience":
                stop1 = es1.step(dev_metrics["per_class_f1"], model1)
                stop2 = es2.step(dev_metrics["per_class_f1"], model2)
            elif strategy == "weighted_macro_f1":
                per_class = dev_metrics["per_class_f1"]
                stopping_score = (
                    sum(w * f for w, f in zip(class_weights, per_class))
                    / sum(class_weights)
                )
                stop1 = es1.step(stopping_score, model1)
                stop2 = es2.step(stopping_score, model2)
            elif strategy == "balanced_dev":
                bal_preds, bal_labels, _ = ensemble_predict(
                    model1, model2, loader_dev_balanced, self.device, modality=cfg.modality)
                bal_metrics = compute_metrics(bal_labels, bal_preds)
                stop1 = es1.step(bal_metrics["macro_f1"], model1)
                stop2 = es2.step(bal_metrics["macro_f1"], model2)
            else:  # baseline, no_early_stopping, scaled_threshold
                stop1 = es1.step(dev_f1, model1)
                stop2 = es2.step(dev_f1, model2)

            logger.info(
                f"Phase 3 epoch {epoch+1}: dev_macro_f1={dev_f1:.4f}, "
                f"dev_err={dev_metrics['error_rate']:.2f}%, "
                f"es_counter1={es1.counter}, es_counter2={es2.counter}"
            )

            if stop1 and stop2:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Restore best models
        es1.restore_best(model1)
        es2.restore_best(model2)

        # ========================
        # Final Evaluation
        # ========================
        logger.info("=== Final Evaluation ===")
        test_preds, test_labels, test_probs = ensemble_predict(model1, model2, loader_test, self.device, modality=cfg.modality)
        test_metrics = compute_metrics(test_labels, test_preds)
        test_ece = compute_ece(test_labels, test_probs)
        dev_preds, dev_labels, dev_probs = ensemble_predict(model1, model2, loader_dev, self.device, modality=cfg.modality)
        dev_metrics = compute_metrics(dev_labels, dev_preds)
        dev_ece = compute_ece(dev_labels, dev_probs)

        results = {
            "task": cfg.task,
            "modality": cfg.modality,
            "budget": cfg.budget,
            "seed_set": cfg.seed_set,
            "test_error_rate": test_metrics["error_rate"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_ece": test_ece,
            "test_per_class_f1": test_metrics["per_class_f1"],
            "dev_error_rate": dev_metrics["error_rate"],
            "dev_macro_f1": dev_metrics["macro_f1"],
            "dev_ece": dev_ece,
            "stopping_strategy": cfg.stopping_strategy,
            "phase1_seed_strategy": cfg.phase1_seed_strategy,
            "phase1_best_epoch": best_phase1_epoch + 1 if cfg.phase1_seed_strategy == "best" else None,
            "lambda1_mean": float(lambda1.mean()),
            "lambda1_std": float(lambda1.std()),
            "lambda2_mean": float(lambda2.mean()),
            "lambda2_std": float(lambda2.std()),
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

        return results
