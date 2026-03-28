#!/usr/bin/env python3
"""Zero-shot classification on CrisisMMD with Llama-3.2-11B-Vision-Instruct.

Supports informative (2 classes) and humanitarian (5 classes) tasks,
across text_only, image_only, and text_image modalities.

Usage:
    python scripts/zeroshot_llama.py --task informative --modality text_only --split test
    python scripts/zeroshot_llama.py --task humanitarian --modality image_only --split train
    python scripts/zeroshot_llama.py --task informative --modality text_image --split test --max-samples 50
"""

import argparse
import csv
import gc
import json
import os
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Label mappings (matching reference notebook numeric schemes)
# ---------------------------------------------------------------------------

# Informative: 0=not_informative, 1=informative (notebook convention)
INFO_ID2LABEL = {0: "not_informative", 1: "informative"}
INFO_LABEL2ID = {v: k for k, v in INFO_ID2LABEL.items()}

# Humanitarian: 5-class scheme (notebook convention)
HUMA_ID2LABEL = {
    0: "affected_individuals",
    1: "rescue_volunteering_or_donation_effort",
    2: "infrastructure_and_utility_damage",
    3: "other_relevant_information",
    4: "not_humanitarian",
}
HUMA_LABEL2ID = {v: k for k, v in HUMA_ID2LABEL.items()}

# ---------------------------------------------------------------------------
# Prompts — Informative (binary: 0/1)
# Exact prompts from temp/llama-3-fewshot-Informative.ipynb
# ---------------------------------------------------------------------------

INFO_TEXT_ONLY_PROMPT = """You are an AI model that classifies the given text into one of two categories based on its informational value. \
Your task is to analyze the given text to determine whether it contains relevant or informative content about a crisis.
Our goal is to collect as much useful information as possible about crises, so your classification should prioritize identifying texts that provide relevant details, even if they are brief or incomplete. \
Avoid being overly restrictive. If the text has any relevant crisis-related information, classify it as informative.
Classify the text delimited by triple quotes (\\"\\"\\" \\"\\"\\"\\") into one of the following categories:
  - **1 (positive):** The text provide details, updates, or any relevant information about a crisis.
  - **0 (not positive):** The text do not contain relevant details or information about a crisis.
Return only the classification label (1 or 0) without any extra text or explanation.

Here is the test Text that you need to classify: \\"\\"\\"{}\\"\\"\\". The category of this test Text is: """

INFO_IMAGE_ONLY_PROMPT = """Does the image give useful information that could help during a crisis?
Respond with '1' if this image provides any information or details about a crisis, and '0' if it does not.

Instructions:
  - You should prioritize identifying texts that provide relevant details, even if they are brief or incomplete.
  - Avoid being overly restrictive. If the text has any relevant crisis-related information, response with '1'.
  - When the meaning of the image is unclear, response with '0'.
  - Do not output any extra text.

Above is the test image that you need to classify. The category of this test image is: """

INFO_TEXT_IMAGE_PROMPT = """Do the given text and the given image give useful information that could help during a crisis?
Respond with '1' if the text and the image provide any information or details about a crisis, and '0' if they do not.

Instructions:
  - You should prioritize identifying texts and the images that provide relevant details, even if they are brief or incomplete.
  - Avoid being overly restrictive. If the text and image have any relevant crisis-related information, response with '1'.
  - When the meaning of the image and the text are unclear, response with '0'.
  - Do not output any extra text.

Above is the given image, and here is the given Text: {}.
The category of this image and Text is: """

# ---------------------------------------------------------------------------
# Prompts — Humanitarian (5 classes: 0-4)
# Exact prompts from temp/llama-3-fewshot-Humanitarian.ipynb
# ---------------------------------------------------------------------------

HUMA_TEXT_ONLY_PROMPT = """You are an expert in disaster response and humanitarian aid data analysis. Examine this text delimited by triple quotes (\\"\\"\\" \\"\\"\\"\\") carefully and classify it into exactly one of these categories (0-4). Respond with ONLY the number, no other text or explanation.

Categories:
0: HUMAN IMPACT - Must show direct human suffering or hardship:
- Deaths, injuries, or missing people
- People struggling without basic needs (food, water, shelter)
- Displaced or evacuated people
- Personal stories of survival or loss
- People stranded or waiting for rescue

1: RESPONSE EFFORTS - Any organized help effort, no matter how small:
- Rescue operations and emergency response
- Aid collection or distribution activities
- Donations of money, supplies, or services
- Volunteer work and relief efforts
- Medical assistance
- Fundraising events for disaster relief

2: INFRASTRUCTURE DAMAGE - Must describe specific physical destruction:
- Destroyed or damaged buildings and homes
- Damaged roads, bridges, or transportation systems
- Disrupted power lines or water systems
- Damaged vehicles or equipment
- Before/after comparisons showing destruction

3: CRISIS UPDATES - Must be specific to the crisis but not fit above categories:
- Weather forecasts and disaster warnings
- Maps or descriptions of impact areas
- Official announcements about the disaster
- Statistics and data about crisis impact
- Crisis reporting without specific damage/casualties/response

4: NOT CRISIS-RELATED - Use when no other category clearly fits:
- General discussion without crisis specifics
- Personal opinions about non-crisis aspects
- Promotional or commercial content
- Unclear connection to crisis
- Content that could apply to non-crisis situations

Important Decision Rules:
- If you see ANY mention of help, rescue, or donations -> Pick 1
- If you see ANY mention of human casualties, suffering, or displacement but not related to volunteer, rescue, donation... -> Pick 0
- If you see ANY specific physical destruction of properties -> Pick 2
- If it's clearly about the crisis but doesn't fit 0-2 -> Pick 3
- If multiple categories could apply, use the one BEST FITS the text
- Only use 4 when you are COMPLETELY SURE no other category fits
Answer with just a single digit (0-4).

Here is the test Text that you need to classify: \\"\\"\\"{}\\"\\"\\". The category of this test Text is: """

HUMA_IMAGE_ONLY_PROMPT = """You are an expert in disaster response and humanitarian aid image analysis. Examine the first image carefully and classify it into exactly one of these categories (0-4). Respond with ONLY the number, no other text or explanation.

Categories:
0: HUMAN IMPACT - Must show PEOPLE who are clearly affected by the disaster: injured, displaced, evacuated, in temporary shelters, or waiting in lines for aid. People must be visibly impacted.

1: RESPONSE EFFORTS - Must show active RESCUE operations, aid distribution, medical treatment, VOLUNTEER work, DONATION, or evacuation in progress. Look for emergency responders, relief workers, or organized aid activities.

2: INFRASTRUCTURE DAMAGE - Must show clear physical damage to buildings, roads, bridges, power lines, VEHICLES or other structures that was caused by the disaster. The damage should be obvious and significant.

3: OTHER CRISIS INFO - Shows verified crisis-related content that doesn't fit above categories: maps of affected areas, emergency warnings, official updates, or documentation of crisis conditions. Must have clear connection to the current disaster.

4: NOT CRISIS-RELATED - Use this for:
- Images where you're unsure if it's related to the crisis
- General photos that could be from any time/place
- Images without clear crisis impact or response
- Stock photos or promotional images
- Any image that doesn't definitively fit categories 0-3

Important:
- If there's ANY sign of rescue or donation, pick 1.
- If there's ANY sign of damage, pick 2.
- If there's ANY sign of obviously distressed or harmed people, pick 0.
- If it's definitely about a crisis but you DO NOT see rescue/damage/impacted people, pick 3.
- Otherwise, pick 4. Also, when you are not sure which number to pick, pick 4.
Answer with just a single digit (0-4).

Above is the test image that you need to classify. The category of this test image is: """

HUMA_TEXT_IMAGE_PROMPT = """You are an expert in disaster response and humanitarian aid data analysis. Examine the given text and image carefully and classify them into exactly one of these categories (0-4). Respond with ONLY the number, no other text or explanation.

Categories:
0: HUMAN IMPACT - Must be about PEOPLE who are clearly affected by the disaster: injured, displaced, evacuated, in temporary shelters, or waiting in lines for aid. People must be visibly impacted.

1: RESPONSE EFFORTS - Must be about active RESCUE operations, aid distribution, medical treatment, VOLUNTEER work, DONATION, or evacuation in progress. Look for emergency responders, relief workers, or organized aid activities.

2: INFRASTRUCTURE DAMAGE - Must be about clear physical damage to buildings, roads, bridges, power lines, VEHICLES or other structures that was caused by the disaster. The damage should be obvious and significant.

3: OTHER CRISIS INFO - Must be about verified crisis-related content that doesn't fit above categories: maps of affected areas, emergency warnings, official updates, or documentation of crisis conditions. Must have clear connection to the current disaster.

4: NOT CRISIS-RELATED - Use this for:
- Images and text where you're unsure if they are related to the crisis
- General texts and photos that could be from any time/place
- Texts and images without clear crisis impact or response
- Texts are not related to a crisis with stock photos or promotional images
- Any text and image that doesn't definitively fit categories 0-3

Important:
- If there's ANY sign of rescue or donation, pick 1.
- If there's ANY sign of damage, pick 2.
- If there's ANY sign of obviously distressed or harmed people, pick 0.
- If the text and image are definitely about a crisis but you DO NOT see rescue/damage/impacted people, pick 3.
- Otherwise, pick 4. Also, when you are not sure which number to pick, pick 4.
Answer with just a single digit (0-4).

Above is the given image, and here is the given Text: {}.
The category of this image and Text is: """

# Prompt lookup: (task, modality) -> prompt template
PROMPTS = {
    ("informative", "text_only"): INFO_TEXT_ONLY_PROMPT,
    ("informative", "image_only"): INFO_IMAGE_ONLY_PROMPT,
    ("informative", "text_image"): INFO_TEXT_IMAGE_PROMPT,
    ("humanitarian", "text_only"): HUMA_TEXT_ONLY_PROMPT,
    ("humanitarian", "image_only"): HUMA_IMAGE_ONLY_PROMPT,
    ("humanitarian", "text_image"): HUMA_TEXT_IMAGE_PROMPT,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def model_slug(model_id: str) -> str:
    """Derive a short folder name from a HuggingFace model ID.

    e.g. 'meta-llama/Llama-3.2-11B-Vision-Instruct' -> 'llama-3.2-11b'
         'Qwen/Qwen2-VL-7B-Instruct' -> 'qwen2-vl-7b'
    """
    name = model_id.split("/")[-1]           # drop org prefix
    name = name.lower()
    # Remove common suffixes
    for suffix in ["-vision-instruct", "-instruct", "-chat", "-vision"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

def _detect_informative_fallback(text):
    """Regex fallback for informative label detection."""
    text_lower = text.lower()
    if re.search(r"\bnot (provide|give|contain|related|relevant)", text_lower):
        return "not_informative"
    if re.search(r"\b(provides?|gives?)\s+(useful |some |relevant )?information\b", text_lower):
        return "informative"
    if re.search(r"\b0[.!]?\b", text):
        return "not_informative"
    if re.search(r"\b1[.!]?\b", text):
        return "informative"
    return "not_informative"  # default


def _detect_humanitarian_fallback(text):
    """Regex fallback for humanitarian label detection (5-class)."""
    text_clean = re.sub(r'[^A-Za-z0-9\s]', '', text).lower()
    if re.search(r"\b(rescue|volunteer|donation|donate|aid|relief|fundrais|help)", text_clean):
        return "rescue_volunteering_or_donation_effort"
    if re.search(r"\b(displaced|evacuated|affected|shelter|strand|injur|dead|death|missing|suffer)", text_clean):
        return "affected_individuals"
    if re.search(r"\b(damag|destroy|collaps|flood).*(building|road|bridge|infrastructure|house|power|vehicle)", text_clean):
        return "infrastructure_and_utility_damage"
    if re.search(r"\b(warning|forecast|map|statistic|update|announce|crisis)", text_clean):
        return "other_relevant_information"
    return "not_humanitarian"  # default


def parse_response(raw_text, task):
    """Parse model output into a string label.

    Splits on 'assistant', checks for leading digit, falls back to regex.
    Returns the string label or 'UNPARSEABLE'.
    """
    id2label = INFO_ID2LABEL if task == "informative" else HUMA_ID2LABEL
    max_id = max(id2label.keys())

    parts = raw_text.split("assistant")
    if len(parts) > 1:
        answer = parts[-1].strip()
    else:
        answer = raw_text.strip()

    # Check if first character is a valid digit
    if answer and answer[0].isdigit():
        digit = int(answer[0])
        if digit <= max_id:
            return id2label[digit]

    # Fallback to keyword detection
    if task == "informative":
        return _detect_informative_fallback(answer)
    else:
        return _detect_humanitarian_fallback(answer)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(task, modality, split, data_root):
    """Load test/train data from preprocessed TSV.

    Returns (fieldnames, rows) where rows is a list of OrderedDicts
    preserving all original columns.  The ``class_label`` column is
    always present regardless of modality.
    """
    tsv_path = Path(data_root) / "CrisisMMD" / "tasks" / task / modality / f"{split}.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(f"Data file not found: {tsv_path}")

    rows = []
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = list(reader.fieldnames)
        for row in reader:
            rows.append(dict(row))

    print(f"Loaded {len(rows)} samples from {tsv_path}")
    return fieldnames, rows


# ---------------------------------------------------------------------------
# Message construction
# ---------------------------------------------------------------------------

def build_messages(prompt, text=None, image_obj=None, modality="text_only"):
    """Build chat messages for the model (zero-shot).

    Returns (messages, images_list).
    """
    user_content = []

    if modality == "text_only":
        user_content.append({"type": "text", "text": prompt.format(text)})
        images = None

    elif modality == "image_only":
        user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": prompt})
        images = [image_obj]

    elif modality == "text_image":
        user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": prompt.format(text)})
        images = [image_obj]

    messages = [{"role": "user", "content": user_content}]
    return messages, images


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_single(model, processor, messages, images, task):
    """Run inference on a single sample. Returns (predicted_label, raw_output)."""
    import torch

    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

    if images:
        inputs = processor(
            text=input_text, images=images,
            return_tensors="pt", truncation=False,
        ).to(model.device)
    else:
        inputs = processor(
            text=input_text,
            return_tensors="pt", truncation=False,
        ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=20,
            min_new_tokens=1,
            do_sample=False,
            num_beams=1,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    raw_output = processor.decode(output_ids[0], skip_special_tokens=True)
    predicted_label = parse_response(raw_output, task)
    return predicted_label, raw_output


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_id):
    """Load the Llama Vision model and processor."""
    import torch
    from transformers import MllamaForConditionalGeneration, AutoProcessor

    print(f"Loading model: {model_id}")
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    print(f"Model loaded on {model.device}")
    return model, processor


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run_zeroshot(
    task, modality, split, model_id, data_root, output_dir,
    max_samples=None, model=None, processor=None,
):
    """Run zero-shot classification and save results.

    If model/processor are passed, reuses them (for notebook use).
    Otherwise loads fresh.

    Returns (predictions_list, metrics_dict).
    """
    from PIL import Image as PILImage
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support, f1_score,
    )

    # Load model if not provided
    own_model = model is None
    if own_model:
        model, processor = load_model(model_id)

    # Load data
    orig_fieldnames, data = load_data(task, modality, split, data_root)
    if max_samples:
        data = data[:max_samples]
        print(f"Limited to {max_samples} samples")

    prompt = PROMPTS[(task, modality)]
    valid_labels = set(INFO_ID2LABEL.values()) if task == "informative" else set(HUMA_ID2LABEL.values())

    predictions = []
    y_true = []
    y_pred = []
    n_unparseable = 0
    start_time = time.time()

    for i, item in enumerate(data):
        # Load image if needed
        image_obj = None
        if modality in ("image_only", "text_image") and item.get("image_path"):
            try:
                image_obj = PILImage.open(item["image_path"]).convert("RGB")
            except Exception as e:
                print(f"  WARNING: Could not load image {item['image_path']}: {e}")
                continue

        messages, images = build_messages(
            prompt, text=item.get("tweet_text"), image_obj=image_obj, modality=modality,
        )
        pred_label, raw_output = predict_single(model, processor, messages, images, task)

        if pred_label == "UNPARSEABLE" or pred_label not in valid_labels:
            n_unparseable += 1

        # Copy original row and append prediction columns
        row = dict(item)
        row["predicted_label"] = pred_label
        row["raw_output"] = raw_output
        predictions.append(row)

        y_true.append(item["class_label"])
        y_pred.append(pred_label)

        # Progress
        if (i + 1) % 50 == 0 or i == len(data) - 1:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(data) - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i+1}/{len(data)}] "
                f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining, "
                f"{rate:.1f} samples/s"
            )

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0,
    )
    m_prec, m_rec, m_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0,
    )

    # Per-class F1
    labels_sorted = sorted(valid_labels)
    per_class_prec, per_class_rec, per_class_f1, per_class_sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_sorted, average=None, zero_division=0,
    )
    per_class_f1_dict = {
        label: round(float(f), 4)
        for label, f in zip(labels_sorted, per_class_f1)
    }

    metrics = {
        "task": task,
        "modality": modality,
        "split": split,
        "model_id": model_id,
        "num_samples": len(predictions),
        "num_unparseable": n_unparseable,
        "accuracy": round(accuracy, 4),
        "weighted_precision": round(float(w_prec), 4),
        "weighted_recall": round(float(w_rec), 4),
        "weighted_f1": round(float(w_f1), 4),
        "macro_precision": round(float(m_prec), 4),
        "macro_recall": round(float(m_rec), 4),
        "macro_f1": round(float(m_f1), 4),
        "per_class_f1": per_class_f1_dict,
    }

    # Save results
    out_path = Path(output_dir) / model_slug(model_id) / task / modality / split
    out_path.mkdir(parents=True, exist_ok=True)

    pred_path = out_path / "predictions.tsv"
    out_fieldnames = orig_fieldnames + ["predicted_label", "raw_output"]
    with open(pred_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(predictions)

    metrics_path = out_path / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  Weighted F1: {w_f1:.4f}")
    print(f"  Macro F1:    {m_f1:.4f}")
    print(f"  Unparseable: {n_unparseable}")

    # Cleanup if we loaded the model
    if own_model:
        del model, processor
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    return predictions, metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot classification on CrisisMMD v2.0 with Llama Vision"
    )
    parser.add_argument("--task", required=True, choices=["informative", "humanitarian"])
    parser.add_argument("--modality", required=True, choices=["text_only", "image_only", "text_image"])
    parser.add_argument("--split", required=True, choices=["train", "test"])
    parser.add_argument("--model-id", default="meta-llama/Llama-3.2-11B-Vision-Instruct")
    parser.add_argument("--data-root", default=None,
                        help="Path to data/ directory (default: auto-detect from script location)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: results/zeroshot)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples (for debugging)")
    args = parser.parse_args()

    # Auto-detect paths relative to repo root
    repo_root = Path(__file__).resolve().parent.parent
    data_root = args.data_root or str(repo_root / "data")
    output_dir = args.output_dir or str(repo_root / "results" / "zeroshot")

    run_zeroshot(
        task=args.task,
        modality=args.modality,
        split=args.split,
        model_id=args.model_id,
        data_root=data_root,
        output_dir=output_dir,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
