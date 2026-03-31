#!/usr/bin/env python3
"""Debug script: show one actual prompt + raw response for each modality.

Picks the first sample from the test set for each (task, modality) combo,
prints the exact prompt sent to the model, and shows the raw response.

Usage:
    python scripts/debug_prompts.py
    python scripts/debug_prompts.py --task informative
    python scripts/debug_prompts.py --task humanitarian --modality text_only
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.zeroshot_llama import (
    PROMPTS, load_data, build_messages, predict_single,
    load_model, parse_response, model_slug,
    INFO_ID2LABEL, HUMA_ID2LABEL,
)

TASKS = ["informative", "humanitarian"]
MODALITIES = ["text_only", "image_only", "text_image"]


def debug_one(task, modality, model, processor, data_root):
    """Run one sample and print everything."""
    from PIL import Image as PILImage

    print(f"\n{'='*80}")
    print(f"  TASK: {task}  |  MODALITY: {modality}")
    print(f"{'='*80}")

    # Load first sample
    _, data = load_data(task, modality, "test", data_root)
    item = data[0]

    # Show input data
    print(f"\n--- INPUT DATA ---")
    for k, v in item.items():
        val = str(v)
        if len(val) > 100:
            val = val[:100] + "..."
        print(f"  {k}: {val}")

    # Load image if needed
    image_obj = None
    if modality in ("image_only", "text_image") and item.get("image_path"):
        img_path = Path(item["image_path"])
        if not img_path.is_absolute():
            img_path = Path(data_root).parent / img_path
        img = PILImage.open(img_path)
        image_obj = img.convert("RGBA").convert("RGB") if img.mode == "P" else img.convert("RGB")
        print(f"\n  [Image loaded: {img.size}, mode={img.mode}]")

    # Build prompt
    prompt = PROMPTS[(task, modality)]
    messages, images = build_messages(
        prompt, text=item.get("tweet_text"), image_obj=image_obj, modality=modality,
    )

    # Show the exact prompt text
    print(f"\n--- PROMPT SENT TO MODEL ---")
    for msg in messages:
        print(f"  role: {msg['role']}")
        for part in msg["content"]:
            if part["type"] == "text":
                print(f"  text: {repr(part['text'][:500])}")
                if len(part["text"]) > 500:
                    print(f"        ... ({len(part['text'])} chars total)")
            elif part["type"] == "image":
                print(f"  [IMAGE PLACEHOLDER]")

    # Show what processor produces
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    print(f"\n--- FORMATTED INPUT (after apply_chat_template) ---")
    print(f"  {repr(input_text[:600])}")
    if len(input_text) > 600:
        print(f"  ... ({len(input_text)} chars total)")

    # Run prediction
    print(f"\n--- MODEL RESPONSE ---")
    pred_label, raw_output = predict_single(model, processor, messages, images, task)

    print(f"  Raw output: {repr(raw_output)}")
    print(f"  Parsed label: {pred_label}")
    print(f"  Gold label:   {item['class_label']}")
    print(f"  Correct:      {'YES' if pred_label == item['class_label'] else 'NO'}")

    # Show label mapping for reference
    id2label = INFO_ID2LABEL if task == "informative" else HUMA_ID2LABEL
    print(f"\n--- LABEL MAPPING ---")
    for k, v in id2label.items():
        print(f"  {k} -> {v}")


def main():
    parser = argparse.ArgumentParser(description="Debug: show prompts and responses")
    parser.add_argument("--task", choices=TASKS, default=None,
                        help="Specific task (default: all)")
    parser.add_argument("--modality", choices=MODALITIES, default=None,
                        help="Specific modality (default: all)")
    parser.add_argument("--model-id", default="meta-llama/Llama-3.2-11B-Vision-Instruct")
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = args.data_root or os.path.join(repo_root, "data")

    tasks = [args.task] if args.task else TASKS
    modalities = [args.modality] if args.modality else MODALITIES

    # Load model once
    model, processor = load_model(args.model_id)

    for task in tasks:
        for modality in modalities:
            try:
                debug_one(task, modality, model, processor, data_root)
            except Exception as e:
                print(f"\n  ERROR: {task}/{modality}: {e}")

    print(f"\n{'='*80}")
    print("  Done.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
