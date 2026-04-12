import json
import os
from pathlib import Path
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_NAME = os.getenv("MODEL_NAME", "nateraw/vit-base-beans")
DATASET_NAME = os.getenv("DATASET_NAME", "beans")
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", "15"))

def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)

    split_name = "validation" if "validation" in dataset else "train"
    ds = dataset[split_name]
    num_samples = min(NUM_SAMPLES, len(ds))
    ds = ds.select(range(num_samples))

    print(f"Loading model: {MODEL_NAME}")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()

    id2label = model.config.id2label
    rows = []
    correct = 0

    for idx, item in enumerate(ds):
        image = item["image"]
        true_label_id = int(item["labels"])
        true_label = id2label[true_label_id]
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            pred_id = int(outputs.logits.argmax(dim=-1).item())
        pred_label = id2label[pred_id]
        is_correct = pred_id == true_label_id
        correct += int(is_correct)
        rows.append(
            {"index": idx,
            "true_label_id": true_label_id,
            "true_label": true_label,
            "pred_label_id": pred_id,
            "pred_label": pred_label,
            "correct": is_correct})

    acc = correct / num_samples if num_samples else 0.0
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "predictions.csv", index=False)

    summary = {
        "model_name": MODEL_NAME,
        "dataset_name": DATASET_NAME,
        "split": split_name,
        "num_samples": num_samples,
        "accuracy_on_sample": acc,
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / "run_log.txt", "w", encoding="utf-8") as f:
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Dataset: {DATASET_NAME}\n")
        f.write(f"Split: {split_name}\n")
        f.write(f"# of samples: {num_samples}\n")
        f.write(f"Accuracy on sample: {acc:.4f}\n")

    print(f"Done. Sample accuracy: {acc:.4f}")
    print("Results saved in outputs/")

if __name__ == "__main__":
    main()
