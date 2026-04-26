import json
import os

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def merge_splits(train_path, val_path, test_path, output_path):
    print("Loading files...")

    train_data = load_json(train_path)
    val_data = load_json(val_path)
    test_data = load_json(test_path)

    merged = {
        "train": train_data,
        "val": val_data,
        "test": test_data
    }

    save_json(merged, output_path)

    print(f"\n✅ annotation.json created at: {output_path}")
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")

if __name__ == "__main__":
    merge_splits(
        "R2GenCMN/data/iu_xray/train.json",
        "R2GenCMN/data/iu_xray/val.json",
        "R2GenCMN/data/iu_xray/test.json",
        "R2GenCMN/data/iu_xray/annotation.json"
    )