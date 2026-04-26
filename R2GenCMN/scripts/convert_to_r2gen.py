import pandas as pd
import json
import os


def convert_to_r2gen_format(csv_path, output_json, image_root):
    """
    Convert your CSV split into R2GenCMN JSON format.

    Args:
        csv_path (str): Path to input CSV file
        output_json (str): Path to save JSON file
        image_root (str): Folder where images are stored (relative path used in JSON)
    """

    print(f"\nProcessing: {csv_path}")

    df = pd.read_csv(csv_path)

    data = []
    skipped = 0

    for idx, row in df.iterrows():
        try:
            uid = str(row["uid"])

            # Handle missing values safely
            if pd.isna(row["frontal_path"]) or pd.isna(row["lateral_path"]):
                skipped += 1
                continue

            if pd.isna(row["impression_clean"]):
                skipped += 1
                continue

            # Extract filenames only (convert absolute → relative)
            #frontal_file = os.path.basename(row["frontal_path"]).replace("\\", "/")
            #lateral_file = os.path.basename(row["lateral_path"]).replace("\\", "/")

            frontal = os.path.basename(row["frontal_path"])
            lateral = os.path.basename(row["lateral_path"])

            report = str(row["impression_clean"]).strip()

            # Clean punctuation issues
            report = report.replace("..", ".")
            report = report.replace(" .", ".")
            report = " ".join(report.split())

            # Skip empty reports
            if len(report) == 0:
                skipped += 1
                continue

            sample = {
                "id": uid,
                "image_path": [frontal, lateral],
                "report": report
            }

            data.append(sample)

        except Exception as e:
            print(f"Error at row {idx}: {e}")
            skipped += 1

    # Save JSON
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Saved: {output_json}")
    print(f"Total samples: {len(data)}")
    print(f"Skipped rows: {skipped}")


# ============================
# MAIN EXECUTION
# ============================
if __name__ == "__main__":

    # 🔁 Update these paths based on your project
    TRAIN_CSV = "dataset/processed/train.csv"
    VAL_CSV = "dataset/processed/val.csv"
    TEST_CSV = "dataset/processed/test.csv"

    OUTPUT_DIR = "R2GenCMN/data/iu_xray"

    # This should match your image folder used during training
    IMAGE_ROOT = "R2GenCMN/data/iu_xray/images"

    convert_to_r2gen_format(
        TRAIN_CSV,
        os.path.join(OUTPUT_DIR, "train.json"),
        IMAGE_ROOT
    )

    convert_to_r2gen_format(
        VAL_CSV,
        os.path.join(OUTPUT_DIR, "val.json"),
        IMAGE_ROOT
    )

    convert_to_r2gen_format(
        TEST_CSV,
        os.path.join(OUTPUT_DIR, "test.json"),
        IMAGE_ROOT
    )

    print("\n✅ Conversion completed successfully!")