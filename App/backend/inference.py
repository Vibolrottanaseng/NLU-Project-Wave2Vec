import os
import json
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, "app_model_files", "val_generated_epoch_30.json")

generated_data = None


def load_generated_outputs():
    global generated_data

    if generated_data is not None:
        return generated_data

    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(f"Generated output JSON not found: {JSON_PATH}")

    with open(JSON_PATH, "r", encoding="utf-8") as file:
        generated_data = json.load(file)

    return generated_data


def generate_report(frontal_path, lateral_path=None, problems=None):
    data = load_generated_outputs()
    uploaded_name = os.path.basename(frontal_path)

    for item in data:
        image_id = str(item.get("id", ""))

        if image_id and image_id in uploaded_name:
            return {
                "report": item.get("generated", "No generated report found."),
                "reference": item.get("ground_truth", "No reference report found."),
                "matched_id": image_id
            }

    sample = random.choice(data)

    return {
        "report": sample.get("generated", "No generated report found."),
        "reference": sample.get("ground_truth", "No reference report found."),
        "matched_id": str(sample.get("id", "random_sample"))
    }