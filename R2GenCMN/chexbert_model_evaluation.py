import json
import torch
import numpy as np
from collections import OrderedDict
from transformers import BertTokenizer
import torch.nn.functional as F


# =========================
# LOAD CHEXBERT MODEL
# =========================
def load_chexbert(checkpoint_path, device):
    import sys
    sys.path.append("./CheXbert/src/")
    from models.bert_labeler import bert_labeler

    model = bert_labeler()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    new_state_dict = OrderedDict()
    for k, v in checkpoint["model_state_dict"].items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    return model


# =========================
# TOKENIZER
# =========================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# =========================
# LOAD DATA
# =========================
def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)

    samples = data["predictions"]

    gts, gens = [], []

    for x in samples:
        ref = x.get("reference", "").replace("<unk>", "").strip()
        pred = x.get("prediction", "").replace("<unk>", "").strip()

        if ref and pred:
            gts.append(ref)
            gens.append(pred)

    return gts, gens


# =========================
# GET LABELS
# =========================
def get_labels(model, texts, device):
    all_preds = []

    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512
            )

            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask)

            preds = []
            for out in outputs:
                pred = torch.argmax(out, dim=1).item()
                preds.append(pred)

            all_preds.append(preds)

    return np.array(all_preds)


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    RESULTS_PATH = "results/iu_xray/A2_outputs_10 epoch.json"
    CHEXBERT_PATH = "chexbert.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    gts, gens = load_data(RESULTS_PATH)
    print(f"Loaded {len(gts)} samples")

    # Load model
    model = load_chexbert(CHEXBERT_PATH, device)

    # Get labels
    print("Extracting labels...")
    gt_labels = get_labels(model, gts, device)
    gen_labels = get_labels(model, gens, device)

    # Convert to binary (standard)
    gt_binary = (gt_labels == 1).astype(int)
    gen_binary = (gen_labels == 1).astype(int)

    # =========================
    # ACCURACY
    # =========================
    total = gt_binary.size
    correct = (gt_binary == gen_binary).sum()

    accuracy = correct / total

    print("\n===== CheXbert Agreement Accuracy =====")
    print(f"Accuracy: {accuracy:.4f}")