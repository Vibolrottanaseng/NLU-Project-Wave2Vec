import json
import torch
import numpy as np
from collections import OrderedDict
from transformers import BertTokenizer
from sklearn.metrics import f1_score
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

    print(f"Loaded CheXbert from {checkpoint_path}")
    return model


# =========================
# TOKENIZER
# =========================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# =========================
# LOAD DATA (FIXED)
# =========================
def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)

    # 🔥 IMPORTANT: your data is inside "predictions"
    samples = data["predictions"]

    gts, gens = [], []

    for x in samples:
        ref = x.get("reference", "").replace("<unk>", "").strip()
        pred = x.get("prediction", "").replace("<unk>", "").strip()

        if len(ref) > 0 and len(pred) > 0:
            gts.append(ref)
            gens.append(pred)

    return gts, gens


# =========================
# GET LABELS + PROBS
# =========================
def get_labels_and_probs(model, texts, device):
    all_preds = []
    all_probs = []

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
            probs = []

            for out in outputs:
                soft = F.softmax(out, dim=1)

                preds.append(torch.argmax(soft, dim=1).item())
                probs.append(soft[:, 1].item())  # positive prob

            all_preds.append(preds)
            all_probs.append(probs)

    return np.array(all_preds), np.array(all_probs)


# =========================
# METRICS
# =========================
def compute_ece(y_true, y_prob, n_bins=10):
    y_true = y_true.flatten()
    y_prob = y_prob.flatten()

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(y_true)

    for i in range(n_bins):
        in_bin = (y_prob > bins[i]) & (y_prob <= bins[i + 1])

        if np.sum(in_bin) > 0:
            acc = np.mean(y_true[in_bin])
            conf = np.mean(y_prob[in_bin])
            ece += (np.sum(in_bin) / N) * np.abs(acc - conf)

    return ece


def compute_brier_score(y_true, y_prob):
    return np.mean((y_prob - y_true) ** 2)


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    RESULTS_PATH = "results/iu_xray/test_prediction_r2gen.json"
    CHEXBERT_PATH = "chexbert.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- LOAD DATA --------
    gts, gens = load_data(RESULTS_PATH)
    print(f"Loaded {len(gts)} samples")

    # 🚨 Safety check
    if len(gts) == 0:
        raise ValueError("No valid samples found. Check JSON structure.")

    print("\nSample check:")
    print("GT:", gts[0])
    print("PRED:", gens[0])

    # -------- LOAD MODEL --------
    model = load_chexbert(CHEXBERT_PATH, device)

    # -------- LABEL EXTRACTION --------
    print("\nExtracting CheXbert labels...")
    gt_labels, _ = get_labels_and_probs(model, gts, device)
    gen_labels, gen_probs = get_labels_and_probs(model, gens, device)

    print("Shape:", gt_labels.shape)

    # -------- STANDARD BINARIZATION --------
    gt_binary = (gt_labels == 1).astype(int)
    gen_binary = (gen_labels == 1).astype(int)

    # =========================
    # F1 SCORES
    # =========================
    macro_f1 = f1_score(gt_binary, gen_binary, average="macro")
    micro_f1 = f1_score(gt_binary, gen_binary, average="micro")

    # Top-5 labels
    top5_idx = [0, 1, 2, 3, 4]

    macro_f1_5 = f1_score(gt_binary[:, top5_idx], gen_binary[:, top5_idx], average="macro")
    micro_f1_5 = f1_score(gt_binary[:, top5_idx], gen_binary[:, top5_idx], average="micro")

    # =========================
    # UNCERTAINTY METRICS
    # =========================
    brier = compute_brier_score(gt_binary, gen_probs)
    ece = compute_ece(gt_binary, gen_probs)

    # =========================
    # RESULTS
    # =========================
    print("\n===== CheXbert Evaluation =====")
    print(f"14-label Macro F1: {macro_f1:.4f}")
    print(f"14-label Micro F1: {micro_f1:.4f}")
    print(f"5-label Macro F1:  {macro_f1_5:.4f}")
    print(f"5-label Micro F1:  {micro_f1_5:.4f}")

    print("\n===== Uncertainty Metrics =====")
    print(f"Brier Score: {brier:.4f}")
    print(f"ECE:         {ece:.4f}")