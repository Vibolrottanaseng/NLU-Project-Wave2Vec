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
def load_chexbert(checkpoint_path):
    import sys
    sys.path.append("./CheXbert/src/")  # adjust if needed

    from models.bert_labeler import bert_labeler

    model = bert_labeler()
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    new_state_dict = OrderedDict()
    for k, v in checkpoint["model_state_dict"].items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    model.cuda()

    print(f"Loaded CheXbert from {checkpoint_path}")
    return model


# =========================
# TOKENIZER
# =========================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# =========================
# GET LABELS + PROBABILITIES
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

                pred = torch.argmax(soft, dim=1).item()
                prob_pos = soft[:, 1].item()  # probability of "positive"

                preds.append(pred)
                probs.append(prob_pos)

            all_preds.append(preds)
            all_probs.append(probs)

    return np.array(all_preds), np.array(all_probs)


# =========================
# ECE
# =========================
def compute_ece(y_true, y_prob, n_bins=10):
    y_true = y_true.flatten()
    y_prob = y_prob.flatten()

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(y_true)

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]

        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)

        if np.sum(in_bin) > 0:
            acc = np.mean(y_true[in_bin])
            conf = np.mean(y_prob[in_bin])
            ece += (np.sum(in_bin) / N) * np.abs(acc - conf)

    return ece


# =========================
# BRIER SCORE
# =========================
def compute_brier_score(y_true, y_prob):
    return np.mean((y_prob - y_true) ** 2)


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    # -------- PATHS --------
    RESULTS_PATH = "results/iu_xray/test_results_best.json"
    CHEXBERT_PATH = "chexbert.pth"

    # -------- LOAD DATA --------
    with open(RESULTS_PATH, "r") as f:
        data = json.load(f)

    gts = [x["ground_truth"] for x in data]
    gens = [x["generated"] for x in data]

    print(f"Loaded {len(gts)} samples")

    # -------- DEVICE --------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- LOAD MODEL --------
    model = load_chexbert(CHEXBERT_PATH)

    # -------- EXTRACT LABELS --------
    print("Extracting labels and probabilities...")
    gt_labels, _ = get_labels_and_probs(model, gts, device)
    gen_labels, gen_probs = get_labels_and_probs(model, gens, device)

    print("Label shapes:", gt_labels.shape, gen_labels.shape)

    # -------- CONVERT TO BINARY --------
    gt_binary = (gt_labels == 1).astype(int)
    gen_binary = (gen_labels == 1).astype(int)

    # =========================
    # F1 SCORES
    # =========================
    macro_f1 = f1_score(gt_binary, gen_binary, average="macro")
    micro_f1 = f1_score(gt_binary, gen_binary, average="micro")

    # Top-5 conditions
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