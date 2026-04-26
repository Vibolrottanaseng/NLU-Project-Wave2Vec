
import json
import torch
import numpy as np
from collections import OrderedDict
from transformers import BertTokenizer
from sklearn.metrics import f1_score


# =========================
# LOAD CHEXBERT MODEL
# =========================
def load_chexbert(checkpoint_path):
    import sys
    sys.path.append("./CheXbert/src/")  # adjust path if needed

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
# GET LABELS FROM TEXT
# =========================
def get_labels(model, texts, device):
    all_outputs = []

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
            # outputs = list of 14 tensors

            preds = []
            for out in outputs:
                pred = torch.argmax(out, dim=1)  # (batch,)
                preds.append(pred.cpu().item())

            all_outputs.append(preds)

    return np.array(all_outputs)


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    # -------- FILE PATHS --------
    RESULTS_PATH = "results/iu_xray/test_prediction_r2gen.json"
    CHEXBERT_PATH = "chexbert.pth"

    # -------- LOAD DATA --------
    with open(RESULTS_PATH, "r") as f:
        data = json.load(f)

    gts = [x["ground_truth"] for x in data]
    gens = [x["prediction"] for x in data]

    print(f"Loaded {len(gts)} samples")

    # -------- DEVICE --------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- LOAD MODEL --------
    model = load_chexbert(CHEXBERT_PATH)

    # -------- GET LABELS --------
    print("Extracting labels...")
    gt_labels = get_labels(model, gts, device)
    gen_labels = get_labels(model, gens, device)

    print("Label shapes:", gt_labels.shape, gen_labels.shape)

    # -------- CONVERT TO BINARY --------
    # Only "positive" (1) is considered
    gt_labels = (gt_labels == 1).astype(int)
    gen_labels = (gen_labels == 1).astype(int)

    # -------- COMPUTE F1 --------
    macro_f1 = f1_score(gt_labels, gen_labels, average="macro")
    micro_f1 = f1_score(gt_labels, gen_labels, average="micro")

    # Top-5 conditions (same as paper)
    top5_idx = [0, 1, 2, 3, 4]

    macro_f1_5 = f1_score(gt_labels[:, top5_idx], gen_labels[:, top5_idx], average="macro")
    micro_f1_5 = f1_score(gt_labels[:, top5_idx], gen_labels[:, top5_idx], average="micro")

    # -------- RESULTS --------
    print("\n===== CheXbert Evaluation =====")
    print(f"14-label Macro F1: {macro_f1:.4f}")
    print(f"14-label Micro F1: {micro_f1:.4f}")
    print(f"5-label Macro F1:  {macro_f1_5:.4f}")
    print(f"5-label Micro F1:  {micro_f1_5:.4f}")
