import json
from bert_score import score

# Load your saved results
with open("results/iu_xray/test_results_best.json", "r") as f:
    data = json.load(f)

# Extract predictions and references
preds = [item["generated"] for item in data]
refs = [item["ground_truth"] for item in data]

# Compute BERTScore
P, R, F1 = score(preds, refs, lang="en", verbose=True)

# Print average scores
print("BERTScore Results:")
print(f"Precision: {P.mean().item():.4f}")
print(f"Recall:    {R.mean().item():.4f}")
print(f"F1:        {F1.mean().item():.4f}")