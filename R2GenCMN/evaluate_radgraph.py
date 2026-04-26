import json
from radgraph import F1RadGraph

# ===== Load your results =====
with open("results/iu_xray/A2_outputs_10 epoch.json", "r") as f:
    data = json.load(f)

gts = [item["ground_truth"] for item in data]
gens = [item["generated"] for item in data]

print(f"Loaded {len(gts)} samples")

# ===== Initialize RadGraph scorer =====
f1radgraph = F1RadGraph(
    reward_level="all",        # IMPORTANT (paper setting)
    model_type="radgraph-xl"   # best model (recommended)
)

# ===== Compute scores =====
mean_reward, reward_list, hyp_ann, ref_ann = f1radgraph(
    hyps=gens,
    refs=gts
)

# ===== Extract scores =====
rg_e, rg_er, rg_full = mean_reward

print("\n===== RadGraph F1 Scores =====")
print(f"RG_E   (Entities F1): {rg_e:.4f}")
print(f"RG_ER  (Entities+Relations F1): {rg_er:.4f}")
print(f"RG_ALL (Final RG-F1): {rg_full:.4f}")