import json
from radgraph import F1RadGraph


# =========================
# LOAD DATA (FIXED)
# =========================
def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)

    samples = data["predictions"]  # 🔥 important

    gts, gens = [], []

    for x in samples:
        ref = x.get("reference", "").replace("<unk>", "").strip()
        pred = x.get("prediction", "").replace("<unk>", "").strip()

        if len(ref) > 0 and len(pred) > 0:
            gts.append(ref)
            gens.append(pred)

    return gts, gens


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    RESULTS_PATH = "results/iu_xray/A2_outputs_10 epoch.json"

    # -------- LOAD DATA --------
    gts, gens = load_data(RESULTS_PATH)
    print(f"Loaded {len(gts)} samples")

    # Safety check
    if len(gts) == 0:
        raise ValueError("No valid samples found. Check JSON structure.")

    print("\nSample check:")
    print("GT:", gts[0])
    print("PRED:", gens[0])

    # =========================
    # RADGRAPH SCORER
    # =========================
    f1radgraph = F1RadGraph(
        reward_level="all",        # paper setting
        model_type="radgraph-xl"   # best model
    )

    # =========================
    # COMPUTE SCORES
    # =========================
    mean_reward, reward_list, hyp_ann, ref_ann = f1radgraph(
        hyps=gens,
        refs=gts
    )

    rg_e, rg_er, rg_full = mean_reward

    # =========================
    # RESULTS
    # =========================
    print("\n===== RadGraph F1 Scores =====")
    print(f"RG_E   (Entities F1):           {rg_e:.4f}")
    print(f"RG_ER  (Entities+Relations F1): {rg_er:.4f}")
    print(f"RG_ALL (Final RG-F1):           {rg_full:.4f}")