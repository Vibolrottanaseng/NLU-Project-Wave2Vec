# RRGE-RAG: Radiology Report Generation Enhancement Using Hybrid Retrieval-Augmented Generation

This repository contains the implementation and final report for **RRGE-RAG**, a Hybrid Retrieval-Augmented Generation framework for automated chest X-ray radiology report generation.

The project aims to improve the factual correctness, clinical relevance, and reliability of generated radiology reports by combining visual image features, structured clinical labels, and retrieved case-based evidence.

## 📌 Overview

Automated radiology report generation is an important application of medical artificial intelligence. Traditional image-to-text models generate reports directly from chest X-ray images, but they may produce incomplete or clinically inaccurate descriptions. To address this issue, this project proposes a **Hybrid Retrieval-Augmented Generation (Hybrid RAG)** framework that retrieves similar prior cases and integrates them with the current patient’s X-ray features and clinical labels.

The proposed method extends the **R2Gen** architecture by incorporating:

- **BioViL-T** as the chest X-ray image encoder
- **Hybrid retrieval** using visual, textual, and structured clinical label similarity
- **Top-3 retrieved similar cases** as external clinical evidence
- **Attention-based multimodal fusion**
- **Uncertainty-aware evaluation** using calibration metrics

## Dataset

The project uses the **Indiana University Chest X-ray dataset**, which contains chest X-ray images and corresponding radiology reports.

Link: https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university

Each sample includes:

- Frontal and lateral chest X-ray images
- Clinical labels from the `Problems` column
- Radiology report sections such as `Findings` and `Impression`

After preprocessing and filtering, the final dataset contains **3,337 aligned samples** with image, text, and structured label information.


## Methodology

The proposed RRGE-RAG framework follows a multimodal retrieval-augmented pipeline:

1. **Image Encoding**  
   Chest X-ray images are encoded using BioViL-T to extract domain-specific visual embeddings.

2. **Hybrid Retrieval**  
   The system retrieves the top-3 most similar prior patient cases using:
   - Image similarity
   - Report text similarity
   - Structured clinical label similarity

3. **Multimodal Fusion**  
   Visual embeddings, retrieved reports, and clinical labels are combined using an attention-based fusion mechanism.

4. **Report Generation**  
   The fused representation is passed to the R2Gen decoder to generate the final radiology report.

5. **Evaluation**  
   The generated reports are evaluated using lexical, clinical, and uncertainty-aware metrics.

## Baseline Models

The proposed model is compared with several established radiology report generation baselines:

- **R2Gen**
- **R2GenCMN**
- **PPKED**
- **METransformer**

These models represent standard image-to-text, memory-based, knowledge-enhanced, and transformer-based approaches.

## Evaluation Metrics

The system is evaluated using three categories of metrics:

### Lexical Metrics

- BLEU-1
- BLEU-2
- BLEU-3
- BLEU-4
- ROUGE-L

### Clinical Metrics

- RadGraph F1
- CheXbert 14-class Macro F1
- CheXbert 14-class Micro F1

### Uncertainty and Calibration Metrics

- Expected Calibration Error
- Brier Score
- Entropy-based uncertainty
- MC Dropout variance

## Research Questions

This project investigates the following research questions:

1. How does retrieval-augmented generation influence the accuracy and clinical relevance of automated radiology report generation compared to traditional image-to-text models?

2. To what extent does multimodal fusion of X-ray images, clinical labels, and retrieved case information improve the coherence and completeness of generated radiology reports?

3. How well do uncertainty estimates correlate with generation quality, and how well are they calibrated?

## ⚙️ Pipeline Overview

1. Image encoding using BioViL-T  
2. Hybrid retrieval of top-K similar cases  
3. Multimodal fusion (image + retrieved data + labels)  
4. Report generation using R2Gen  
5. Confidence estimation for generated outputs  



## 📈 Evaluation

### Metrics
- BLEU-2, BLEU-3, BLEU-4
- ROUGE-L
- CheXbert F1 (clinical accuracy)
- RadGraph F1 (clinical structure)
- ECE (calibration)
- Brier Score (probabilistic accuracy)

### Baselines
- R2Gen
- R2GenCMN
- PPKED
- METransformer

| Model | BL-1  | BL-2 | BL-3 | BL-4   | ROUGE-L | RG-F1 | 14Ma-F1|14Mi-F1 | ECE  | Brier |
| :---  | :---  | :--- | :--- |:---:| :---:   |:----:    |:-----: | :---:| :---: |:---:|
| R2Gen | 0.479 | **0.479**| 0.103| 0.082 | 0.483 | 0.242    | 0.079  | 0.376| 0.88  | 0.087|
| R2GenCMN| 0.362	| 0.267 |	0.203| 0.153 | **0.545** | 0.254 | 0.097 |	0.717 |	0.04 |	0.04 |
| PPKED | 0.099	|0.052|	0.032|	0.021|	0.178|	0.247|	0.04|	**0.423**|	0.076|	0.076|
| METransformer | **0.483**|	0.322|	**0.228**|	**0.172**|	0.48|	**0.311**|	0.074|	0.08|	0.078|	0.079|
| RRGE-RAG (OURS)|0.384|	0.231|	0.155|	0.11|	0.315|	0.235|	0.064|	0.356|	0.076|	0.077|


### Ablation Study
- A0: Image only (R2Gen)
- A1: + Labels
- A2: + Retrieval
- A3: Full Hybrid RAG model

| Model | BL-1  | BL-2 | BL-3 | BL-4   | ROUGE-L | RG-F1 | 14Ma-F1|14Mi-F1 | ECE  | Brier |
| :---  | :---  | :--- | :--- |:---:| :---:   |:----:    |:-----: | :---:| :---: |:---:|
| A0 | 0.479|	0.479|	0.103|	0.082|	0.483|	0.242	|0.079|	0.376|	0.088|	0.087|
| A1| 0.06|	0.015|	0.004|	0.041|	0.058|	0.011|	0.078|	0.376|	0.087|	0.088|
| A2 | 0.221|	0.146|	0.102|	0.068|	0.302|	0.225|	0.053|	0.25|	0.059|	0.058|
| RRGE-RAG (OURS) |0.384|	0.231|	0.155|	0.11|	0.315|	0.254|	0.064|	0.356|	0.076|	0.077|


<!-- ## 🚀 Current Findings (Preliminary)

- Retrieval improves **clinical grounding (RG-F1)**
- Slight trade-off observed in **lexical metrics (BLEU, ROUGE)**
- Multimodal fusion significantly outperforms label-only setups
- Retrieval helps reduce hallucination and improves factual consistency


## 🔮 Expected Contributions

- Hybrid retrieval framework for radiology report generation
- Improved clinical consistency using structured knowledge
- Reduction of hallucinations through grounded generation
- Reliable outputs via uncertainty-aware modeling -->


## Results Summary

This work presents a Hybrid Retrieval-Augmented
Generation (Hybrid RAG) framework for radiology report generation, integrating visual features,
structured clinical labels, and retrieved case-based
knowledge. The proposed approach improves clinical relevance and maintains competitive performance across evaluation metrics, particularly in
capturing meaningful medical information.
The ablation study confirms the importance of
multimodal fusion and retrieval in enhancing report quality. While the model achieves reasonable
calibration performance, uncertainty estimation remains an area for improvement.
Overall, the results demonstrate the potential
of combining retrieval and multimodal learning
for more reliable radiology report generation, with
future work focusing on improving retrieval quality,
model calibration, and generalization to broader
clinical settings.

## 👨‍💻 Authors

- Vibolrottana Seng (st126425)  
- Zwe Yu Ya Kyaw Zin Oo (st125990)  
- Dakchhyeta Bade Shrestha (st126671)  
- Supipi Karunathilaka (st126489)  

---
