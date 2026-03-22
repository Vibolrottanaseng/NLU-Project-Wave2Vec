
# 🩺 RRGE-RAG:Radiology Report Generation Enhancement Using Hybrid Retrieval Augmented Generation

## 📌 Overview
This project proposes a **hybrid Retrieval-Augmented Generation (RAG) framework** for automated chest X-ray report generation. The system integrates **visual features, structured clinical knowledge, and retrieved case-based evidence** to generate clinically relevant radiology reports.

The proposed approach builds upon **R2Gen** as the report generation backbone and enhances it with **multi-signal retrieval and attention-based multimodal fusion**.

---

📌 Research Questions

RQ1:
How does the integration of retrieval-augmented generation (RAG) influence the accuracy and clinical relevance of automated radiology report generation compared to traditional image-to-text models?

RQ2:
To what extent does multimodal fusion of X-ray images, structured clinical labels (Problems), and retrieved case information improve the coherence and completeness of generated radiology reports?

RQ3:
How effectively can uncertainty estimation mechanisms reflect the reliability of generated radiology reports in clinical decision-support settings?

---

## 🎯 Objectives
- Generate accurate and coherent radiology reports from chest X-ray images  
- Improve clinical relevance using structured labels (*Problems*)  
- Leverage retrieval-based evidence to reduce hallucinations  
- Provide a foundation for reliable clinical decision support  

---

## 🧠 Proposed Method

### 🔹 Input
- Frontal and lateral chest X-ray images  
- Structured clinical labels (*Problems column*)  

### 🔹 Core Components
- **Generator:** R2Gen  
- **Image Encoder:** BioViL-T  
- **Text Encoder:** BiomedBERT (PubMedBERT-style embeddings)  

### 🔹 Key Features
- Hybrid multi-signal retrieval (image + text + labels)  
- Case-level retrieval (image + report + labels together)  
- Attention-based multimodal fusion  
- Structured clinical knowledge integration  
- Uncertainty estimation for reliability  

---

## 📊 Dataset
- **Dataset:** Indiana University Chest X-ray Dataset (IU-Xray)  
- **Source:** https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university  

### Dataset Details
- 3,851 patient records (initial)  
- Each patient includes:
  - Frontal and lateral X-ray images  
  - Clinical report sections (Findings, Impression, etc.)  
  - Problems (structured labels)

### Preprocessed Dataset
- Filtered dataset: **3,337 samples**
- Criteria:
  - Both image views available  
  - Non-empty *Problems* and *Findings*  

---

## ⚙️ Methodology Overview
1. Image feature extraction using BioViL-T  
2. Hybrid retrieval of similar cases  
3. Multimodal fusion (image + retrieved data + labels)  
4. Report generation using R2Gen  
5. Confidence estimation for outputs  

---

## 📈 Evaluation Plan

### Metrics
- BLEU-2 (BL-2)  
- BLEU-3 (BL-3)  
- BLEU-4 (BL-4)  
- ROUGE-L  

### Baselines
- R2Gen  
- R2GenCMN  
- PPKED  
- METransformer  

### Ablation Study
- A0: R2Gen (Image only)  
- A1: + Problems  
- A2: + Retrieval  
- A3: Full Model  

---

## 🚀 Expected Contributions
- Improved report quality using retrieval-augmented generation  
- Better clinical consistency through structured knowledge  
- Reduced hallucination via grounded generation  
- Reliable outputs with uncertainty estimation  

---

## 👨‍💻 Authors
- *Vibolrottana Seng	- st126425*
- *Zwe Yu Ya Kyaw Zin Oo	st125990*
- *Dakchhyeta Bade Shrestha	st126671*
- *Supipi Karunathilaka	st126489*

---

