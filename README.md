# RRGE-RAG: Radiology Report Generation Enhancement using Hybrid Retrieval-Augmented Generation

## 📌 Project Status: In Progress

This repository contains the **ongoing implementation** of a hybrid Retrieval-Augmented Generation (RAG) framework for automated chest X-ray report generation.

The project extends the R2Gen architecture by integrating **multi-signal retrieval**, **structured clinical knowledge**, and **attention-based multimodal fusion** to improve the clinical accuracy and reliability of generated reports.

---

## 📌 Overview

Automated radiology report generation is a challenging task due to the need for **clinical accuracy and contextual understanding**. Traditional image-to-text models often suffer from **hallucinations and incomplete findings**.

To address this, our approach introduces a **Hybrid RAG framework** that:
- Retrieves similar prior cases
- Integrates structured clinical labels (Problems)
- Combines multimodal information for grounded report generation

---

## 📌 Current Progress

### ✅ Completed
- Dataset preprocessing and filtering (IU-Xray)
- Implementation of baseline models:
  - R2Gen
  - R2GenCMN
  - PPKED
  - METransformer
- Integration of **BioViL-T** as image encoder
- Implementation of **Problem (label) encoding** using multi-hot vectors
- Initial **Hybrid Retrieval Module**:
  - Image similarity (BioViL-T embeddings)
  - Text similarity (BiomedBERT embeddings)
  - Label similarity (multi-hot vectors)
- Attention-based **multimodal fusion mechanism**
- Training pipeline and evaluation setup
- Initial experimental results and ablation studies

---

### 🔄 Ongoing Work
- Optimization of hybrid retrieval weighting strategy
- Improvement of fusion mechanism
- Fine-tuning generation performance (BLEU / ROUGE)
- Calibration analysis using:
  - Expected Calibration Error (ECE)
  - Brier Score

---

### ⏳ Planned
- Retrieval efficiency improvement (e.g., FAISS indexing)
- Advanced fusion strategies
- Extensive hyperparameter tuning
- Final evaluation and comparative analysis
- Clinical validation and qualitative analysis

---

## 📌 Research Questions

**RQ1:** How does retrieval-augmented generation (RAG) influence the accuracy and clinical relevance compared to traditional image-to-text models?

**RQ2:** To what extent does multimodal fusion (image + labels + retrieval) improve report coherence and completeness?

**RQ3:** How effectively do uncertainty estimation mechanisms reflect the reliability of generated reports?

---

## 🎯 Objectives

- Generate accurate and coherent radiology reports
- Improve clinical relevance using structured labels (Problems)
- Reduce hallucinations through retrieval-based grounding
- Provide reliable predictions with uncertainty estimation

---

## 🧠 Methodology

### 🔹 Input
- Frontal and lateral chest X-ray images
- Structured clinical labels (Problems)

### 🔹 Core Components
- **Generator:** R2Gen
- **Image Encoder:** BioViL-T
- **Text Encoder:** BiomedBERT

### 🔹 Key Features
- Hybrid multi-signal retrieval (image + text + labels)
- Case-level retrieval (image + report + labels)
- Attention-based multimodal fusion
- Structured clinical knowledge integration
- Uncertainty estimation for reliability

---

## 📊 Dataset

**Indiana University Chest X-ray Dataset (IU-Xray)**  
🔗 https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university

### Dataset Details
- 3,851 patient records (original)
- Each includes:
  - Frontal and lateral images
  - Findings and Impression reports
  - Problems (clinical labels)

### Preprocessed Dataset
- Final size: **3,337 samples**
- Filtering criteria:
  - Both image views available
  - Non-empty Problems and Findings

---

## ⚙️ Pipeline Overview

1. Image encoding using BioViL-T  
2. Hybrid retrieval of top-K similar cases  
3. Multimodal fusion (image + retrieved data + labels)  
4. Report generation using R2Gen  
5. Confidence estimation for generated outputs  

---

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

### Ablation Study
- A0: Image only (R2Gen)
- A1: + Labels
- A2: + Retrieval
- A3: Full Hybrid RAG model

---

## 🚀 Current Findings (Preliminary)

- Retrieval improves **clinical grounding (RG-F1)**
- Slight trade-off observed in **lexical metrics (BLEU, ROUGE)**
- Multimodal fusion significantly outperforms label-only setups
- Retrieval helps reduce hallucination and improves factual consistency

---

## 🔮 Expected Contributions

- Hybrid retrieval framework for radiology report generation
- Improved clinical consistency using structured knowledge
- Reduction of hallucinations through grounded generation
- Reliable outputs via uncertainty-aware modeling

---

## 👨‍💻 Authors

- Vibolrottana Seng (st126425)  
- Zwe Yu Ya Kyaw Zin Oo (st125990)  
- Dakchhyeta Bade Shrestha (st126671)  
- Supipi Karunathilaka (st126489)  

---
