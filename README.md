# 🧠 Human vs AI Text Detector  
### _Text Mining Application in Cybersecurity_  
> **Master Big Data & IoT – ENSAM Casablanca (2024–2025)**  
> **By:** Ayoub Aamir  

---

## 🌐 Project Overview

In an era where **AI-generated content** blends seamlessly with human writing, the rise of *textual deepfakes* poses a major threat to **cybersecurity** and **digital forensics**.  
This project presents a **binary text classification system** capable of distinguishing **machine-generated** text from **human-written** content using advanced **Text Mining** and **Natural Language Processing (NLP)** techniques.

Our study replicates and extends the work from  
🧾 *“Is the Digital Forensics and Incident Response Pipeline Ready for Text-Based Threats in LLM Era?” (Bhandarkar et al., 2024)*

---

## 🎯 Objectives

- Detect and classify texts as **AI-generated** or **human-written**.  
- Compare the performance of traditional ML methods vs. Deep Learning approaches.  
- Evaluate robustness across two datasets (`AuTexTification` and `Yelp`).  
- Benchmark against state-of-the-art detection techniques in **Neural Text Detection (NTD)**.

---

## ⚙️ Technical Stack

| Category | Tools / Libraries |
|-----------|------------------|
| **Language** | Python 3.x |
| **Environment** | Visual Studio Code |
| **Data Processing** | `pandas`, `numpy`, `json`, `re`, `pathlib` |
| **Machine Learning** | `scikit-learn`, `joblib` |
| **Deep Learning** | `tensorflow`, `keras` |
| **NLP** | `nltk`, `transformers`, `textstat`, `lexicalrichness`, `empath` |

---

## 🧩 Architecture

### **Pipeline Steps**
1. **Preprocessing**
   - Cleaning, normalization, tokenization, lemmatization  
   - Feature extraction via **TF-IDF**, **Empath**, **Writeprints**, **LFI**
2. **Modeling**
   - Classical ML: Logistic Regression, Random Forest, SVM, Decision Tree  
   - Deep Learning: CNN for linguistic indicators (LFI)
3. **Evaluation**
   - Metrics: Precision, Recall, F1-score, Macro-F1, AUC-ROC  
4. **Deployment**
   - Models saved in `.joblib` and `.h5` formats  
   - Results exported as `.json` reports

---

## 🧠 Implemented Models

| Model | Features | Algorithm |
|--------|-----------|------------|
| **WriteprintsRFC** | Stylometric (10 features) | Random Forest |
| **LFI + CNN** | Linguistic & psychological (21 features) | Convolutional Neural Network |
| **Empath-DT** | Semantic (194 features) | Decision Tree |
| **N-Grams SVM** | Character 2–4-grams (TF-IDF) | SVM Linear |
| **GPT-2 Rank** | Probability-based token ranks | Logistic Regression |

---

## 📊 Results Summary

| Method | Dataset | Macro-F1 |
|---------|----------|----------|
| LIWC + DT | AuTexTification: 0.60 / Yelp: 0.73 |
| Writeprints + RFC | AuTexTification: 0.83 / Yelp: 0.59 |
| LFI + CNN | AuTexTification: **0.81** / Yelp: 0.67 |
| Log Likelihood | AuTexTification: 0.71 |
| GPT-2 Rank | AuTexTification: 0.34 |

**Key Insights**
- Hybrid models (LFI + CNN) outperform traditional ones by +21%.  
- Dataset characteristics strongly impact performance.  
- Statistical-only detectors (Rank) underperform against modern LLMs.

---

## 📈 Visual Results

- 📊 **Confusion matrices** comparing both datasets (train/val/test phases).  
- 📉 **Accuracy curves** showing model generalization capabilities.  
- 🧾 **F1-score analysis** illustrating dataset-specific behavior.

---

## ⚠️ Limitations

- **Hardware constraints** (CPU-only, no GPU acceleration).  
- **Limited availability** of high-quality open datasets for AI/human text detection.  
- **Generalization** still challenging for short or creative text segments.

---

## 🚀 Future Work

- Fine-tuning **Transformers** (e.g., RoBERTa, DeBERTa) on multi-domain datasets.  
- Integration of **multilingual** and **adversarially co-authored** corpora (e.g., FLAME).  
- Real-time detection API for cybersecurity systems.  
- Benchmark inclusion in **DFIR pipelines** for adaptive threat analysis.

---

## 📚 References

- [Bhandarkar et al., 2024 – *Is the DFIR Pipeline Ready for Text-Based Threats?*](https://arxiv.org/abs/2407.17870)  
- [He et al., 2023 – *MGTBench: Benchmarking Machine-Generated Text Detection*](https://arxiv.org/abs/2303.14822)  
- [Chakraborty et al., 2024 – *On the Possibilities of AI-Generated Text Detection*](https://proceedings.mlr.press/v235/chakraborty24a.html)

---

## 🧩 Repository Structure

```bash
📁 HumanVsAI-Text-Detector/
│
├── analysis/                
│   ├── decision_tree_comparison.csv   # Evaluation metrics comparing decision tree models
│   └── visualize_decision_tree.py     # Script to visualize trained decision trees
│
├── data/                        
│   ├── raw_data/                     # Original raw datasets
│   │   ├── dataset1/                 # Dataset1 (English text)
│   │   │   ├── train_en.csv          # Training set
│   │   │   ├── test_en.csv           # Test set
│   │   │   ├── validation.csv        # Validation set
│   │   │   └── new_test.csv          # Extra test set
│   │   └── dataset2/                 # Dataset2 (e.g., Yelp reviews)
│   │
│   ├── processed_data/               # Cleaned datasets ready for feature extraction
│   ├── tokenized_data/               # Tokenized datasets for NLP pipelines
│   ├── writeprints_features/         # Extracted Writeprints feature matrices
│   └── EMPATHdataset1/               # Extracted EMPATH features for Dataset1
│
├── models/                          # Trained models and results (outputs of training scripts)
│   ├── decision_tree/               
│   │   ├── dataset1/                # Decision tree models for Dataset1
│   │   │   ├── model.joblib         # Serialized trained model
│   │   │   ├── results.json         # Evaluation metrics (accuracy, F1-score, etc.)
│   │   │   └── tree_rules.txt       # Human-readable rules of the tree
│   │   └── dataset2/                # Decision tree models for Dataset2
│   │       ├── model.joblib
│   │       ├── results.json
│   │       └── tree_rules.txt
│   │
│   ├── writeprints_rfc/             # Random Forest Classifier using Writeprints features
│   │   ├── dataset1/                
│   │   │   ├── model.joblib
│   │   │   └── results.json
│   │   └── dataset2/                
│   │       ├── model.joblib
│   │       └── results.json
│   │
│   └── ...                          # Other classifiers (SVM, Logistic Regression, CNN, RankDetector, etc.)
│       ├── dataset1/
│       └── dataset2/
│
├── src/                             # Source code for pipelines
│   ├── data_loading/                
│   │   ├── load_raw_data.py         # Load Dataset1 from CSV
│   │   ├── load_raw_data_yelp.py    # Load Dataset2/Yelp
│   │   ├── split_data.py            # Split Dataset1 into train/validation/test
│   │   └── split_dataset2_yelp.py   # Split Dataset2 into train/test
│   │
│   ├── data_processing/             
│   │   ├── preprocessDataset1.py    # Clean and preprocess Dataset1
│   │   ├── preprocessedDataset2.py  # Clean and preprocess Dataset2
│   │   ├── lfi_feature_extractordataset1.py # Log-likelihood features for Dataset1
│   │   ├── lfi_feature_extractordataset2.py # Log-likelihood features for Dataset2
│   │   ├── liwc_features.py         # Extract LIWC features (Dataset1)
│   │   ├── liwc_features_dataset2.py # Extract LIWC features (Dataset2)
│   │   ├── writeprints_extractor.py # Generic Writeprints feature extractor
│   │   ├── writeprints_extractor_dataset1.py
│   │   └── writeprints_extractor_dataset2.py
│   │
│   ├── models/                      
│   │   ├── dataset1/                
│   │   │   ├── Train_Decision_Tree_dataset1.py
│   │   │   ├── train_Ifi_cnn_dataset1.py
│   │   │   ├── LogisticRegression_tfidf_dataset1.py
│   │   │   ├── RankDetector_dataset1.py
│   │   │   ├── SVM_classifierdataset1.py
│   │   │   └── train_writeprints_rfc_dataset1.py
│   │   │
│   │   └── dataset2/
│   │       ├── Train_Decision_Tree.py
│   │       ├── train_Ifi_cnn_dataset2.py
│   │       ├── LogisticRegression_tfidf_dataset2_yelp.py
│   │       ├── RankDetector_dataset2.py
│   │       └── train_writeprints_random_forest.py
│   │
│   └── utils/
│       └── paths.py                 # Centralized path management using pathlib
│
└── requirements.txt                 # Python dependencies
