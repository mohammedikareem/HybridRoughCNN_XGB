# Hybrid Rough Set + CNN + XGBoost Framework

This repository implements a **hybrid intelligent framework** for ransomware traffic classification.
It combines:
- 🧩 **Rough Set feature selection** for dimensionality reduction
- 🧠 **1D CNN** for deep feature extraction
- 🚀 **XGBoost** for final classification

## 📚 Pipeline Overview
1. Data preprocessing & encoding  
2. Rough Set-based feature selection  
3. CNN feature extraction  
4. XGBoost classification  
5. Evaluation & visualization  

## ⚙️ Usage
```bash
pip install -r requirements.txt
python src/main_pipeline.py
```

## 📊 Output
- Confusion Matrix  
- Feature Importance (XGBoost)  
- Accuracy & Classification Report  
- Saved pipeline: `models/hybrid_model_pipeline.pkl`

## 🧠 Dataset
Dataset: `final(2).csv` (Ransomware dataset, local or Google Drive)

## 🧾 Citation
If you use this project, please cite:
**Mohammed Ibrahim Kareem (2025)**, *Hybrid Rough Set + CNN + XGBoost Pipeline for Ransomware Detection.*
