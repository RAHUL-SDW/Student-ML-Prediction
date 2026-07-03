# 🎓 Adaptive Meta-Cognitive Hybrid AI Framework
## Student Dropout & Placement Risk Prediction

A college-level ML project with a genuine novel contribution: a **Meta-Cognitive Evaluation Module** that assesses the *reliability* of every prediction.

---

## 📁 Project Structure

```
student_ml_project/
│
├── app.py                      ← Streamlit entry point
├── generate_dataset.py         ← Synthetic dataset generator (run once)
├── requirements.txt
│
├── modules/
│   ├── preprocessing.py        ← Clean → Encode → Feature-engineer → Scale
│   ├── model_trainer.py        ← Train 4 models, pick best by F1
│   ├── meta_cognitive.py       ← 🔑 Novel: 5-score reliability evaluator
│   ├── recommender.py          ← Rule-based recommendation engine
│   └── eda.py                  ← Plotly chart helpers
│
├── pages/
│   ├── 1_🏠_Home.py
│   ├── 2_📊_Dataset_Analysis.py
│   ├── 3_🤖_Train_Model.py
│   ├── 4_👤_Single_Prediction.py
│   └── 5_📂_Upload_Dataset.py
│
├── data/
│   └── students_raw.csv        ← Auto-generated (800 students)
│
└── models/
    ├── dropout_best.pkl        ← Best dropout model
    ├── placement_best.pkl      ← Best placement model
    ├── scaler.pkl              ← Fitted StandardScaler
    └── feat_cols.txt           ← Feature column names
```

---

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dataset + pre-train models (optional — app does this automatically)
python generate_dataset.py

# 3. Launch the app
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---


## 🤖 Models Compared (6 total)

| Model | Type |
|-------|------|
| Logistic Regression | Linear baseline |
| Decision Tree | Interpretable rules |
| Random Forest | Bagging ensemble |
| XGBoost | Gradient boosting |
| **Hard Voting** | Majority vote across all 4 base models |
| **Soft Voting** | Averages predicted probabilities across all 4 base models |

Best model selected automatically by **F1 Score**. The Train Model page shows a full comparison table (all 6 models, colour-coded) plus per-model confusion matrices.

---

## 📓 Two Ways to Run This Project

### Option A — Streamlit Web App
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Option B — Jupyter Notebook (`Student_ML_Project.ipynb`)
Self-contained, runs top-to-bottom with no Streamlit dependency — ideal for viva/demo or submitting as a standalone deliverable.
```bash
pip install jupyter scikit-learn xgboost pandas numpy matplotlib seaborn
jupyter notebook Student_ML_Project.ipynb
```
Already pre-executed with all outputs (charts, tables, predictions) saved inside, so it can be viewed directly without re-running.

---

## 📊 Features Used

**Academic:** CGPA, Attendance, Semester GPA, Internal Marks, Backlogs  
**Skills:** Coding, Communication, Technical, Aptitude, English Proficiency  
**Experience:** Internship, Projects, Certifications  
**Engineered:** Academic Consistency, Skill Readiness, Overall Performance Score

---

## 🔮 Prediction Modes

1. **Sample Dataset** — built-in 800-student dataset with EDA  
2. **Single Student** — form-based prediction with meta-cognitive scores & recommendations  
3. **Batch Upload** — upload CSV, get bulk predictions, download results  

---
