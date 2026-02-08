# Email Spam Filter Using Machine Learning

## Project Overview
This project builds a beginner-friendly yet industry-aligned email spam detection system using classic NLP and machine learning techniques. It supports real-time email text input, predicts spam vs. not spam, displays confidence, and highlights phishing keywords for extra safety.

## Functional Features
- Classify emails as **Spam** or **Not Spam**
- Detect phishing and malicious content via keyword highlighting
- Real-time text input with prediction confidence
- Model comparison between **Naive Bayes** and **SVM**

## Tech Stack
- **Python 3**
- **Pandas / NumPy** for data handling
- **NLTK** for text preprocessing (tokenization, stop-word removal, lemmatization)
- **Scikit-learn** for TF-IDF vectorization and modeling
- **Streamlit** for the web app

## Architecture Diagram (ASCII)
```
User Input
    |
    v
[Streamlit UI] --> preprocess_text() --> TF-IDF Vectorizer --> Model (NB / SVM)
    |                                                           |
    |-------------------- confidence + label -------------------|
    |
    v
Phishing Keyword Highlighter
```

## Project Structure
```
email-spam-filter/
├── data/
│   └── spam.csv
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── app.py
├── model/
│   └── spam_model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## Dataset
Use a public labeled dataset such as:
- [SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- [Enron Spam Dataset](https://www.cs.cmu.edu/~enron/)

Place the dataset as `data/spam.csv` with the following columns:
- `label` (values: `spam` or `ham`)
- `text` (message content)

A small sample is included to help you run the pipeline quickly.

## Setup Instructions
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to Run Locally
### 1) Preprocess the dataset
```bash
python -m src.preprocess
```

### 2) Train models and save artifacts
```bash
python -m src.train
```

### 3) Start the Streamlit app
```bash
streamlit run app.py
```

## Sample Output
```
Prediction: spam
Confidence: 92%
Phishing keywords detected: verify, account
```

## Model Evaluation
During training, the script generates:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

Results are stored in `model/metrics.json`.

## Future Scope
- Add deep learning models (LSTM, BERT)
- Email header analysis for spoofing detection
- API deployment with FastAPI
- Continuous learning from user feedback

## Commands Summary
```bash
python -m src.preprocess
python -m src.train
streamlit run app.py
```
