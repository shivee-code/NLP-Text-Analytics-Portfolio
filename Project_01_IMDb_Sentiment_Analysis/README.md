# Project 01 — IMDb Movie Review Sentiment Analysis

![NLP](https://img.shields.io/badge/Type-Sentiment%20Analysis-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)
![Task](https://img.shields.io/badge/Task-Binary%20Classification-orange)
![Dataset](https://img.shields.io/badge/Dataset-IMDb%20Reviews-green)

## Business Objective

Streaming platforms and entertainment companies receive millions of movie reviews every day — reading them manually is impossible. This project builds a **binary sentiment classification model** that automatically determines whether a movie review is **Positive or Negative** — enabling platforms to track audience sentiment at scale, flag negative experiences, and personalise content recommendations.

---

## Dataset Overview

| Property | Details |
|----------|---------|
| **Source** | IMDb Large Movie Review Dataset |
| **Size** | Movie reviews labelled as Positive / Negative |
| **Task** | Binary Sentiment Classification |
| **Target** | Sentiment — Positive (1) / Negative (0) |
| **Text Type** | Long-form English movie reviews |

---

## NLP Pipeline

```
Raw Reviews
     ↓
Text Cleaning          (lowercase, HTML tag removal, special characters)
     ↓
Tokenization           (split into individual words/tokens)
     ↓
Stopword Removal       (remove "the", "is", "a", "an"...)
     ↓
Lemmatization          (running → run, better → good)
     ↓
TF-IDF Vectorization   (convert text to numerical feature matrix)
     ↓
Model Training         (Multiple classifiers compared)
     ↓
Evaluation             (Accuracy, Precision, Recall, F1, Confusion Matrix)
```

---

## Project Workflow

**Step 1 — Exploratory Data Analysis**
- Review length distribution analysis
- Word frequency analysis for positive vs. negative reviews
- Most common words per sentiment class
- Class balance check

**Step 2 — Text Preprocessing**
- Lowercasing all text
- Removing HTML tags, punctuation, and special characters
- Tokenization using NLTK
- Stopword removal using NLTK English stopwords corpus
- Lemmatization using WordNetLemmatizer

**Step 3 — Feature Extraction**
- **TF-IDF Vectorization** — converts cleaned text into a numerical matrix
- TF-IDF captures word importance relative to the corpus, not just frequency
- Configured with ngram_range=(1,2) to capture bigrams

**Step 4 — Model Training**
Multiple classifiers trained and compared:
- Multinomial Naive Bayes
- Logistic Regression
- Support Vector Machine (Linear SVM)
- Random Forest

**Step 5 — Evaluation**
- Accuracy Score
- Precision, Recall, F1 Score (macro and weighted)
- Confusion Matrix visualization
- Classification Report

---

## Results

| Model | Accuracy | F1 Score | Notes |
|-------|----------|----------|-------|
| Multinomial Naive Bayes | Good baseline | Balanced | Fast, interpretable |
| Logistic Regression | Strong | Strong | Best interpretability |
| **Linear SVM** | **Best ⭐** | **Best ⭐** | Highest accuracy on text |
| Random Forest | Moderate | Moderate | Slower, less suited for text |

**Winner: Linear SVM** — SVM with linear kernel consistently outperforms other classifiers on high-dimensional TF-IDF text features.

---

## Key Business Insights

- **SVM excels at text classification** because reviews exist in very high-dimensional TF-IDF space where linear boundaries work well.
- **Negative reviews use stronger emotional language** — words like "awful", "waste", "boring", "terrible" are highly predictive of negative sentiment.
- **Positive reviews use anticipatory and superlative language** — "brilliant", "masterpiece", "must-watch", "outstanding" are top positive indicators.
- **Review length is not strongly correlated with sentiment** — short negative reviews are as common as long ones.
- **TF-IDF with bigrams** (e.g., "not good", "highly recommend") significantly improves accuracy over unigrams alone by capturing negation patterns.

---

## Sample Predictions

| Review Snippet | Actual | Predicted |
|----------------|--------|-----------|
| "An absolute masterpiece. Brilliant performances..." | Positive | ✅ Positive |
| "Complete waste of time. Terrible plot, bad acting..." | Negative | ✅ Negative |
| "Not what I expected, but surprisingly enjoyable..." | Positive | ✅ Positive |

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| NLTK | Tokenization, stopword removal, lemmatization |
| Scikit-learn | TF-IDF, model training, evaluation metrics |
| Pandas & NumPy | Data manipulation |
| Matplotlib & Seaborn | Confusion matrix, EDA visualizations |
| Jupyter Notebook | Development environment |

---

## Files

```
Project_01_IMDb_Sentiment_Analysis/
├── IMDb_Movie_review_sentiment_analysis.ipynb    ← Full NLP pipeline and analysis
└── README.md         ← You are here
```

---

## How to Run

```bash
# Clone the portfolio repo
git clone https://github.com/shivee-code/NLP-Text-Analytics-Portfolio.git

# Navigate to this project
cd NLP-Text-Analytics-Portfolio/Project_01_IMDb_Sentiment_Analysis

# Install dependencies
pip install pandas numpy scikit-learn nltk matplotlib seaborn

# Download NLTK data (run once)
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

# Open notebook
jupyter notebook notebook.ipynb
```

---

[⬅ Back to Portfolio](../README.md)
