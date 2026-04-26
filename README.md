# NLP Text Analytics Portfolio

![NLP](https://img.shields.io/badge/Natural%20Language%20Processing-NLTK%20%7C%20Gensim%20%7C%20Scikit--learn-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)
![Status](https://img.shields.io/badge/Status-Active-green)
![Projects](https://img.shields.io/badge/Projects-3-purple)

A curated collection of **end-to-end Natural Language Processing projects** covering Sentiment Analysis, Multi-class Text Classification, and Unsupervised Topic Modeling — applied to real-world text datasets including movie reviews, news articles, and document corpora.

---

## About This Portfolio

This portfolio demonstrates practical NLP skills across the full text analytics pipeline — from raw text preprocessing and feature extraction to model building, evaluation, and topic interpretation. Each project applies industry-standard NLP techniques using Python.

**Target Roles:** NLP Engineer · Data Scientist · AI Analyst · Machine Learning Engineer · Text Analytics Analyst

---

## Projects Overview

| # | Project | NLP Task | Technique | Dataset | Key Result |
|---|---------|----------|-----------|---------|------------|
| 01 | [IMDb Sentiment Analysis](#project-01--imdb-movie-review-sentiment-analysis) | Sentiment Analysis | TF-IDF + ML Models | IMDb Movie Reviews | Binary classification — Positive / Negative |
| 02 | [News Article Classification](#project-02--news-article-classification) | Multi-class Classification | TF-IDF + Naive Bayes / SVM | News Articles | Multi-category news classification |
| 03 | [Topic Modeling — LDA vs NMF](#project-03--topic-modeling-lda-vs-nmf-comparison) | Topic Modeling | LDA (Gensim) + NMF (Scikit-learn) | Document Corpus | Full algorithm comparison — when to use which |

---

## Project Details

### Project 01 — IMDb Movie Review Sentiment Analysis

**Business Problem:** Automatically classify movie reviews as Positive or Negative to help streaming platforms understand audience sentiment at scale — without reading millions of reviews manually.

**Approach:** Built a binary sentiment classification pipeline using TF-IDF vectorization and multiple ML classifiers with full NLP preprocessing.

**NLP Pipeline:**
- Text cleaning → Tokenization → Stopword removal → Lemmatization → TF-IDF → Classification

**Key Skills:** Sentiment Analysis · TF-IDF · Binary Classification · NLTK · Text Preprocessing · Confusion Matrix

📁 [View Project](./Project_01_IMDb_Sentiment_Analysis/)

---

### Project 02 — News Article Classification

**Business Problem:** Automatically categorize news articles into topics (Sports, Politics, Technology, Business, Entertainment) to power news recommendation engines and content management systems.

**Approach:** Multi-class text classification using TF-IDF feature extraction combined with Naive Bayes and Support Vector Machine classifiers.

**NLP Pipeline:**
- Text cleaning → Tokenization → Stopword removal → TF-IDF vectorization → Multi-class classification

**Key Skills:** Multi-class Classification · TF-IDF · Naive Bayes · SVM · Precision/Recall/F1 · NLTK

📁 [View Project](./Project_02_News_Article_Classification/)

---

### Project 03 — Topic Modeling: LDA vs NMF Comparison

**Business Problem:** Discover hidden thematic structure in large document collections — and evaluate which unsupervised topic modeling algorithm (LDA or NMF) produces better, more actionable results for real-world text data.

**Approach:** Implemented both Latent Dirichlet Allocation (Gensim) and Non-negative Matrix Factorization (Scikit-learn) on the same corpus — with a full side-by-side comparison of topic quality, interpretability, and use-case fit.

**What Makes This Project Strong:**
- LDA implemented with Gensim + pyLDAvis interactive visualization
- NMF implemented with Scikit-learn TF-IDF pipeline
- Coherence Score used to objectively evaluate topic quality
- Full LDA vs NMF comparison table — when to use which in industry

**Key Skills:** LDA · NMF · Gensim · pyLDAvis · TF-IDF · Coherence Score · Unsupervised NLP · Algorithm Comparison · Scikit-learn

📁 [View Project](./Project_03_Topic_Modeling_LDA_vs_NMF/)

---

## NLP Pipeline — How Each Project Works

```
Raw Text
    ↓
Text Cleaning         (lowercase, punctuation removal, special chars)
    ↓
Tokenization          (split text into words/tokens)
    ↓
Stopword Removal      (remove "the", "is", "and"...)
    ↓
Lemmatization         (running → run, better → good)
    ↓
Feature Extraction    (TF-IDF / Bag-of-Words / Document-Term Matrix)
    ↓
Model Training        (Classification / Topic Modeling)
    ↓
Evaluation & Insights (Accuracy / F1 / Coherence Score / Visualization)
```

---

## Tech Stack

| Category | Tools & Libraries |
|----------|------------------|
| Language | Python 3.8+ |
| NLP Libraries | NLTK · spaCy · Gensim |
| Feature Extraction | TF-IDF · Bag-of-Words · CountVectorizer |
| Machine Learning | Scikit-learn |
| Topic Modeling | Gensim (LDA) · Scikit-learn NMF · pyLDAvis |
| Data Manipulation | Pandas · NumPy |
| Visualization | Matplotlib · Seaborn · pyLDAvis |
| Environment | Jupyter Notebook · Google Colab |
| Version Control | Git · GitHub |

---

## Repository Structure

```
NLP-Text-Analytics-Portfolio/
│
├── Project_01_IMDb_Sentiment_Analysis/
│   ├── IMDb_Movie_review_sentiment_analysis.ipynb
│   └── README.md
│
├── Project_02_News_Article_Classification/
│   ├── news_article_classification.ipynb
│   └── README.md
│
├── Project_03_Topic_Modeling_LDA_vs_NMF/
│   ├── LDA_image_topic1
│   ├── LDA_image_topic2
│   ├── LDA_image_topic3
│   ├── topic_modeling_with_LDA.ipynb
│   ├── topic_modeling_with_LDA.py
│   ├── lda_visualization.html
│   ├── topic_modeling_with_NMF.ipynb
│   ├── topic_modeling_with_NMF.py
│   └── README.md
│
└── README.md   ← You are here
```

---

## Key Skills Demonstrated

- **Sentiment Analysis:** Binary opinion classification from unstructured text
- **Text Classification:** Multi-class document categorization using TF-IDF + ML
- **Topic Modeling:** Unsupervised topic discovery using LDA and NMF
- **NLP Preprocessing:** Tokenization · Stopword removal · Lemmatization · Text normalization
- **Feature Engineering:** TF-IDF vectorization · Bag-of-Words · Document-term matrices
- **Model Evaluation:** Accuracy · Precision · Recall · F1 Score · Coherence Score
- **Visualization:** Confusion Matrix · Word Clouds · pyLDAvis interactive charts

---

## Connect With Me

**Name:** Shivam Kumar <br>
**LinkedIn:** [Shivam-Kumar](https://www.linkedin.com/in/shivam-kumar-2a0371246/) <br>
**GitHub:** [shivee-code](https://github.com/shivee-code)

---

> *"Text data is everywhere. The ability to extract meaning from it is a superpower."*
