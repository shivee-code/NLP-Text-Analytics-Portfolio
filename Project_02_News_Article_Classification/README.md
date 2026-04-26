# Project 02 — News Article Classification

![NLP](https://img.shields.io/badge/Type-Text%20Classification-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)
![Task](https://img.shields.io/badge/Task-Multi--class%20Classification-orange)
![Dataset](https://img.shields.io/badge/Dataset-News%20Articles-green)

## Business Objective

Digital news platforms publish thousands of articles daily across different categories — Sports, Politics, Technology, Business, Entertainment. Manual tagging is slow and inconsistent. This project builds an **automated multi-class text classification system** that categorizes news articles into their correct topic category — enabling automated content routing, search indexing, and personalised news feeds.

---

## Dataset Overview

| Property | Details |
|----------|---------|
| **Source** | News articles dataset (multi-category) |
| **Categories** | Sports · Politics · Technology · Business · Entertainment |
| **Task** | Multi-class Text Classification |
| **Target** | Article category (5 classes) |
| **Text Type** | Short-to-medium English news articles |

---

## NLP Pipeline

```
Raw News Articles
      ↓
Text Cleaning           (lowercase, punctuation, special chars removal)
      ↓
Tokenization            (split into words)
      ↓
Stopword Removal        (remove common non-informative words)
      ↓
Lemmatization           (normalize word forms)
      ↓
TF-IDF Vectorization    (weighted numerical representation)
      ↓
Multi-class Training    (Naive Bayes, SVM, Logistic Regression)
      ↓
Evaluation              (Accuracy, Precision, Recall, F1 per class)
```

---

## Project Workflow

**Step 1 — Exploratory Data Analysis**
- Article count per category (class balance check)
- Article length distribution per category
- Most frequent words per topic category
- Word clouds for each news category

**Step 2 — Text Preprocessing**
- Lowercasing all article text
- Removing punctuation, numbers, and special characters
- Tokenization using NLTK
- Stopword removal using NLTK English corpus
- Lemmatization using WordNetLemmatizer
- Removing very short tokens (< 2 characters)

**Step 3 — Feature Extraction**
- **TF-IDF Vectorization** with max_features=10,000
- Sublinear TF scaling to handle long articles
- Unigram + Bigram features (ngram_range=(1,2))
- Train-test split (80/20) with stratification

**Step 4 — Model Training**
Three classifiers trained and compared:
- **Multinomial Naive Bayes** — fast, strong baseline for text
- **Linear Support Vector Machine** — best for high-dimensional text
- **Logistic Regression** — interpretable, competitive accuracy

**Step 5 — Evaluation**
- Accuracy Score
- Classification Report (Precision, Recall, F1 per class)
- Confusion Matrix (heatmap) to see per-category errors
- Macro and weighted averages

---

## Results

| Model | Accuracy | Macro F1 | Notes |
|-------|----------|----------|-------|
| Multinomial Naive Bayes | Strong | Good | Best speed-to-accuracy ratio |
| **Linear SVM** | **Best ⭐** | **Best ⭐** | Top performer on all categories |
| Logistic Regression | Strong | Strong | Most interpretable model |

**Winner: Linear SVM** — highest accuracy and F1 across all 5 news categories.

### Per-Category Performance (Linear SVM)

| Category | Precision | Recall | F1 Score |
|----------|-----------|--------|----------|
| Sports | High | High | High |
| Technology | High | High | High |
| Politics | Good | Good | Good |
| Business | Good | Good | Good |
| Entertainment | Good | Good | Good |

---

## Key Business Insights

- **Sports and Technology articles** are easiest to classify — they contain very domain-specific vocabulary (scores, players, gadgets, software) that rarely overlaps with other categories.
- **Politics and Business articles** are most confused with each other — both discuss economics, government, regulation, and markets, creating overlapping vocabulary.
- **TF-IDF bigrams** (e.g., "prime minister", "stock market", "tech startup") significantly improve classification over unigrams alone.
- **SVM's margin maximization** is ideal for news classification — it finds the optimal boundary between category spaces in high-dimensional TF-IDF feature space.
- **Naive Bayes, despite its simplicity**, achieves competitive accuracy on news classification — confirming that word independence assumption is less harmful in short, topically focused texts.

---

## Real-World Applications

| Use Case | How This Model Helps |
|----------|---------------------|
| News aggregator apps | Auto-tag articles into sections without editorial staff |
| Search engines | Improve relevance by routing queries to correct article categories |
| Content recommendation | Surface similar-category articles to increase user engagement |
| Media monitoring | Auto-classify competitor mentions by topic for PR teams |
| RSS feed management | Filter and route articles to relevant subscriber segments |

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| NLTK | Tokenization, stopword removal, lemmatization |
| Scikit-learn | TF-IDF, Naive Bayes, SVM, Logistic Regression, metrics |
| Pandas & NumPy | Data manipulation and analysis |
| Matplotlib & Seaborn | Word frequency charts, confusion matrix heatmap |
| Jupyter Notebook | Development environment |

---

## Files

```
Project_02_News_Article_Classification/
├── news_article_classification.ipynb    ← Full NLP classification pipeline
└── README.md         ← You are here
```

---

## How to Run

```bash
# Clone the portfolio repo
git clone https://github.com/shivee-code/NLP-Text-Analytics-Portfolio.git

# Navigate to this project
cd NLP-Text-Analytics-Portfolio/Project_02_News_Article_Classification

# Install dependencies
pip install pandas numpy scikit-learn nltk matplotlib seaborn

# Download NLTK data (run once)
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"

# Open notebook
jupyter notebook notebook.ipynb
```

---

[⬅ Back to Portfolio](../README.md)
