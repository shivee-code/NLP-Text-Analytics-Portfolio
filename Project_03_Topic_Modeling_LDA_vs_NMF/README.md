# Project 03 — Topic Modeling: LDA vs NMF Comparison

![NLP](https://img.shields.io/badge/Type-Topic%20Modeling-purple)
![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)
![LDA](https://img.shields.io/badge/Algorithm%201-LDA%20%7C%20Gensim-orange)
![NMF](https://img.shields.io/badge/Algorithm%202-NMF%20%7C%20Scikit--learn-blue)
![Visualization](https://img.shields.io/badge/Visualization-pyLDAvis%20Interactive-green)

## Business Objective

Organizations managing large volumes of unstructured text — news archives, customer feedback, research papers, support tickets — need to understand **what topics are being discussed** at scale, without manually reading every document. This project implements and compares **two industry-standard unsupervised topic modeling algorithms** — LDA (Latent Dirichlet Allocation) and NMF (Non-negative Matrix Factorization) — on the same corpus to identify which produces better, more actionable results and when each should be used in real-world pipelines.

---

## Why Compare LDA vs NMF?

In industry, choosing the wrong topic modeling algorithm leads to poor topic quality, wasted engineering effort, and uninterpretable results. This project answers a critical question data scientists face:

> *"I have a text corpus. Should I use LDA or NMF — and why?"*

By implementing both on the same data and comparing results objectively using **Coherence Score**, this project demonstrates analytical depth beyond just running a single algorithm.

---

## Algorithms Overview

### Part A — LDA (Latent Dirichlet Allocation)

LDA is a **probabilistic generative model** that assumes:
- Every **document** is a probabilistic mixture of topics
- Every **topic** is a probabilistic mixture of words

```
Document = 30% Topic_Economy + 50% Topic_Politics + 20% Topic_Sports
Topic_Economy = [0.15 "market"] + [0.12 "trade"] + [0.10 "growth"] + ...
```

**Input:** Bag-of-Words counts matrix
**Library:** Gensim `LdaModel`
**Visualization:** pyLDAvis interactive HTML

---

### Part B — NMF (Non-negative Matrix Factorization)

NMF is a **linear algebra matrix decomposition** technique that factorizes the document-term matrix into two non-negative matrices:

```
Document-Term Matrix (V)  ≈  W  ×  H
                               ↑      ↑
                       Document-    Topic-Word
                       Topic        Matrix
                       Matrix
```

Because all values must be **non-negative**, NMF produces additive, parts-based representations — sharper and more deterministic than LDA.

**Input:** TF-IDF matrix
**Library:** Scikit-learn `NMF`
**Initialization:** `nndsvda` (deterministic, better convergence)

---

## Project Workflow

```
Raw Text Corpus
      ↓
Text Preprocessing          (shared by both algorithms)
      ↓
         ┌──────────────────────────────────┐
         ↓                                  ↓
  Bag-of-Words                         TF-IDF Matrix
  (for LDA)                            (for NMF)
         ↓                                  ↓
  Gensim LDA Model               Scikit-learn NMF Model
         ↓                                  ↓
  Topic Extraction               Topic Extraction
  + Coherence Score              + Topic-Word Weights
         ↓                                  ↓
  pyLDAvis Visualization         Bar Chart Visualization
         ↓                                  ↓
         └──────────────────────────────────┘
                       ↓
              LDA vs NMF Comparison
              (Topic quality, overlap, interpretability)
                       ↓
              Final Recommendation
              (When to use which in industry)
```

---

## Text Preprocessing Pipeline (Shared)

**Step 1 — Text Cleaning**
- Lowercase conversion
- Punctuation and special character removal
- Number removal

**Step 2 — Tokenization**
- NLTK word tokenizer

**Step 3 — Stopword Removal**
- NLTK English stopwords corpus
- Custom domain-specific stopwords added

**Step 4 — Lemmatization**
- NLTK WordNetLemmatizer (verbs + nouns)
- Tokens shorter than 3 characters removed

---

## Part A — LDA Implementation

**Dictionary & Corpus Creation**
- Gensim `Dictionary` built from preprocessed tokens
- `filter_extremes`: removed words in < 2 docs or > 50% of docs
- Corpus converted to Bag-of-Words: `[(word_id, count), ...]`

**Optimal Topics — Coherence Score Analysis**
- Ran LDA for `n_topics` = 2 to 15
- Plotted Coherence Score (c_v metric) vs. number of topics
- Selected `n_topics` at the "elbow" — best coherence before diminishing returns

**LDA Model Hyperparameters**
```python
LdaModel(
    corpus=corpus,
    num_topics=optimal_k,
    id2word=dictionary,
    alpha='auto',        # auto-tune document-topic density
    eta='auto',          # auto-tune topic-word density
    passes=20,           # training passes over corpus
    random_state=42      # reproducibility
)
```

**Output**
- Top 10 keywords per topic
- Per-document dominant topic assignment
- `lda_visualization.html` — interactive pyLDAvis explorer

---

## Part B — NMF Implementation

**TF-IDF Vectorization**
```python
TfidfVectorizer(
    max_features=10000,
    min_df=2,
    max_df=0.95,
    ngram_range=(1, 2)   # unigrams + bigrams
)
```

**NMF Model**
```python
NMF(
    n_components=optimal_k,
    init='nndsvda',      # deterministic initialization
    random_state=42,
    max_iter=400
)
```

**Output**
- Top 10 keywords per topic from H matrix rows
- Document-topic weight matrix (W) for dominant topic assignment
- Bar chart visualization of top words per topic

---

## Results

### Topics Discovered (Illustrative — Same Corpus)

| Topic | LDA Top Words | NMF Top Words | Interpretation |
|-------|--------------|--------------|----------------|
| Topic 1 | economy, market, stock, trade, price | stock market, trade policy, economic growth | Finance |
| Topic 2 | team, game, player, match, goal | football team, match result, player score | Sports |
| Topic 3 | technology, software, app, digital, AI | artificial intelligence, software development | Technology |
| Topic 4 | election, vote, party, government, minister | government policy, election result, party leader | Politics |
| Topic 5 | film, music, celebrity, award, artist | film award, music artist, box office | Entertainment |

> *NMF bigrams (ngram_range=(1,2)) produce more specific, phrase-level topic keywords — making topics slightly more interpretable than LDA unigrams.*

---

## LDA vs NMF — Full Comparison

| Aspect | LDA | NMF |
|--------|-----|-----|
| **Mathematical Basis** | Probabilistic generative model | Matrix factorization (linear algebra) |
| **Input Format** | Bag-of-Words counts | TF-IDF weighted matrix |
| **Topic Overlap** | Soft — documents are probability mixtures | Hard — cleaner, more distinct boundaries |
| **Determinism** | Stochastic — varies per run | Deterministic — same result every time |
| **Short Texts** | Less effective | More effective ✓ |
| **Long Documents** | Better — captures probabilistic mixing ✓ | Works but LDA preferred |
| **Topic Sharpness** | Good | Sharper ✓ |
| **Visualization** | pyLDAvis — excellent interactive ✓ | Bar charts — simpler |
| **Speed** | Slower on large corpora | Faster ✓ |
| **Reproducibility** | Needs `random_state` (still varies) | Fully reproducible ✓ |
| **Best For** | News, books, research papers | Reviews, tweets, customer feedback |
| **Industry Use** | Research, NLP exploration | Production pipelines ✓ |

### Final Recommendation

| Situation | Use |
|-----------|-----|
| Long documents (articles, reports, books) | **LDA** |
| Short texts (reviews, tweets, support tickets) | **NMF** |
| Need reproducible, deterministic output | **NMF** |
| Need interactive visualization to present to stakeholders | **LDA** (pyLDAvis) |
| Topic overlap is natural in your corpus | **LDA** |
| Topics should be clearly distinct | **NMF** |
| Speed matters in a production pipeline | **NMF** |

---

## Key Business Insights

- **Both algorithms discovered the same 5 high-level themes** on this corpus — confirming that real topics exist in the data, not artifacts of the algorithm.
- **NMF produced slightly sharper topic keywords** because TF-IDF input downweights common words before factorization — giving NMF a cleaner signal.
- **LDA's pyLDAvis visualization** is significantly more powerful for stakeholder presentations — the inter-topic distance map makes topic relationships immediately intuitive.
- **Coherence Score** objectively confirmed that 5 topics was the optimal number — removing the guesswork from `n_topics` selection.
- **In a production setting**, NMF would be preferred for this corpus — deterministic output, faster inference, and comparable topic quality make it more pipeline-friendly.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| NLTK | Tokenization · Stopwords · Lemmatization |
| Gensim | LDA model · Dictionary · Corpus · CoherenceModel |
| Scikit-learn | TF-IDF Vectorizer · NMF decomposition |
| pyLDAvis | Interactive LDA topic visualization (HTML) |
| Pandas & NumPy | Data manipulation and matrix operations |
| Matplotlib & Seaborn | Coherence plot · NMF bar charts · Topic heatmaps |
| Jupyter Notebook | Development environment |

---

## Files

```
Project_03_Topic_Modeling_LDA_vs_NMF/
├── topic_modeling_with_LDA.ipynb     ← LDA full implementation
├── topic_modeling_with_LDA.py        ← LDA Python script
├── lda_visualization.html            ← pyLDAvis interactive output (open in browser)
├── topic_modeling_with_NMF.ipynb     ← NMF full implementation
├── topic_modeling_with_NMF.py        ← NMF Python script
├── LDA_image_topic1
├── LDA_image_topic1
├── LDA_image_topic1 
└── README.md                         ← You are here
```

---

## How to Run

```bash
# Clone the portfolio repo
git clone https://github.com/shivee-code/NLP-Text-Analytics-Portfolio.git

# Navigate to this project
cd NLP-Text-Analytics-Portfolio/Project_03_Topic_Modeling_LDA_vs_NMF

# Install dependencies
pip install pandas numpy gensim nltk pyldavis scikit-learn matplotlib seaborn

# Download NLTK data (run once)
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"

# Run LDA notebook
jupyter notebook Topic_Modeling_with_LDA.ipynb

# Run NMF notebook
jupyter notebook Topic_Modeling_with_NMF.ipynb

# View interactive LDA visualization
# Open lda_visualization.html in any browser
```

---

[⬅ Back to Portfolio](../README.md)
