# BBC Text Representations Assignment

Text representation learning and evaluation on BBC News dataset - comparing sparse and dense methods for classification and retrieval tasks.

## 📋 Assignment Overview

This project implements and compares multiple text representation methods:
- **Sparse**: One-Hot Encoding, Bag-of-Words, N-grams, TF-IDF
- **Dense**: Word2Vec (Skip-gram & CBOW with NS/HS), GloVe embeddings

**Tasks:**
1. Health & efficiency metrics (vocabulary, sparsity, OOV, speed, memory)
2. Classification (Logistic Regression/SVM on 5 BBC categories)
3. Retrieval (query generation and ranking with MAP@5, Recall@10)

**Due Date:** December 18, 2025, 12:00 PM (Hard Deadline)

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
pip or conda
```

### Installation
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/bbc-text-representations.git
cd bbc-text-representations

# Install dependencies
pip install -r requirements.txt

# Download required NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Download Data
1. Place `bbc-text.csv` in the `data/` folder
2. Download GloVe embeddings:
   ```bash
   cd data/
   wget http://nlp.stanford.edu/data/glove.6B.zip
   unzip glove.6B.zip
   # Keep only glove.6B.100d.txt, delete others to save space
   ```

---

## 📁 Project Structure

```
bbc-text-representations/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── data/                              # Datasets (gitignored)
│   ├── bbc-text.csv                  # Original BBC dataset
│   ├── master.csv                     # Generated with folds
│   └── glove.6B.100d.txt             # GloVe embeddings
│
├── notebooks/                         # Development notebooks
│   ├── 01_setup_and_preprocessing.ipynb
│   ├── 02_sparse_methods.ipynb
│   ├── 03_dense_methods.ipynb
│   ├── 04_classification.ipynb
│   ├── 05_retrieval_and_analysis.ipynb
│   └── final_notebook.ipynb          # MERGED - FOR SUBMISSION
│
├── cache/                             # Intermediate results (gitignored)
│   ├── train_processed.pkl
│   ├── dev_processed.pkl
│   ├── test_processed.pkl
│   └── vocab.pkl
│
├── models/                            # Trained models (gitignored)
│   ├── w2v_sg_ns.model
│   ├── w2v_cbow_ns.model
│   └── ...
│
└── outputs/                           # Final submission files
    ├── results.json                   # All metrics
    ├── preds_test.csv                 # Test predictions
    ├── queries.json                   # Generated queries
    ├── rankings.json                  # Query rankings
    └── report.pdf                     # Analysis report
```

---

## 🔧 Development Workflow

### Step 1: Data Preparation
Run `01_setup_and_preprocessing.ipynb`:
- Creates `master.csv` with stratified 5-fold splits
- Generates deterministic train/dev/test split from roll number
- Preprocesses text (lowercase, tokenize, stopwords, lemmatize)
- Saves processed data to `cache/`

### Step 2: Sparse Representations
Run `02_sparse_methods.ipynb`:
- One-Hot Encoding (top 2000 tokens)
- Bag-of-Words (unigram counts)
- N-grams (unigrams + bigrams)
- TF-IDF with manual verification
- Calculates health metrics

### Step 3: Dense Representations
Run `03_dense_methods.ipynb`:
- Word2Vec Skip-gram (Negative Sampling & Hierarchical Softmax)
- Word2Vec CBOW (Negative Sampling & Hierarchical Softmax)
- GloVe (pretrained 100d)
- TF-IDF weighted pooling for document vectors

### Step 4: Classification
Run `04_classification.ipynb`:
- Train Logistic Regression / Linear SVM
- Tune hyperparameter C on DEV set
- Evaluate on TEST set (Macro-F1, Accuracy)
- Generate `preds_test.csv`

### Step 5: Retrieval & Analysis
Run `05_retrieval_and_analysis.ipynb`:
- Generate 20 deterministic queries (15 TF-IDF + 5 negation)
- Rank TEST documents by cosine similarity
- Calculate MAP@5, Recall@10, Negation Top-1%
- Compare Word2Vec NS vs HS vs GloVe
- Generate `queries.json`, `rankings.json`, `results.json`

### Step 6: Final Submission
1. Merge all notebooks into `final_notebook.ipynb`
2. Run top-to-bottom to ensure reproducibility
3. Run validation script to verify outputs
4. Write `report.pdf` (1-2 pages)
5. Package submission files

---

## 📊 Metrics Tracked

### Health Metrics
- Vocabulary size (V)
- Non-zero entries (nnz)
- Sparsity
- OOV rate on TEST
- Top-k coverage (k=100, 500)
- Training time (seconds)
- Transform time (ms/doc)
- Memory usage (MB)
- Tokens/sec (Word2Vec only)

### Classification
- Macro-F1 (primary)
- Accuracy

### Retrieval
- MAP@5
- Recall@10
- Negation Top-1% accuracy

---

## ✅ Pre-Submission Checklist

- [ ] All notebooks run without errors
- [ ] `final_notebook.ipynb` runs top-to-bottom
- [ ] Validation script passes: `python validator.py`
- [ ] All required files in `outputs/`:
  - [ ] `results.json`
  - [ ] `preds_test.csv`
  - [ ] `queries.json`
  - [ ] `rankings.json`
  - [ ] `report.pdf`
- [ ] Report compares Word2Vec NS vs HS vs GloVe
- [ ] QUERY_SIGNATURE and RANK_SIGNATURE printed

---

## 📝 Submission Format

**Folder name:** `<YOUR_ROLL>/` (e.g., `SE22UARI001/`)

**Contents:**
```
SE22UARI001/
├── notebook.ipynb        # Renamed from final_notebook.ipynb
├── results.json
├── preds_test.csv
├── queries.json
├── rankings.json
└── report.pdf
```

**Zip and submit:** `SE22UARI001.zip`

---

## 🛠️ Technologies Used

- **Text Processing:** NLTK, spaCy
- **Representations:** scikit-learn, gensim
- **Machine Learning:** scikit-learn
- **Data Handling:** pandas, numpy
- **Embeddings:** GloVe (Stanford NLP)

---

## 📚 Resources

- [Assignment Document](link-to-assignment-doc)
- [BBC Dataset](link-to-bbc-dataset)
- [GloVe Embeddings](https://nlp.stanford.edu/projects/glove/)
- [Gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html)

---

## 🤝 Contributing

This is an academic assignment. Individual work only.

---

## 📧 Contact

**Name:** Your Name  
**Roll Number:** SE22UXXXXX  
**Email:** your.email@example.com

---

## 📄 License

This project is for educational purposes only.