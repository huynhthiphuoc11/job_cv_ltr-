# Learning to Rank - Job-Candidate Matching System
# Experimental Setup Documentation

**Research Project**: NCKH-25-26  
**Objective**: Build and evaluate a Learning to Rank (LTR) system for job-candidate recommendation  
**Date**: February 2026

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Datasets](#datasets)
3. [Experimental Pipeline](#experimental-pipeline)
4. [Feature Engineering](#feature-engineering)
5. [Relevance Labeling](#relevance-labeling)
6. [Model Architecture](#model-architecture)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Reproducibility](#reproducibility)
9. [Usage Instructions](#usage-instructions)

---

## 🎯 Overview

This experimental setup implements a **Learning to Rank (LTR)** approach for matching job candidates to job postings. The system uses:

- **Multi-feature approach**: Text similarity, semantic embeddings, skill overlap, experience matching
- **LambdaMART algorithm**: Gradient boosting decision trees with LambdaMART objective
- **Graded relevance labels**: 5-point scale (0-4) for nuanced ranking
- **Scientific methodology**: Reproducible, well-documented, evaluation-focused

---

## 📊 Datasets

### Job Posting Dataset
- **File**: `jobs_vietnamworks_formatted_fixed.csv`
- **Size**: 6,000 job postings
- **Source**: VietnamWorks, TopCV, ITviec, CareerBuilder (2024-2025)
- **Coverage**: Multi-industry (IT, Marketing, Finance, Manufacturing, Education, Logistics)
- **Key Columns**:
  - `title`: Job title
  - `company`: Company name
  - `skills`: Required skills (comma-separated)
  - `description`: Job description
  - `experience_years_min/max`: Experience requirements
  - `location`: Job location
  - `industry`: Industry sector

### Synthetic Resume Dataset
- **File**: `synthetic_resumes.csv`
- **Size**: 180,000 synthetic resumes
- **Purpose**: Ensure diversity and scale for LTR training
- **Key Columns**:
  - `UserID`: Unique candidate identifier
  - `Skills`: Candidate skills (comma-separated)
  - `Work Experience`: Work history description
  - `Years of Experience`: Total years of experience
  - `Education`: Education level
  - `Desired Job`: Target job title
  - `Location`: Preferred location

---

## 🔬 Experimental Pipeline

The experimental setup follows this systematic pipeline:

### 1. Environment Setup
- **Python Version**: 3.10
- **Key Libraries**: 
  - `pandas`, `numpy` (data manipulation)
  - `scikit-learn` (preprocessing, metrics)
  - `lightgbm` (LambdaMART implementation)
  - `sentence-transformers` (semantic embeddings)
  - `torch` (deep learning backend)
- **Random Seed**: 42 (for reproducibility)

### 2. Data Loading & EDA
- Load both datasets
- Analyze data distributions
- Identify missing values and data quality issues
- Visualize key statistics

### 3. Data Preprocessing
- **Text Cleaning**: Lowercase, remove special characters, normalize whitespace
- **Missing Value Handling**: Fill with empty strings
- **Normalization**: Standardize text formats

### 4. Job-Resume Pair Generation
- **Strategy**: For each job, sample 20 candidates
- **Sampling**: Random sampling to ensure diversity
- **Output**: ~60,000 job-resume pairs (3,000 jobs × 20 candidates)

### 5. Feature Engineering
Four categories of features:

#### A. Skill Overlap Features
- **Jaccard Similarity**: Skill set overlap between job and resume
- **Formula**: `|Job_Skills ∩ Resume_Skills| / |Job_Skills ∪ Resume_Skills|`

#### B. Text Similarity Features
- **TF-IDF Cosine Similarity**: Statistical text similarity
- **Vectorization**: TF-IDF with bi-grams, max 500 features
- **Similarity**: Cosine similarity between job and resume TF-IDF vectors

#### C. Semantic Embedding Features
- **Model**: Sentence-BERT (`all-MiniLM-L6-v2`)
- **Embeddings**: 384-dimensional sentence embeddings
- **Similarity**: Cosine similarity between job and resume embeddings

#### D. Numerical Features
- **Experience Match**: Normalized years of experience (0-1 scale)
- **Range**: 0 to 20 years (normalized)

### 6. Relevance Label Construction

**5-point graded relevance scale**:

| Score | Description | Criteria |
|-------|-------------|----------|
| **4** | Perfect match | Skill overlap > 0.5 AND semantic similarity > 0.7 |
| **3** | Good match | Skill overlap > 0.3 AND semantic similarity > 0.5 |
| **2** | Fair match | Skill overlap > 0.15 AND semantic similarity > 0.3 |
| **1** | Poor match | Skill overlap > 0.05 AND semantic similarity > 0.15 |
| **0** | Irrelevant | Minimal or no match |

**Labeling Logic**:
```python
combined_score = 0.6 * skill_overlap + 0.4 * semantic_similarity
```

### 7. Data Split (Query-based)
- **Train**: 70% of jobs
- **Validation**: 15% of jobs
- **Test**: 15% of jobs
- **Important**: Splits based on job IDs to prevent leakage

### 8. Model Training
- **Algorithm**: LambdaMART (via LightGBM)
- **Objective**: Lambda rank optimization
- **Metric**: NDCG@5, NDCG@10, NDCG@20
- **Early Stopping**: 50 rounds on validation NDCG
- **Hyperparameters**:
  - Learning rate: 0.05
  - Num leaves: 31
  - Feature fraction: 0.9
  - Bagging fraction: 0.8
  - Max iterations: 500

### 9. Evaluation
- **Primary Metric**: NDCG@10 (Normalized Discounted Cumulative Gain at top 10)
- **Secondary Metrics**: NDCG@5, NDCG@20
- **Analysis**: Feature importance, ranking quality visualization

---

## 🔧 Feature Engineering

### Feature Summary

| Feature Name | Type | Description | Range |
|--------------|------|-------------|-------|
| `feat_skill_overlap` | Numerical | Jaccard similarity of skills | [0, 1] |
| `feat_tfidf_similarity` | Numerical | TF-IDF cosine similarity | [0, 1] |
| `feat_embedding_similarity` | Numerical | Semantic embedding similarity | [-1, 1] |
| `feat_resume_years_exp_norm` | Numerical | Normalized years of experience | [0, 1] |

**Total Features**: 4 core features

### Feature Engineering Principles
1. **Complementary Information**: Each feature captures different aspects of match quality
2. **Semantic + Lexical**: Combine both semantic (embeddings) and lexical (TF-IDF, overlap) features
3. **Scalability**: All features are computationally efficient
4. **Interpretability**: Features have clear business meaning

---

## 🏷️ Relevance Labeling

### Labeling Strategy

The relevance labels are **automatically generated** based on multi-feature thresholds:

1. **Skill-centric**: Primary weight on skill overlap (60%)
2. **Semantics-aware**: Secondary weight on semantic similarity (40%)
3. **Graded relevance**: 5-point scale for nuanced ranking

### Label Distribution (Expected)

- **Label 0 (Irrelevant)**: ~40-50%
- **Label 1 (Poor)**: ~20-25%
- **Label 2 (Fair)**: ~15-20%
- **Label 3 (Good)**: ~8-12%
- **Label 4 (Perfect)**: ~3-5%

This distribution reflects real-world job matching scenarios where most candidates are not perfect matches.

---

## 🤖 Model Architecture

### LambdaMART (LightGBM Implementation)

**Why LambdaMART?**
- **Pairwise ranking**: Optimizes pairwise ranking loss
- **NDCG-focused**: Directly optimizes for NDCG metric
- **Gradient boosting**: Powerful ensemble method
- **Industry standard**: Widely used in search and recommendation systems

**Model Configuration**:
```python
{
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [5, 10, 20],
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'seed': 42
}
```

---

## 📈 Evaluation Metrics

### Primary Metric: NDCG@K

**NDCG (Normalized Discounted Cumulative Gain)** at position K:

```
NDCG@K = DCG@K / IDCG@K

where:
DCG@K = Σ(i=1 to K) [ (2^rel_i - 1) / log2(i + 1) ]
IDCG@K = DCG@K with ideal ranking
```

**Why NDCG?**
- Accounts for position in ranking (higher positions more important)
- Handles graded relevance (0-4 scale)
- Normalized for fair comparison across queries
- Industry-standard metric for ranking tasks

### Evaluation Positions
- **NDCG@5**: Top 5 candidates (most critical for UI)
- **NDCG@10**: Top 10 candidates (balanced view)
- **NDCG@20**: Top 20 candidates (comprehensive evaluation)

---

## 🔄 Reproducibility

### Ensuring Reproducibility

1. **Fixed Random Seed**: `RANDOM_SEED = 42` throughout pipeline
2. **Library Versions**:
   - Python: 3.10
   - LightGBM: Latest stable
   - Sentence-Transformers: Latest stable
   - PyTorch: 2.3+
3. **Deterministic Operations**: All sampling with fixed seed
4. **Documentation**: Complete code with comments
5. **Data Versioning**: Fixed dataset snapshots

### Reproducibility Checklist
- ✅ Random seed set for `random`, `numpy`, `torch`
- ✅ Fixed train/val/test splits
- ✅ Deterministic model training
- ✅ Complete code documentation
- ✅ Environment specifications

---

## 🚀 Usage Instructions

### Running the Experimental Notebook

1. **Install Dependencies**:
```bash
pip install pandas numpy scikit-learn lightgbm sentence-transformers torch matplotlib seaborn tqdm
```

2. **Open Jupyter Notebook**:
```bash
jupyter notebook experimental_setup.ipynb
```

3. **Run All Cells**:
   - Sequential execution recommended
   - First run may take 15-30 minutes (embedding generation)
   - Subsequent runs faster with cached embeddings

4. **Expected Outputs**:
   - Data statistics and visualizations
   - Feature distributions
   - Label distribution plots
   - Model training logs
   - Test set NDCG scores
   - Feature importance charts

### Experiment Workflow

```
1. Environment Setup (Cell 1-2)
   ↓
2. Data Loading (Cell 3-5)
   ↓
3. EDA & Exploration (Cell 6-8)
   ↓
4. Preprocessing (Cell 9-10)
   ↓
5. Pair Generation (Cell 11)
   ↓
6. Feature Engineering (Cell 12-17)
   ↓
7. Relevance Labeling (Cell 18-19)
   ↓
8. Train/Val/Test Split (Cell 20)
   ↓
9. LambdaMART Training (Cell 21-22)
   ↓
10. Evaluation & Analysis (Cell 23-26)
   ↓
11. Results Visualization (Cell 27-34)
```

---

## 📝 Key Research Contributions

1. **Comprehensive Feature Set**: Combines lexical, semantic, and numerical features
2. **Graded Relevance**: Multi-level labeling for nuanced ranking
3. **Scalable Pipeline**: Handles large resume datasets efficiently
4. **Scientific Rigor**: Reproducible, well-documented, evaluation-focused
5. **Real-World Dataset**: Multi-industry job postings from Vietnam market

---

## 🔍 Next Steps and Future Work

### Immediate Extensions
1. **Hyperparameter Tuning**: Grid search for optimal LambdaMART parameters
2. **Additional Features**: Add education match, location proximity, industry alignment
3. **Alternative Models**: Compare with RankNet, ListNet, BERT-based models
4. **Cross-Validation**: K-fold CV for robust performance estimates

### Long-term Research Directions
1. **Real-World Validation**: Test on actual applicant-job match data
2. **Fairness Analysis**: Evaluate demographic biases in rankings
3. **Online Learning**: Incorporate user feedback for continuous improvement
4. **Explainability**: Add SHAP/LIME for model interpretability

---

## 📚 References

1. **LambdaMART**: Burges, C. J. (2010). From RankNet to LambdaRank to LambdaMART: An Overview.
2. **NDCG**: Järvelin, K., & Kekäläinen, J. (2002). Cumulated gain-based evaluation of IR techniques.
3. **Sentence-BERT**: Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.
4. **LightGBM**: Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree.

---

## 👥 Team

**Research Project**: NCKH-25-26  
**Institution**: [Your Institution]  
**Contact**: [Your Contact]

---

## 📄 License

This experimental setup is for research purposes.

---

**Last Updated**: February 9, 2026
