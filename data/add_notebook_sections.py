"""
Script to add comprehensive LTR experimental sections to Jupyter notebook
"""
import json

# Load existing notebook
with open('experimental_setup.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Additional cells for preprocessing, feature engineering, and LTR training
new_cells = [
    # Section 3: Data Preprocessing
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Data Preprocessing\n",
            "\n",
            "### 3.1 Text Cleaning and Normalization"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def clean_text(text):\n",
            "    \"\"\"Clean and normalize text data\"\"\"\n",
            "    if pd.isna(text):\n",
            "        return ''\n",
            "    text = str(text).lower()\n",
            "    text = re.sub(r'[^a-z0-9\\s,.]', ' ', text)\n",
            "    text = re.sub(r'\\s+', ' ', text)\n",
            "    return text.strip()\n",
            "\n",
            "print(\"Cleaning job descriptions...\")\n",
            "if 'description' in df_jobs.columns:\n",
            "    df_jobs['description_clean'] = df_jobs['description'].apply(clean_text)\n",
            "\n",
            "if 'skills' in df_jobs.columns:\n",
            "    df_jobs['skills_clean'] = df_jobs['skills'].apply(clean_text)\n",
            "\n",
            "print(\"Cleaning resume data...\")\n",
            "if 'Skills' in df_resumes.columns:\n",
            "    df_resumes['skills_clean'] = df_resumes['Skills'].apply(clean_text)\n",
            "\n",
            "if 'Work Experience' in df_resumes.columns:\n",
            "    df_resumes['experience_clean'] = df_resumes['Work Experience'].apply(clean_text)\n",
            "\n",
            "print(\"Text cleaning complete!\")"
        ]
    },
    # Section 4: Pair Generation
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Job-Resume Pair Generation\n",
            "\n",
            "### Strategy:\n",
            "- For each job, sample both relevant and non-relevant candidates\n",
            "- Create balanced dataset for LTR training\n",
            "- Total pairs: ~50,000-100,000"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Pair generation parameters\n",
            "CANDIDATES_PER_JOB = 20  # Sample 20 candidates per job\n",
            "NUM_JOBS_SAMPLE = 3000  # Use subset for faster experimentation\n",
            "\n",
            "print(f\"Generating job-resume pairs...\")\n",
            "print(f\"Jobs: {NUM_JOBS_SAMPLE}, Candidates per job: {CANDIDATES_PER_JOB}\")\n",
            "print(f\"Expected pairs: {NUM_JOBS_SAMPLE * CANDIDATES_PER_JOB:,}\")\n",
            "\n",
            "# Sample jobs\n",
            "sampled_jobs = df_jobs.sample(n=min(NUM_JOBS_SAMPLE, len(df_jobs)), random_state=RANDOM_SEED)\n",
            "\n",
            "# Generate pairs\n",
            "pairs = []\n",
            "for idx, job in tqdm(sampled_jobs.iterrows(), total=len(sampled_jobs), desc=\"Generating pairs\"):\n",
            "    # Sample candidates randomly\n",
            "    sampled_resumes = df_resumes.sample(n=CANDIDATES_PER_JOB, random_state=RANDOM_SEED+idx)\n",
            "    \n",
            "    for _, resume in sampled_resumes.iterrows():\n",
            "        pairs.append({\n",
            "            'job_id': idx,\n",
            "            'resume_id': resume.get('UserID', resume.name),\n",
            "            'job_title': job.get('title', ''),\n",
            "            'job_skills': job.get('skills_clean', ''),\n",
            "            'job_description': job.get('description_clean', ''),\n",
            "            'resume_skills': resume.get('skills_clean', ''),\n",
            "            'resume_experience': resume.get('experience_clean', ''),\n",
            "            'resume_years_exp': resume.get('Years of Experience', 0)\n",
            "        })\n",
            "\n",
            "df_pairs = pd.DataFrame(pairs)\n",
            "print(f\"\\nGenerated {len(df_pairs):,} job-resume pairs\")\n",
            "df_pairs.head()"
        ]
    },
    # Section 5: Feature Engineering
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Feature Engineering\n",
            "\n",
            "### Feature Categories:\n",
            "1. **Text Similarity Features**: TF-IDF cosine similarity, skill overlap\n",
            "2. **Embedding Features**: Semantic embeddings using Sentence-BERT\n",
            "3. **Numerical Features**: Experience matching, education level\n",
            "4. **Categorical Features**: Location match, industry match"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"=\" * 80)\n",
            "print(\"FEATURE ENGINEERING\")\n",
            "print(\"=\" * 80)\n",
            "\n",
            "# Feature 1: Skill overlap (Jaccard similarity)\n",
            "def skill_overlap(job_skills, resume_skills):\n",
            "    \"\"\"Calculate Jaccard similarity between job and resume skills\"\"\"\n",
            "    if not job_skills or not resume_skills:\n",
            "        return 0.0\n",
            "    job_set = set(str(job_skills).split())\n",
            "    resume_set = set(str(resume_skills).split())\n",
            "    if not job_set or not resume_set:\n",
            "        return 0.0\n",
            "    intersection = len(job_set & resume_set)\n",
            "    union = len(job_set | resume_set)\n",
            "    return intersection / union if union > 0 else 0.0\n",
            "\n",
            "print(\"\\n1. Computing skill overlap...\")\n",
            "df_pairs['feat_skill_overlap'] = df_pairs.apply(\n",
            "    lambda x: skill_overlap(x['job_skills'], x['resume_skills']), axis=1\n",
            ")\n",
            "\n",
            "print(f\"   Skill overlap range: [{df_pairs['feat_skill_overlap'].min():.3f}, {df_pairs['feat_skill_overlap'].max():.3f}]\")\n",
            "print(f\"   Mean: {df_pairs['feat_skill_overlap'].mean():.3f}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Feature 2: TF-IDF Cosine Similarity\n",
            "print(\"\\n2. Computing TF-IDF similarity...\")\n",
            "\n",
            "# Combine job description and skills\n",
            "df_pairs['job_text'] = df_pairs['job_description'] + ' ' + df_pairs['job_skills']\n",
            "df_pairs['resume_text'] = df_pairs['resume_experience'] + ' ' + df_pairs['resume_skills']\n",
            "\n",
            "# TF-IDF vectorization\n",
            "tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2), min_df=2)\n",
            "all_texts = pd.concat([df_pairs['job_text'], df_pairs['resume_text']])\n",
            "tfidf.fit(all_texts)\n",
            "\n",
            "job_tfidf = tfidf.transform(df_pairs['job_text'])\n",
            "resume_tfidf = tfidf.transform(df_pairs['resume_text'])\n",
            "\n",
            "# Compute cosine similarity\n",
            "tfidf_similarity = []\n",
            "for i in tqdm(range(len(df_pairs)), desc=\"Computing TF-IDF similarity\"):\n",
            "    sim = cosine_similarity(job_tfidf[i], resume_tfidf[i])[0][0]\n",
            "    tfidf_similarity.append(sim)\n",
            "\n",
            "df_pairs['feat_tfidf_similarity'] = tfidf_similarity\n",
            "print(f\"   TF-IDF similarity range: [{min(tfidf_similarity):.3f}, {max(tfidf_similarity):.3f}]\")\n",
            "print(f\"   Mean: {np.mean(tfidf_similarity):.3f}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Feature 3: Semantic Embeddings (Sentence-BERT)\n",
            "print(\"\\n3. Computing semantic embeddings...\")\n",
            "\n",
            "# Load pre-trained model\n",
            "model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast and effective\n",
            "\n",
            "# Generate embeddings\n",
            "print(\"   Encoding job texts...\")\n",
            "job_embeddings = model.encode(df_pairs['job_text'].tolist(), show_progress_bar=True, batch_size=32)\n",
            "\n",
            "print(\"   Encoding resume texts...\")\n",
            "resume_embeddings = model.encode(df_pairs['resume_text'].tolist(), show_progress_bar=True, batch_size=32)\n",
            "\n",
            "# Compute cosine similarity\n",
            "embedding_similarity = []\n",
            "for i in range(len(job_embeddings)):\n",
            "    sim = cosine_similarity([job_embeddings[i]], [resume_embeddings[i]])[0][0]\n",
            "    embedding_similarity.append(sim)\n",
            "\n",
            "df_pairs['feat_embedding_similarity'] = embedding_similarity\n",
            "print(f\"   Embedding similarity range: [{min(embedding_similarity):.3f}, {max(embedding_similarity):.3f}]\")\n",
            "print(f\"   Mean: {np.mean(embedding_similarity):.3f}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Feature 4: Experience Match (numerical)\n",
            "print(\"\\n4. Computing experience match...\")\n",
            "\n",
            "# Normalize years of experience\n",
            "df_pairs['feat_resume_years_exp_norm'] = df_pairs['resume_years_exp'] / 20.0  # Assuming max 20 years\n",
            "\n",
            "print(f\"   Years of experience normalized: [{df_pairs['feat_resume_years_exp_norm'].min():.3f}, {df_pairs['feat_resume_years_exp_norm'].max():.3f}]\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Summary of all features\n",
            "feature_cols = [col for col in df_pairs.columns if col.startswith('feat_')]\n",
            "print(f\"\\n{'='*80}\")\n",
            "print(f\"FEATURE SUMMARY\")\n",
            "print(f\"{'='*80}\")\n",
            "print(f\"Total features engineered: {len(feature_cols)}\")\n",
            "print(f\"\\nFeatures: {feature_cols}\")\n",
            "print(f\"\\nFeature statistics:\")\n",
            "print(df_pairs[feature_cols].describe().T)"
        ]
    }
]

# Add new cells to notebook
notebook['cells'].extend(new_cells)

# Save updated notebook
with open('experimental_setup.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Added {len(new_cells)} new cells to notebook")
print(f"Total cells now: {len(notebook['cells'])}")
