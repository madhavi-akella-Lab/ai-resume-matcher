import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def get_models(st):
    @st.cache_resource
    def _load_embedder():
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _load_embedder()


def compute_match_score(resume_text, jd_text, embedder):
    vecs = embedder.encode([resume_text, jd_text], show_progress_bar=False)
    score = cosine_similarity([vecs[0]], [vecs[1]])[0][0]
    normalized = max(0.0, min(1.0, (score - 0.2) / 0.7))
    return round(float(normalized) * 100, 1), round(float(score) * 100, 1)


def keyword_gap(resume_text, jd_text):
    tech_terms = [
        "python", "sql", "java", "scala", "spark", "pyspark", "kafka",
        "airflow", "dbt", "databricks", "snowflake", "redshift", "bigquery",
        "aws", "azure", "gcp", "terraform", "docker", "kubernetes", "git",
        "mlflow", "sagemaker", "bedrock", "langchain", "openai", "huggingface",
        "rag", "llm", "nlp", "machine learning", "deep learning", "etl", "elt",
        "data warehouse", "data lake", "lakehouse", "power bi", "tableau",
        "informatica", "talend", "alteryx", "streamlit", "fastapi", "flask",
        "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "faiss",
        "pinecone", "chroma", "vector database", "embeddings",
        "medallion", "delta lake", "ci/cd", "devops", "agile", "scrum",
        "restful", "api", "microservices", "hadoop", "hive",
        "looker", "azure data factory", "adf", "adls",
    ]
    resume_lower = resume_text.lower()
    jd_lower = jd_text.lower()
    matched = sorted([t for t in tech_terms if t in resume_lower and t in jd_lower])
    missing = sorted([t for t in tech_terms if t not in resume_lower and t in jd_lower])
    return matched, missing


def analyze_gaps(resume_text, jd_text, embedder):
    matched, missing = keyword_gap(resume_text, jd_text)
    strengths = ", ".join(matched) if matched else "No direct keyword matches found."
    gaps = ", ".join(missing) if missing else "No major keyword gaps detected — great alignment!"
    suggestions = []
    if missing:
        suggestions.append(f"1. Add these missing keywords naturally into your resume: {', '.join(missing[:5])}")
    if len(matched) < 5:
        suggestions.append("2. Your resume may need more technical keywords — expand your skills section.")
    else:
        suggestions.append("2. Good keyword coverage — focus on quantifying your achievements.")
    suggestions.append("3. Tailor your summary section to mirror the language used in the job description.")
    return {
        "gaps": gaps,
        "strengths": strengths,
        "suggestions": "\n".join(suggestions),
    }
