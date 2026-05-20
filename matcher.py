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
        "aws", "azure", "gcp", "terraform", "docker", "k
