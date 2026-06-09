import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

df = pd.read_csv("data/jobs.csv")

df["combined_text"] = (
    df["title"].fillna("") *3 + " " +
    df["job_domain"].fillna("") + " " +
    df["description"].fillna("")
)

model = SentenceTransformer("all-MiniLM-L6-v2")

job_embeddings = model.encode(
    df["combined_text"].tolist(),
    show_progress_bar=True
)

np.save("job_embeddings.npy", job_embeddings)

print("Embeddings saved successfully!")