from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


df = pd.read_csv("data/jobs_with_clusters.csv")
job_embeddings = np.load("job_embeddings.npy")


kmeans = joblib.load("kmeans_model.pkl")


from functools import lru_cache

@lru_cache()
def get_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = get_model()



class SkillRequest(BaseModel):
    skills: str



@app.get("/")
def home():
    return {"message": "AI Career API is running 🚀"}


@app.post("/recommend")
def recommend(data: SkillRequest):

  
    user_embedding = model.encode([data.skills])

  
    cluster = kmeans.predict(user_embedding)[0]

   
    cluster_indices = df[df["cluster"] == cluster].index.tolist()

    cluster_embeddings = job_embeddings[cluster_indices]

   
    similarities = cosine_similarity(user_embedding, cluster_embeddings)[0]

  
    top_k = np.argsort(similarities)[::-1][:10]

    results = []

    for i in top_k:
        idx = cluster_indices[i]

        score = float(similarities[i])

       
        if score < 0.30:
            continue


        results.append({
            "title": df.iloc[idx]["title"],
            "domain": df.iloc[idx]["job_domain"],
            "score": round(score, 2),
            "cluster": int(cluster)
        })

    return {"recommendations": results}