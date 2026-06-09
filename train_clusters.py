import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


df = pd.read_csv("data/jobs.csv")


job_embeddings = np.load("job_embeddings.npy")


NUM_CLUSTERS = 11


kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(job_embeddings)


df.to_csv("data/jobs_with_clusters.csv", index=False)


import joblib
joblib.dump(kmeans, "kmeans_model.pkl")

print("Clusters created and saved successfully!")