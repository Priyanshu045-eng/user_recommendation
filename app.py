from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = FastAPI(title="User Recommendation API")

# -----------------------------
# 🔹 Data Model for incoming users
# -----------------------------
class User(BaseModel):
    user_id: int
    name: str
    branch: str      # new field
    year: int        # new field
    interests: str

# -----------------------------
# 🔹 API Endpoint: Recommend Users
# -----------------------------
@app.post("/recommend_users/")
def recommend_users_api(users: List[User], user_id: int, top_n: int = 10):
    # Convert input data (from backend / frontend) into DataFrame
    users_df = pd.DataFrame([u.dict() for u in users])

    # TF-IDF Vectorization (based on interests only)
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(users_df['interests'])

    # Compute similarity matrix
    similarity = cosine_similarity(vectors)

    # Find the target user index
    if user_id not in users_df['user_id'].values:
        return {"error": f"User with id {user_id} not found."}

    user_index = users_df[users_df['user_id'] == user_id].index[0]

    # Get similarity scores
    sim_scores = list(enumerate(similarity[user_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top N similar users (excluding the user themselves)
    top_users = [
        {
            "user_id": int(users_df.iloc[i[0]]['user_id']),
            "name": users_df.iloc[i[0]]['name'],
            "branch": users_df.iloc[i[0]]['branch'],
            "year": int(users_df.iloc[i[0]]['year']),
            "similarity_score": round(float(i[1]), 3)
        }
        for i in sim_scores[1:top_n + 1]
    ]

    return {
        "requested_user": {
            "name": users_df.iloc[user_index]['name'],
            "branch": users_df.iloc[user_index]['branch'],
            "year": int(users_df.iloc[user_index]['year'])
        },
        "recommended_users": top_users
    }

# Example root route
@app.get("/")
def home():
    return {"message": "User Recommendation API is running 🚀"}
