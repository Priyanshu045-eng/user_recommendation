from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = FastAPI(title="User Recommendation API (MongoDB-Compatible)")

# ------------------------------------------------------
# 🔹 Data Model (matching your MongoDB Mongoose schema)
# ------------------------------------------------------
class User(BaseModel):
    user_id: int                   # Used only for identification in API
    name: str
    email: str
    branch: str
    year: str                      # e.g., "1st", "2nd", "3rd", "4th"
    interests: List[str]           # List of strings like ["AI", "Web Development"]

# ------------------------------------------------------
# 🔹 API Endpoint: Recommend similar users
# ------------------------------------------------------
@app.post("/recommend_users/")
def recommend_users_api(users: List[User], user_id: int, top_n: int = 10):
    # Convert to DataFrame
    users_df = pd.DataFrame([u.dict() for u in users])

    # Ensure valid user_id
    if user_id not in users_df["user_id"].values:
        return {"error": f"User with id {user_id} not found."}

    # Join list of interests into a single string for TF-IDF processing
    users_df["interests_text"] = users_df["interests"].apply(lambda x: " ".join(x))

    # TF-IDF Vectorization based on interests
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(users_df["interests_text"])

    # Compute cosine similarity matrix
    similarity = cosine_similarity(vectors)

    # Find target user index
    user_index = users_df[users_df["user_id"] == user_id].index[0]

    # Compute similarity scores for all other users
    sim_scores = list(enumerate(similarity[user_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top N similar users (excluding the target user itself)
    top_users = [
        {
            "user_id": int(users_df.iloc[i[0]]["user_id"]),
            "name": users_df.iloc[i[0]]["name"],
            "email": users_df.iloc[i[0]]["email"],
            "branch": users_df.iloc[i[0]]["branch"],
            "year": users_df.iloc[i[0]]["year"],
            "interests": users_df.iloc[i[0]]["interests"],
            "similarity_score": round(float(i[1]), 3)
        }
        for i in sim_scores[1: top_n + 1]
    ]

    # Return result
    return {
        "requested_user": {
            "user_id": int(users_df.iloc[user_index]["user_id"]),
            "name": users_df.iloc[user_index]["name"],
            "email": users_df.iloc[user_index]["email"],
            "branch": users_df.iloc[user_index]["branch"],
            "year": users_df.iloc[user_index]["year"],
            "interests": users_df.iloc[user_index]["interests"]
        },
        "recommended_users": top_users
    }

# ------------------------------------------------------
# 🔹 Root Route
# ------------------------------------------------------
@app.get("/")
def home():
    return {"message": "User Recommendation API (MongoDB Compatible) is running 🚀"}
