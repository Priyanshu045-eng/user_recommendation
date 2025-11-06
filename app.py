from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import motor.motor_asyncio
import pandas as pd
import os

app = FastAPI(title="User Recommendation API (MongoDB-Compatible)")

# MongoDB connection (replace with your MongoDB URI)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client.college_community
users_collection = db.users

# Pydantic model for response
class RecommendedUser(BaseModel):
    user_id: str
    name: str
    email: str
    branch: str
    year: str
    interests: List[str]
    similarity_score: float

@app.post("/recommend_users/")
async def recommend_users_api(user_id: str, top_n: int = 10):
    # Fetch all users from MongoDB
    users_cursor = users_collection.find({}, {"_id": 1, "name": 1, "email": 1, "branch": 1, "year": 1, "interests": 1})
    users_list = await users_cursor.to_list(length=1000)

    if not users_list:
        return {"error": "No users found in database."}

    # Map MongoDB _id to string user_id
    for user in users_list:
        user["user_id"] = str(user["_id"])

    # Check if user exists
    if user_id not in [u["user_id"] for u in users_list]:
        return {"error": f"User with id {user_id} not found."}

    # Create DataFrame
    users_df = pd.DataFrame(users_list)
    users_df["interests_text"] = users_df["interests"].apply(lambda x: " ".join(x))

    # TF-IDF and similarity
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(users_df["interests_text"])
    similarity = cosine_similarity(vectors)

    user_index = users_df[users_df["user_id"] == user_id].index[0]
    sim_scores = list(enumerate(similarity[user_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    top_users = [
        {
            "user_id": users_df.iloc[i[0]]["user_id"],
            "name": users_df.iloc[i[0]]["name"],
            "email": users_df.iloc[i[0]]["email"],
            "branch": users_df.iloc[i[0]]["branch"],
            "year": users_df.iloc[i[0]]["year"],
            "interests": users_df.iloc[i[0]]["interests"],
            "similarity_score": round(float(i[1]), 3)
        }
        for i in sim_scores[1: top_n + 1]
    ]

    return {
        "requested_user": users_df.iloc[user_index].to_dict(),
        "recommended_users": top_users
    }

@app.get("/")
def home():
    return {"message": "User Recommendation API is running 🚀"}
