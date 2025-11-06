from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import motor.motor_asyncio
import pandas as pd
import os

app = FastAPI(title="User Recommendation API (MongoDB-Compatible)")

# ---------------- MongoDB Setup ----------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
print("MONGO_URI =", MONGO_URI)

client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client.college_community
users_collection = db.users

# ✅ Startup event to verify MongoDB connection
@app.on_event("startup")
async def startup_db_client():
    try:
        await client.admin.command('ping')
        print("✅ MongoDB connected successfully!")
    except Exception as e:
        print("❌ MongoDB connection failed:", e)

# ---------------- Request/Response Models ----------------
class RecommendRequest(BaseModel):
    user_id: str
    top_n: int = 10  # Optional, defaults to 10 if not provided

class RecommendedUser(BaseModel):
    user_id: str
    name: str
    email: str
    branch: str
    year: str
    interests: List[str]
    similarity_score: float

# ---------------- Routes ----------------
@app.post("/recommend_users/")
async def recommend_users_api(request: RecommendRequest):
    user_id = request.user_id
    top_n = request.top_n

    try:
        # Fetch all users from MongoDB
        users_cursor = users_collection.find({}, {"_id": 1, "name": 1, "email": 1, "branch": 1, "year": 1, "interests": 1})
        users_list = await users_cursor.to_list(length=1000)

        if not users_list:
            return {"error": "No users found in database."}

        # Map _id to string user_id and remove _id
        for user in users_list:
            user["user_id"] = str(user["_id"])
            user.pop("_id", None)

        # Validate user_id
        if user_id not in [u["user_id"] for u in users_list]:
            return {"error": f"User with id {user_id} not found."}

        # Convert interests to text for TF-IDF
        users_df = pd.DataFrame(users_list)
        users_df["interests_text"] = users_df["interests"].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))

        # TF-IDF similarity calculation
        vectorizer = TfidfVectorizer(stop_words="english")
        vectors = vectorizer.fit_transform(users_df["interests_text"])
        similarity = cosine_similarity(vectors)

        # Index of requested user
        user_index = users_df[users_df["user_id"] == user_id].index[0]
        sim_scores = sorted(list(enumerate(similarity[user_index])), key=lambda x: x[1], reverse=True)

        # Top N recommended users (excluding self)
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
            for i in sim_scores[1: top_n + 1]  # skip self
        ]

        requested_user = {
            "user_id": users_df.iloc[user_index]["user_id"],
            "name": users_df.iloc[user_index]["name"],
            "email": users_df.iloc[user_index]["email"],
            "branch": users_df.iloc[user_index]["branch"],
            "year": users_df.iloc[user_index]["year"],
            "interests": users_df.iloc[user_index]["interests"]
        }

        return {"requested_user": requested_user, "recommended_users": top_users}

    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def home():
    return {"message": "User Recommendation API is running 🚀"}
