import os
import json
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

from supabase import create_client
from supabase.client import ClientOptions

from transformers import pipeline

load_dotenv()
url = os.environ.get('SUPABASE_URL')
key = os.environ.get('SUPABASE_KEY')
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------- ESTABLISH DATABASE CONNECTION / RETRIEVE DATA ---------
supabase = create_client(
    url,
    key,
    options=ClientOptions(
        postgrest_client_timeout=10,
        storage_client_timeout=10,
        schema="public",
    )
)

#data = supabase.table("Facebook").select("text").execute()
data = supabase.table("supa_fb_data").select("text, facebookUrl, time").execute()
'''
for data_obj in data.data:
    custom_string = f"{data_obj['raw']}"
    print(custom_string)
'''

supabase.auth.sign_out()
# --------- FINISH DATA RETRIEVAL ---------

df = pd.DataFrame(data.data)

# 2. Add season column
def infer_season(date_str):
    if not date_str or pd.isna(date_str):
        return "unknown"
    date = pd.to_datetime(date_str, errors="coerce")
    if pd.isna(date):
        return "unknown"
    month = date.month
    if month in [12,1,2]: return "winter"
    if month in [3,4,5]: return "spring"
    if month in [6,7,8]: return "summer"
    if month in [9,10,11]: return "fall"
    return "unknown"

df["season"] = df["time"].apply(infer_season)

# 3. Emotion Determination
def get_emotions():
    classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None)

    def get_top_emotion(text):
        """
        Detect the single dominant emotion and its score.
        Returns:
            emotion (str): the top emotion label
            score (float): the confidence score (0–1)
        """
        try:
            results = classifier(text[:512])[0]
            # Get the highest score
            top = max(results, key=lambda r: r["score"])
            return pd.Series([top["label"].lower(), round(top["score"], 4)])
        except Exception:
            return pd.Series(["error", 0.0])

    df["text"].apply(get_top_emotion)
    # Apply to all posts
    df[["dominant_emotion", "emotion_score"]] = df["text"].apply(get_top_emotion)

    # Trail vibe keywords
    trail_keywords = {
    "chill": ["relaxing", "easy", "calm", "peaceful", "simple", "casual"],
    "social": ["group", "friends", "people", "together", "meetup", "funny"],
    "fun": ["enjoyable", "exciting", "awesome", "great", "adventure", "cool"],
    "challenging": ["steep", "hard", "tough", "long", "climb", "intense"],
    "scenic": ["view", "beautiful", "scenery", "mountain", "lake", "waterfall"],
    "wildlife": ["animals", "birds", "deer", "bear", "squirrel", "nature"]
    }

    def tag_trail_keywords(text):
        #"""Return tags like 'chill', 'fun', 'scenic', etc. based on keywords in post."""
        #text_lower = text.lower()
        matched_tags = []

        for tag, keywords in trail_keywords.items():
            if text == None:
                matched_tags.append("general")
            elif any(word in text.lower() for word in keywords):
                matched_tags.append(tag)

        if not matched_tags:
            matched_tags.append("general")
        return ", ".join(matched_tags)

    df["trail_tags"] = df["text"].apply(tag_trail_keywords)

    # Print counts
    emotion_counts = df["dominant_emotion"].value_counts()
    print("Emotion counts:")
    print(emotion_counts)

# 3. Optionally load reflections if they exist
# ---- CONFIG ----
EMBED_MODEL = "text-embedding-3-large"
EMBED_FILE = "trail_post_embeddings.npy"
DATA_FILE = "facebook_dataset_draft.json"

# ---- CLEAN DATA ----
df = df[df["text"].astype(str).str.strip().ne("")]
df = df.reset_index(drop=True)

texts = df["text"].astype(str).tolist()

# ---- GENERATE NEW EMBEDDINGS ----
print("⚠️ Regenerating embeddings from Supabase data...")

embeddings = []
BATCH_SIZE = 100

for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i:i+BATCH_SIZE]
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=batch
    )
    batch_embeds = [item.embedding for item in response.data]
    embeddings.extend(batch_embeds)
    print(f"Embedded {i + len(batch)} / {len(texts)}")

embeddings = np.array(embeddings)

np.save(EMBED_FILE, embeddings)
df.to_json(DATA_FILE, orient="records")

# ---- SEARCH ----
def search_similar_posts(query, top_k=12):
    q_embed = client.embeddings.create(model=EMBED_MODEL, input=query).data[0].embedding
    sims = cosine_similarity([q_embed], embeddings)[0]
    top_idx = sims.argsort()[-top_k:][::-1]
    return df.iloc[top_idx].assign(similarity=sims[top_idx])

def get_post_reflections(row):
    return {
        "manager": row.get("manager_reflection"),
        "new_hiker": row.get("new_hiker_reflection"),
        "experienced_hiker": row.get("experienced_hiker_reflection")
    }

def get_all_posts(limit=None):
    return df.head(limit) if limit else df

def get_post_by_index(idx):
    if 0 <= idx < len(df):
        row = df.iloc[idx]
        return {
            "text": row["text"],
            "trail_tags": row.get("trail_tags"),
            "dominant_emotion": row.get("dominant_emotion"),
            "season": row.get("season"),
            "reflections": get_post_reflections(row)
        }
    return None

# Done
#print(f"✅ Loaded {len(df)} posts from {JSON_FILE} with season tags added.")
print(df["season"].value_counts())
