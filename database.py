import os
import json
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# File paths
JSON_FILE = "facebook_dataset_draft.json"
EMBED_FILE = "trail_post_embeddings.npy"
REF_FILE = "gpt_reflected_v5.csv"

# 1. Load raw JSON
df = pd.read_json(JSON_FILE)

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

# 3. Optionally load reflections if they exist
if os.path.exists(REF_FILE):
    df_ref = pd.read_csv(REF_FILE)
    df = df.merge(
        df_ref[['text','time','manager_reflection','new_hiker_reflection','experienced_hiker_reflection']],
        on=['text','time'], how='left'
    )

# 4. Load or generate embeddings
if os.path.exists(EMBED_FILE):
    embeddings = np.load(EMBED_FILE)
else:
    print("⚠️ Generating embeddings...")
    texts = df["text"].astype(str).fillna("").tolist()
    embeddings = []
    BATCH_SIZE = 100
    for i in range(0, len(texts), BATCH_SIZE):
        batch = [t for t in texts[i:i+BATCH_SIZE] if t.strip()]
        response = client.embeddings.create(model="text-embedding-3-small", input=batch)
        batch_embeds = [r.embedding for r in response.data]
        embeddings.extend(batch_embeds)
        print(f"✅ Embedded {i + len(batch)} / {len(texts)}")
    embeddings = np.array(embeddings)
    np.save(EMBED_FILE, embeddings)

# 5. Helper functions
def search_similar_posts(query, top_k=5):
    q_embed = client.embeddings.create(model="text-embedding-3-small", input=query).data[0].embedding
    sims = cosine_similarity([q_embed], embeddings[:len(df)])[0]
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
print(f"✅ Loaded {len(df)} posts from {JSON_FILE} with season tags added.")
print(df["season"].value_counts())
