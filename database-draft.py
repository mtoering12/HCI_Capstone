import json
import pandas as pd
import os
from dotenv import load_dotenv
from transformers import pipeline
import openai
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load dataset
json_file_path = "facebook_dataset_draft.json"
with open(json_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

if "text" not in df.columns:
    raise ValueError(f"'text' field not found in JSON keys: {list(df.columns)}")

# Adding the seasons to the dataframe for filtering
df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
def get_season(month):
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    elif month in [9, 10, 11]:
        return "fall"
    else:
        return None

df["season"] = df["time"].dt.month.apply(get_season)

# Emotion detection
classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)

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
    """Return tags like 'chill', 'fun', 'scenic', etc. based on keywords in post."""
    text_lower = text.lower()
    matched_tags = []

    for tag, keywords in trail_keywords.items():
        if any(word in text_lower for word in keywords):
            matched_tags.append(tag)

    if not matched_tags:
        matched_tags.append("general")
    return ", ".join(matched_tags)

df["trail_tags"] = df["text"].apply(tag_trail_keywords)

# Print counts
emotion_counts = df["dominant_emotion"].value_counts()
print("Emotion counts:")
print(emotion_counts)

# OpenAI API setup
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai.api_key)

selected_posts = df.reset_index(drop=True)
print(f"Processing {len(selected_posts)} posts total.")

# Define GPT prompts for each role
roles = {
    "manager": (
        "You are a paid trail manager for a local parks department."
        "You coordinate volunteers, maintain trail quality, and communicate updates to the public and to park authorities. "
        "Due to other tasks, you are not actively monitoring social media and online discussion related to your trail often, checking once or twice a day.."
        "Your goals are to maintain and keep trails safe, maximze maintaince that reduces risks and improve the hikers experience, and mainintain trust to the hiekrs"
        "You appreciate posts that provide useful feedback, highlight maintenance issues, or promote community involvement."
    ),
    "new_hiker": (
        "You are a new hiker who recently started exploring local trails."
        "You are also a college student that is looking to pick up hiking as a new hobby, and to primarily spend this hobby to make new friends"
        "You prefer more safer, easier, beginner-friendly hikes that will allow you to enjoy your time with friends and be socialable, and maximizing your opportunity to socialize"
        "You primarily only look for posts that will help you find or plan your next hike, before you do decide to hike"
        "You prefer trails and locations with an easier barrier to entry and are worthwhile, as you have a limited amount of time"
        ""
    ),
    "experienced_hiker": (
        "You are an experienced hiker who knows many trails and may contribute actively to the hiking community. "
        "You value posts that share genuine experiences, technical trail details, and scenic recommendations. "
        "You’re less impressed by vague or repetitive posts."
        "You are checking groups also for detailed trial conditions, new locations, or safety warnings"
    ),
}


def select_fill_in_prompt():
    print("\nChoose a prompt:")
    for i, p in enumerate(FILL_IN_TEMPLATES, start=1):
        print(f"{i}. {p['title']}")
    print("Type a number to continue, or 'exit' to quit.\n")

    while True:
        choice = input("Select a number: ").strip().lower()
        if choice in {"exit", "quit"}:
            return None, None
        if choice.isdigit() and 1 <= int(choice) <= len(FILL_IN_TEMPLATES):
            break
        print("Invalid choice. Please enter a number from the list.\n")

    prompt = FILL_IN_TEMPLATES[int(choice) - 1]
    responses = {}
    for ph in prompt["placeholders"]:
        responses[ph] = input(f"Fill in '{ph.replace('_', ' ')}': ").strip()

    query = prompt["template"].format(**responses)
    return query, prompt["role"]


# def ask_gpt(role_name, role_desc, post_text, post_time, post_tags):
#     """Ask GPT if a post is interesting and detect sarcasm."""
#     prompt = f"""{role_desc}

# You are reviewing a social media post from a hiking group

# Post (time: {post_time}):
# "{post_text}"

# Tags: {post_tags}

# Instructions:
# Answer the questions based on your role's perspective.
# Use the post text and timing to guide your response. When relevant, quote short phrases directly from the post.

# Questions:
# 1. Is this post interesting to you? (yes or no)
# 2. Why or why not? Explain in 2–3 sentences from your role’s perspective. Mention any key insights or takeaways, quoting the post if appropriate.
# 3. Is there any sarcasm detected or insincerity? (yes or no)
# Provide short, clear answers, each on a new line.
# End your response with: "— Reflection from {role_name.capitalize()}"
# """

#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.6
#         )
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         return f"Error: {str(e)}"


# Collect GPT reflections
# reflections = []

# for i, row in tqdm(selected_posts.iterrows(), total=len(selected_posts)):
#     post_text = row["text"]
#     post_time = row["time"]
#     post_tags = row["trail_tags"]
#     result = {"text": post_text, "time": post_time, "trail_tags": post_tags}

#     for role_name, desc in roles.items():
#         reasoning = ask_gpt(role_name, desc, post_text, post_time, post_tags)
#         result[f"{role_name}_reflection"] = reasoning
#         time.sleep(0.3)  # slight delay to prevent rate limits

#     reflections.append(result)

# # Merge reflections into dataframe
# ref_df = pd.DataFrame(reflections)
# combined_df = selected_posts.merge(ref_df, on=["text", "time", "trail_tags"], how="left")
# output_path = "gpt_reflected_hiking_posts_with_locations.csv"
# combined_df.to_csv(output_path, index=False, encoding="utf-8")
# print(f"Saved enhanced data with GPT reflections to {output_path}")

import asyncio
import aiohttp
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=openai.api_key)

async def ask_gpt_async(role_name, role_desc, post_text, post_time, post_tags):
    """Asynchronous GPT call."""
    prompt = f"""{role_desc}

You are reviewing a social media post from a hiking group

Post (time: {post_time}):
"{post_text}"

Tags: {post_tags}

Instructions:
Answer the questions based on your role's perspective.
Use the post text and timing to guide your response. When relevant, quote short phrases directly from the post.

Questions:
1. Is this post interesting to you? (yes or no)
2. Why or why not? Explain in 2–3 sentences from your role’s perspective. Mention any key insights or takeaways, quoting the post if appropriate.
3. Is there any sarcasm detected or insincerity? (yes or no)
Provide short, clear answers, each on a new line.
End your response with: "— Reflection from {role_name.capitalize()}"
"""
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

async def process_post(row):
    """Process all roles for a single post."""
    post_text = row["text"]
    post_time = row["time"]
    post_tags = row["trail_tags"]
    result = {"text": post_text, "time": post_time, "trail_tags": post_tags}

    tasks = [
        ask_gpt_async(role_name, desc, post_text, post_time, post_tags)
        for role_name, desc in roles.items()
    ]

    reflections = await asyncio.gather(*tasks)
    for (role_name, _), reflection in zip(roles.items(), reflections):
        result[f"{role_name}_reflection"] = reflection
    return result

async def process_all_posts(df, concurrency=20):
    """Run GPT calls concurrently in batches."""
    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async def sem_task(row):
        async with semaphore:
            return await process_post(row)

    tasks = [sem_task(row) for _, row in df.iterrows()]
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        result = await f
        results.append(result)

    return results

# Run async batch processing
reflections = asyncio.run(process_all_posts(selected_posts))

# Merge results
ref_df = pd.DataFrame(reflections)
combined_df = selected_posts.merge(ref_df, on=["text", "time", "trail_tags"], how="left")

output_path = "gpt_reflected_v5.csv"
combined_df.to_csv(output_path, index=False, encoding="utf-8")
print(f"Saved enhanced data with GPT reflections to {output_path}")

from openai import OpenAI
client = OpenAI(api_key=openai.api_key)

# === Interactive Chat Interface ===
print("\nTrail Insights Chat. Ask questions about hiking posts, emotions, or roles.")
print("Type 'exit' to quit.\n")

# Check or create embeddings
EMBED_FILE = "trail_post_embeddings.npy"

if os.path.exists(EMBED_FILE):
    print("Loaded existing embeddings.")
    embeddings = np.load(EMBED_FILE)
else:
    print("Generating embeddings for posts (first time only)...")
texts = combined_df["text"].astype(str).fillna("").tolist()
embeddings = []

BATCH_SIZE = 100
for i in range(0, len(texts), BATCH_SIZE):
    batch = [t for t in texts[i:i+BATCH_SIZE] if t.strip()]  # filter empties
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=batch
    )
    batch_embeds = [r.embedding for r in response.data]
    embeddings.extend(batch_embeds)
    print(f"✅ Embedded {i + len(batch)} / {len(texts)}")

embeddings = np.array(embeddings)
np.save(EMBED_FILE, embeddings)



# Function to search most relevant posts
def search_similar_posts(query, top_k=5, season=None):
    # Filter by season if provided
    df_to_search = (
        combined_df[combined_df["season"].str.lower() == season]
        if season else combined_df
    ).reset_index(drop=True)  # Ensure index alignment

    # Get query embedding
    q_embed = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    # Slice embeddings to the same length as filtered dataframe
    sims = cosine_similarity([q_embed], embeddings[:len(df_to_search)])[0]

    # Get top matches
    top_idx = sims.argsort()[-top_k:][::-1]

    # Return results with similarity values
    return df_to_search.iloc[top_idx].assign(similarity=sims[top_idx])


# Function to ask GPT based on relevant context
def answer_query(query, role=None):
    top_posts = search_similar_posts(query)
    context = "\n\n".join(
        f"- {row['text']} (tags: {row['trail_tags']}, emotion: {row['dominant_emotion']})"
        for _, row in top_posts.iterrows()
    )

    role_prompt = f"from the perspective of a {role}" if role else ""
    prompt = f"""
You are an AI assistant summarizing hiking posts {role_prompt}.
Use the context below to answer the user's question clearly and accurately.
`   `   If relevant, mention specific posts or patterns.

Context:
{context}

Question: {query}

Answer:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()


# chat loop in the terminal
# === Unified Trail Insights Chat ===
messages = [
    {
        "role": "system",
        "content": (
            "You are Trail Insights, an assistant that helps users explore patterns "
            "and themes in hiking-related social media posts. "
            "Use retrieved post context for accurate answers. "
            "Combine perspectives of hikers, managers, and outdoor enthusiasts "
            "to provide balanced insights about trail conditions, emotions, and community trends."
        )
    }
]

print("\nTrail Insights Chat — ask questions about hiking posts, emotions, or themes.")
print("Type 'exit' to quit.\n")

while True:
    query, role = select_fill_in_prompt()
    if query is None:
        print("Exiting chat.")
        break

    # Retrieve top similar posts for grounding
    print("\nSearching relevant posts...\n")
    try:
        top_posts = search_similar_posts(query)
    except Exception as e:
        print(f"Error finding similar posts: {e}")
        continue

    context = "\n\n".join(
        f"- {row['text']} (tags: {row['trail_tags']}, emotion: {row['dominant_emotion']})"
        for _, row in top_posts.iterrows()
    )

    # Build message using conversation history
    user_message = f"""
Context:
{context}

Question: {query}

Answer clearly and factually using the context above. If you see emotional or social patterns, describe them.
"""
    messages.append({"role": "user", "content": user_message})

    print("\nThinking...\n")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.5,
        )
        answer = response.choices[0].message.content.strip()
        print(f"💬 {answer}\n")
        messages.append({"role": "assistant", "content": answer})
    except Exception as e:
        print(f"Error: {e}\n")
