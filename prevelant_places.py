import os
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
import asyncio
from collections import Counter
from tqdm.asyncio import tqdm_asyncio
from supabase import create_client, ClientOptions

# ------------------ LOAD ENV ------------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

client = AsyncOpenAI(api_key=OPENAI_KEY)

# ------------------ CONNECT TO SUPABASE ------------------
supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY,
    options=ClientOptions(schema="public")
)

response = supabase.table("supa_fb_data").select("text, facebookUrl, time").execute()
supabase.auth.sign_out()

df = pd.DataFrame(response.data)
print(f"Loaded {len(df)} posts.")

# ------------------ GPT LOCATION EXTRACTION ------------------

MODEL = "gpt-4o-mini"

async def extract_locations_and_snippet(text):
    if not text or not text.strip():
        return {"locations": [], "snippet": ""}

    prompt = f"""
Your task: Identify real geographic locations mentioned in the text.
Return **valid JSON ONLY** in this format:

{{
  "locations": ["Location1", "Location2"],
  "snippet": "very short (5-10 word) excerpt showing the mention"
}}

Rules:
- Only include REAL places (trails, mountains, towns, national forests, etc.)
- If no locations exist, return {{ "locations": [], "snippet": "" }}

Text:
{text}
"""

    try:
        res = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = res.choices[0].message.content.strip()

        # FORCE CLEAN JSON
        if not raw.startswith("{"):
            return {"locations": [], "snippet": ""}

        return eval(raw)

    except Exception as e:
        print("GPT error:", e)
        return {"locations": [], "snippet": ""}


async def process_all(df):
    all_locations = []
    per_post_locations = []
    snippets = []

    print("\nExtracting locations...\n")
    for text in tqdm_asyncio(df["text"], total=len(df)):
        result = await extract_locations_and_snippet(text)
        per_post_locations.append(result["locations"])
        snippets.append(result["snippet"])
        all_locations.extend(result["locations"])

    return per_post_locations, snippets, Counter(all_locations)


async def main():
    per_post_locations, snippets, freq = await process_all(df)

    df["locations"] = per_post_locations
    df["snippet"] = snippets

    df.to_csv("supabase_locations.csv", index=False)

    print("\n============================")
    print("TOP 20 MOST MENTIONED LOCATIONS")
    print("============================\n")

    for loc, count in freq.most_common(20):
        print(f"{loc}: {count}")

asyncio.run(main())
