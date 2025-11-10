from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from database import search_similar_posts
import os
from dotenv import load_dotenv

# Load OpenAI key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__, static_folder="static", template_folder="templates")

# ----------------------------
# Fill-in templates for frontend
# ----------------------------
FILL_IN_TEMPLATES = [
    {"title": "I want to visit...", "template": "I want to visit {location}. What can you tell me about trails or posts mentioning it?", "placeholders": ["location"], "role": None},
    {"title": "I am... and I want...", "template": "I am {self_desc}, and I want {goal}. Which trails or posts might fit?", "placeholders": ["self_desc", "goal"], "role": None},
    {"title": "What are people saying this season about...", "template": "What are people saying this {season_topic}? Summarize key emotions or trail issues.", "placeholders": ["season_topic"], "role": None},
    {"title": "Tell me about trails that are...", "template": "Tell me about trails that are {adjective}.", "placeholders": ["adjective"], "role": None},
]

# ----------------------------
# Routes
# ----------------------------

@app.route("/")
def home():
    """Serve the main HTML page."""
    return render_template("index.html")


@app.route("/templates", methods=["GET"])
def list_templates():
    """Return available fill-in templates as JSON."""
    return jsonify(FILL_IN_TEMPLATES)


@app.route("/ask", methods=["POST"])
def ask_trailpulse():
    """Handle GPT queries using filled-in template and retrieved post context."""
    data = request.get_json()

    query = data.get("query")
    role = data.get("role")
    template_index = data.get("template_index")
    placeholders = data.get("placeholders", {})

    # Format query if using a template
    if template_index is not None and query is None:
        try:
            template = FILL_IN_TEMPLATES[int(template_index)]
            query = template["template"].format(**placeholders)
            role = template.get("role")
        except Exception as e:
            return jsonify({"error": f"Invalid template or placeholders: {e}"}), 400

    if not query:
        return jsonify({"error": "Missing query"}), 400

    # Retrieve top posts for context
    try:
        top_posts = search_similar_posts(query)
    except Exception as e:
        return jsonify({"error": f"Search failed: {e}"}), 500

    context = "\n\n".join(
        f"- {row['text']} (tags: {row.get('trail_tags')}, emotion: {row.get('dominant_emotion')})"
        for _, row in top_posts.iterrows()
    )

    # Build GPT prompt
    role_prompt = f"from the perspective of a {role}" if role else ""
    prompt = f"""
You are TrailPulse, an AI assistant that helps users explore hiking community posts {role_prompt}.
Use the posts below to inform your answer.

Context:
{context}

Question: {query}

Answer clearly and factually using the context above. If you see emotional or social patterns, describe them.
"""

    # Call GPT
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        answer = response.choices[0].message.content.strip()
        return jsonify({"response": answer, "context": context})
    except Exception as e:
        return jsonify({"error": f"GPT call failed: {e}"}), 500


# ----------------------------
# Run server
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
