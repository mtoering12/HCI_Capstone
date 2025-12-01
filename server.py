from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from openai import OpenAI
from supabase import create_client, Client
from database import search_similar_posts
from database import get_emotions
import os
from dotenv import load_dotenv

# Load OpenAI key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

supabase: Client = create_client(
    os.getenv("SUPABASE_URL"), 
    os.getenv("SUPABASE_KEY")
)

db = SQLAlchemy()

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://postgres:HCICapstoneProject1234@db.ksrmbbmdoqiybuqshrre.supabase.co:5432/postgres"

db.init_app(app)

# Creating database table if not already created
class SupaFbData(db.Model):
    uuid = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String, unique=False, nullable=True)
    facebookUrl = db.Column(db.String, unique=False, nullable=True)
    id = db.Column(db.String, unique=False, nullable=True)
    time = db.Column(db.DateTime, unique=False, nullable=True)
with app.app_context():
    db.create_all()

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

# Serve static file for links
@app.route("/trail.html", methods=["GET"])
def get_trail_file():
    return render_template("trail.html")

@app.route("/other.html", methods=["GET"])
def get_mood_summaries_file():
    get_emotions()
    return render_template("other.html")

@app.route("/feelings.html", methods=["GET"])
def get_feelings_file():
    return render_template("feelings.html")

@app.route("/pictures.html", methods=["GET"])
def get_pictures_file():
    return render_template("pictures.html")

# GET request for ALL posts
@app.route("/db", methods=["GET"])
def get_posts():
    response = supabase.table("supa_fb_data").select("*").execute()

    return jsonify(response.data)

# GET request for single post, by UUID
@app.route("/db/uuid", methods=["GET"])
def get_post():
    post_uuid = request.args.get('uuid')

    response = supabase.table("supa_fb_data").select("*").eq("uuid", post_uuid).execute()

    return jsonify(response.data)

# POST request for insertion into database
@app.route("/db", methods=["POST"])
def push_post():
    data = request.get_json()

    response = supabase.table("supa_fb_data").insert(
        {"text": data.get("text"), 
         "facebookUrl": None, 
         "id": None, 
         "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).execute()

    return jsonify(response.data)

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
#if __name__ == "__main__":
#    app.run(debug=True)
