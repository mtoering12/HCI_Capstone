# TrailPulse Dashboard

TrailPulse is an interactive web application that visualizes hiking trail data through emotional summaries, seasonal filtering, and map-based exploration. The goal is to connect hikers’ experiences with trail metadata, helping users discover trails that fit their mood, physical goals, or social needs.

## Features
- Interactive Map (Leaflet.js): Displays trail pins with popups showing AI-generated summaries and recent sightings.
- Trail Cards: Dynamic list of trails with metadata (distance, difficulty, tags).
- Search & Filters: Find trails by mood, location, or features (e.g., water, scrambles).
- AI Chatbot: Fill-in-the-blank templates for natural queries like:
  - “I want to visit {location}…”
  - “I am {self_desc}, and I want {goal}…”
  - “What are people saying this {season_topic}?”
- Seasonal Filtering: Highlights trails based on seasonal conditions and user sentiment.
- Database Integration (Supabase):** Scraped social media data (Facebook) structured into a backend database, connected to the frontend for live updates.

## Tech Stack
- Frontend: HTML, CSS, JavaScript
- Mapping: Leaflet.js
- Backend: Supabase (PostgreSQL)
- Data Collection: Facebook scraping → CSV → Supabase
- AI Integration: Chatbot templates + AI-generated emotional summaries

## Run
- py -m venv venv
- venv/Scripts/Activate (Windows)
- pip install -r requirements.txt
- flask --app server run
