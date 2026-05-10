# Moodify — Intelligent Playlist Curator

A full-stack music intelligence application that segments Spotify playlists by mood using machine learning and corrects classifications using a hybrid LLM pipeline.

---

## What it does

Moodify takes a Spotify playlist CSV, analyses each track across 9 audio dimensions, and clusters songs into moods using a Gaussian Mixture Model with custom emotional feature weighting. It then generates a **Mood Transition Journey** — a sequenced playlist that guides the listener smoothly from one emotional state to another.

A two-phase LLM correction pipeline improves classification accuracy beyond what audio features alone can achieve, and a Streamlit analytics dashboard makes the entire ML pipeline transparent and interpretable.

---

## Architecture

```
Spotify Playlist CSV
        ↓
Ingestion & Validation → Cleaning & Feature Engineering
        ↓
GMM Clustering (9 audio dimensions, custom weights)
        ↓
Two-Phase LLM Correction (Gemini + Genius API)
        ↓
Mood Transition Journey (Valence × Energy interpolation)
        ↓
Streamlit Dashboard + FastAPI Backend
```

---

## Core Components

### 1. GMM Clustering — `mood_engine.py`
- Uses **Gaussian Mixture Model** instead of K-means — GMM gives probabilistic cluster assignments, which better reflects the reality that moods overlap rather than sit in hard boundaries
- Custom emotional feature weighting across 9 audio dimensions:
  - Energy × 1.6
  - Valence × 1.5
  - Danceability × 1.3
  - Plus: BPM, acousticness, speechiness, instrumentalness, liveness, loudness
- Probabilistic assignments mean a track can belong to multiple mood clusters with varying confidence scores

### 2. Mood Transition Journey — `mood_transition.py`
- Maps each mood to a 2D emotional coordinate space using **Valence × Energy** axes
- Uses `np.linspace` to linearly interpolate a path between two mood states
- At each waypoint along the path, the nearest matching track is selected — producing a playlist that transitions smoothly across emotional states rather than jumping abruptly

### 3. Two-Phase LLM Correction Pipeline
- **Phase 1:** Gemini API performs fast batch mood classification using song name, artist, and genre metadata
- **Phase 2:** For tracks with low classification confidence, lyrics are fetched via the Genius API and fed back into the classifier for re-evaluation
- This hybrid approach catches misclassifications that audio features alone miss — for example, a high-energy track with melancholic lyrics

### 4. Streamlit Analytics Dashboard — `dashboard.py`
- **t-SNE visualisation** of the full music universe — shows how tracks cluster in 2D emotional space
- Mood distribution charts across the playlist
- AI confidence breakdown per track
- LLM correction tracking — shows which tracks were reclassified and why
- Makes the ML pipeline fully transparent and auditable

### 5. Recommendation Engine — `similarity_engine.py`
- Cosine similarity matrix across all audio feature vectors
- NetworkX song graph with a 0.85 similarity threshold for edges
- Graph-based recommendation logic surfaces tracks that are emotionally adjacent but not identical

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (port 5000) |
| Frontend / Dashboard | Streamlit |
| ML Clustering | Gaussian Mixture Model (scikit-learn) |
| Dimensionality Reduction | t-SNE |
| LLM Classification | Gemini API |
| Lyrics Retrieval | Genius API |
| Audio Features | Spotify Web API |
| Graph Processing | NetworkX |
| Language | Python |

---

## Running Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your API credentials to environment variables
# SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET
# GEMINI_API_KEY
# GENIUS_API_TOKEN

# 3. Start the FastAPI backend
uvicorn app:app --port 5000

# 4. Launch the Streamlit dashboard (separate terminal)
streamlit run dashboard.py
```

---

## Key Design Decisions

**Why GMM over K-means?**
Moods are not discrete categories — a track can be simultaneously melancholic and energetic. GMM's probabilistic assignments reflect this reality. K-means forces every track into exactly one cluster, which produces worse results for mood-based segmentation.

**Why a two-phase LLM pipeline?**
Audio features capture how a song sounds. Lyrics capture what a song means. These are often misaligned — a fast-tempo track can be deeply sad. Using Gemini for speed and Genius lyrics for accuracy on low-confidence tracks gives the best of both approaches.

**Why Valence × Energy for mood transition?**
These two dimensions are the strongest predictors of perceived mood in music psychology research. Valence captures positivity/negativity; Energy captures intensity. Together they define a 2D space that maps intuitively to emotional experience.

---

## Author

**Tarush Sharma**  
PGDM — IMI Bhubaneswar  
[linkedin.com/in/tarush-sharma-94a955314](https://www.linkedin.com/in/tarush-sharma-94a955314/) · [tarush-101.github.io](https://tarush-101.github.io)
