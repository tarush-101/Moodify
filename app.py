from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import shutil
from pathlib import Path

import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Phase-1 imports
from src.ingestion.loader import load_csv
from src.ingestion.validator import validate_schema
from src.ingestion.cleaner import clean_dataframe

# Phase-2 imports
from src.mood.mood_engine import run_mood_clustering
from src.mood.mood_mapping import attach_mood_names
from src.mood.playlist_filter import get_tracks_for_mood
from src.mood.mood_transition import generate_mood_journey
from src.mood.llm_mood_corrector import correct_dataframe_moods

# Recommendation system
from src.recommendation.similarity_engine import build_similarity_matrix, get_similar_songs
from src.recommendation.mood_expansion import recommend_from_mood
from src.recommendation.mood_journey_recommender import recommend_mood_transition
from src.recommendation.song_graph import build_song_graph
from src.recommendation.spotify_recommender import recommend_from_spotify
from src.recommendation.spotify_search_recommender import recommend_from_spotify_search


# --- Spotify Configuration ---
# Paste your API credentials inside the quotes!
CLIENT_ID = 'xyz'
CLIENT_SECRET = 'xyz'
REDIRECT_URI = 'xyz'
SCOPE = 'playlist-modify-public playlist-modify-private'

# Initialize Spotify OAuth manager globally
sp_oauth = SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope=SCOPE
)

# --- API Request Schema ---
class PlaylistRequest(BaseModel):
    mood_name: str
    track_uris: List[str]


# --- FastAPI Setup ---
app = FastAPI(title="Moodify API")

# Allow HTML frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_PATH = Path("temp_upload.csv")


@app.get("/")
def root():
    return {"message": "Moodify backend running"}


# --- NEW: Spotify Authentication Endpoints ---

@app.get("/login")
def login():
    """Redirects the user to Spotify's authorization page."""
    auth_url = sp_oauth.get_authorize_url()
    return RedirectResponse(auth_url)


@app.get("/callback")
def callback(request: Request):
    """Catches the Spotify redirect and exchanges the code for a token."""
    code = request.query_params.get("code")
    if not code:
        raise HTTPException(status_code=400, detail="Authorization code not found.")
    
    # This automatically saves the token to your .cache file
    sp_oauth.get_access_token(code)
    return {"message": "Successfully authenticated with Spotify! You can close this window."}


# --- Core App Endpoints ---

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Receive CSV → run ML → return playlists grouped by mood."""
    try:
        # Save uploaded file temporarily
        with UPLOAD_PATH.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # ---------- Phase-1 ----------
        df = load_csv(str(UPLOAD_PATH))
        validate_schema(df)
        df = clean_dataframe(df)

        # ---------- Phase-2 ----------
        df, centroids = run_mood_clustering(df)
        df = attach_mood_names(df, centroids)

        # ---------- LLM correction ----------
        df = correct_dataframe_moods(df)

        # Store dataframe and models in app state
        app.state.dataframe = df
        app.state.similarity_matrix = build_similarity_matrix(df)
        app.state.song_graph = build_song_graph(df)

        # ---------- Build playlists ----------
        moods = df["mood"].dropna().unique()
        playlists = {}

        for mood in moods:
            mood_tracks = get_tracks_for_mood(df, mood)
            playlists[mood] = mood_tracks.to_dict(orient="records")

        app.state.playlists = playlists

        return {
            "status": "success",
            "moods": list(moods),
            "playlists": playlists,
        }

    except Exception as e:
        print("UPLOAD ERROR:", e)
        raise HTTPException(status_code=400, detail=str(e))

    finally:
        # Clean up temp file if it exists
        if UPLOAD_PATH.exists():
            UPLOAD_PATH.unlink()


@app.post("/create-playlist")
def create_spotify_playlist(request: PlaylistRequest):

    try:

        print(f"Authenticating with Spotify to create '{request.mood_name}' playlist...")

        sp = spotipy.Spotify(
            auth_manager=SpotifyOAuth(
                client_id=CLIENT_ID,
                client_secret=CLIENT_SECRET,
                redirect_uri=REDIRECT_URI,
                scope=SCOPE
            )
        )

        playlist_name = f"Moodify: {request.mood_name.capitalize()}"
        playlist_description = f"A dynamically generated playlist for your {request.mood_name} mood!"

        print(f"Creating playlist '{playlist_name}'...")

        payload = {
            "name": playlist_name,
            "public": False,
            "description": playlist_description
        }

        playlist = sp._post("me/playlists", payload=payload)

        playlist_id = playlist["id"]

        app.state.last_playlist_id = playlist_id

        print("Adding songs to the playlist...")

        uris = request.track_uris

        for i in range(0, len(uris), 100):

            chunk = uris[i:i + 100]

            items_payload = {"uris": chunk}

            sp._post(f"playlists/{playlist_id}/items", payload=items_payload)

        print("Playlist generated successfully!")

        return {
            "status": "success",
            "playlist_id": playlist_id,
            "spotify_url": f"https://open.spotify.com/playlist/{playlist_id}"
        }

    except Exception as e:

        raise HTTPException(status_code=400, detail=str(e))   

@app.post("/mood_journey")
async def mood_journey(start_mood: str, end_mood: str):
    if not hasattr(app.state, "dataframe"):
        raise HTTPException(status_code=400, detail="Data not loaded")

    df = app.state.dataframe
    journey_df = generate_mood_journey(df, start_mood, end_mood)
    journey_df = journey_df.replace({float("nan"): None})
    tracks = journey_df.to_dict(orient="records")
    return {"tracks": tracks}


@app.get("/similar_song")
def similar_song(track_name: str):
    df = app.state.dataframe
    sim = app.state.similarity_matrix
    songs = get_similar_songs(df, sim, track_name)
    return songs.to_dict(orient="records")


@app.get("/recommend_mood")
def recommend_mood(mood: str):
    df = app.state.dataframe
    songs = recommend_from_mood(df, mood)
    return songs.to_dict(orient="records")


@app.get("/recommend_journey")
def recommend_journey(start_mood: str, end_mood: str):
    df = app.state.dataframe
    playlist = recommend_mood_transition(df, start_mood, end_mood)
    return playlist.to_dict(orient="records")


@app.get("/discover")
def discover(mood: str):
    df = app.state.dataframe
    token_info = sp_oauth.get_cached_token()

    if not token_info:
        return []

    sp = spotipy.Spotify(auth=token_info["access_token"])
    songs = recommend_from_spotify_search(sp, df, mood)
    return songs


@app.post("/add_track")
def add_track(uri: str):

    try:

        sp = spotipy.Spotify(
            auth_manager=SpotifyOAuth(
                client_id=CLIENT_ID,
                client_secret=CLIENT_SECRET,
                redirect_uri=REDIRECT_URI,
                scope=SCOPE
            )
        )

        if not hasattr(app.state, "last_playlist_id"):
            raise HTTPException(status_code=400, detail="No playlist created yet")

        playlist_id = app.state.last_playlist_id

        payload = {
            "uris": [uri]
        }

        sp._post(f"playlists/{playlist_id}/items", payload=payload)

        return {"success": True}

    except Exception as e:
        raise HTTPException(status_code=403, detail=str(e))