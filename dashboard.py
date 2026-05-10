import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE 

# Import your backend pipeline
from src.ingestion.cleaner import clean_dataframe
from src.mood.mood_engine import run_mood_clustering
from src.mood.mood_mapping import attach_mood_names
from src.mood.llm_mood_corrector import correct_dataframe_moods

@st.cache_data
def run_full_pipeline(df):
    df = clean_dataframe(df)
    df, centroids = run_mood_clustering(df)
    df = attach_mood_names(df, centroids)
    df["ml_mood"] = df["mood"].copy() 
    df = correct_dataframe_moods(df)
    
    # Add a status column to easily track what the LLM did
    df["Status"] = df.apply(lambda x: "Changed by LLM" if x["ml_mood"] != x["mood"] else "Unchanged", axis=1)
    
    return df

st.set_page_config(page_title="Moodify AI Dashboard", layout="wide")

st.title("🎵 Moodify AI Pipeline Dashboard")

st.write("""
This dashboard visualizes how Moodify processes music:

CSV → Cleaning → ML Clustering → Mood Mapping → LLM Correction → Playlist Generation
""")

uploaded_file = st.file_uploader("Upload Exportify CSV", type="csv")

if uploaded_file is None:
    st.warning("Upload your Exportify CSV file to start.")
    st.stop()

# ----------------------------
# Load CSV
# ----------------------------

df = pd.read_csv(uploaded_file)
st.subheader("Raw CSV Data")
st.dataframe(df.head())

# ----------------------------
# Run Pipeline
# ----------------------------

st.subheader("Running AI Pipeline")

with st.spinner("Processing data (this will only run once!)..."):
    df = run_full_pipeline(df)

st.success("Pipeline completed")

# ----------------------------
# Processed Dataset
# ----------------------------

st.subheader("Processed Dataset Snapshot")

# Rename columns just for the display table so it's easy to read
display_df = df[["Track Name", "Artist Name(s)", "ml_mood", "mood", "confidence", "Status"]].rename(columns={
    "ml_mood": "Initial ML Mood",
    "mood": "Final Verified Mood",
    "confidence": "AI Confidence"
})
st.dataframe(display_df.head(10))

# ----------------------------
# Metrics & Distributions
# ----------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("Mood Distribution")
    mood_counts = df["mood"].value_counts()
    fig, ax = plt.subplots()
    mood_counts.plot(kind="bar", ax=ax, color='skyblue')
    ax.set_title("Final Songs per Mood")
    ax.set_xlabel("Mood")
    ax.set_ylabel("Number of Songs")
    st.pyplot(fig)

with col2:
    st.subheader("AI Confidence Breakdown")
    if "confidence" in df.columns:
        conf_counts = df["confidence"].value_counts()
        fig_conf = px.pie(
            names=conf_counts.index,
            values=conf_counts.values,
            hole=0.4,
            template="plotly_dark"
        )
        st.plotly_chart(fig_conf, use_container_width=True)

# ----------------------------
# ML Cluster Visualization (t-SNE)
# ----------------------------

AUDIO_FEATURES = [
    "Danceability", "Energy", "Loudness", "Speechiness", 
    "Acousticness", "Instrumentalness", "Liveness", "Valence", "Tempo"
]

st.subheader("Interactive Music Universe (t-SNE Map)")

clean_df = df.dropna(subset=AUDIO_FEATURES + ["Track Name", "Artist Name(s)", "mood"])
features = clean_df[AUDIO_FEATURES]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

safe_perplexity = min(30, len(features) - 1) 
tsne = TSNE(n_components=2, perplexity=safe_perplexity, random_state=42)
reduced = tsne.fit_transform(scaled_features)

plot_df = pd.DataFrame({
    "X": reduced[:, 0],
    "Y": reduced[:, 1],
    "Song": clean_df["Track Name"],
    "Artist": clean_df["Artist Name(s)"],
    "Final Mood": clean_df["mood"]
})

fig = px.scatter(
    plot_df, x="X", y="Y", color="Final Mood",
    hover_name="Song", hover_data=["Artist", "Final Mood"],
    title="How the AI grouped similar songs (Hover to explore)",
    template="plotly_dark" 
)

fig.update_xaxes(showticklabels=False, title="")
fig.update_yaxes(showticklabels=False, title="")
st.plotly_chart(fig, use_container_width=True) 

# ----------------------------
# LLM Corrections Detail
# ----------------------------

st.subheader("LLM Mood Corrections")

corrections = df[df["Status"] == "Changed by LLM"]

if len(corrections) == 0:
    st.info("The LLM agreed with all ML predictions. No changes made.")
else:
    st.warning(f"The LLM stepped in and corrected {len(corrections)} tracks.")
    
    display_corrections = corrections[[
        "Track Name", "Artist Name(s)", "ml_mood", "mood", "confidence"
    ]].rename(columns={
        "ml_mood": "Initial ML Mood",
        "mood": "Final Verified Mood",
        "confidence": "AI Confidence"
    })
    
    st.dataframe(display_corrections)

# ----------------------------
# Mood Journey Simulation
# ----------------------------

st.subheader("Mood Journey Simulation")

moods = df["mood"].dropna().unique()

with st.form("mood_journey_form"):
    start_mood = st.selectbox("Start Mood", moods)
    end_mood = st.selectbox("End Mood", moods)
    submitted = st.form_submit_button("Generate Mood Journey")

if submitted:
    start_tracks = df[df["mood"] == start_mood].sort_values("Energy")
    end_tracks = df[df["mood"] == end_mood].sort_values("Energy")

    playlist = pd.concat([
        start_tracks.head(5),
        end_tracks.tail(5)
    ])

    st.write("Generated Playlist")
    
    display_playlist = playlist[[
        "Track Name", "Artist Name(s)", "Energy", "mood"
    ]].rename(columns={"mood": "Final Mood"})
    
    st.dataframe(display_playlist)

    fig, ax = plt.subplots()
    ax.plot(range(len(playlist)), playlist["Energy"], marker='o')
    ax.set_title("Energy Transition")
    ax.set_xlabel("Song Order")
    ax.set_ylabel("Energy")
    st.pyplot(fig)