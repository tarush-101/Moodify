import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

AUDIO_FEATURES = [
    "Danceability",
    "Energy",
    "Loudness",
    "Speechiness",
    "Acousticness",
    "Instrumentalness",
    "Liveness",
    "Valence",
    "Tempo",
]

FEATURE_WEIGHTS = {
    "Danceability": 1.3,
    "Energy": 1.6,
    "Loudness": 1.0,
    "Speechiness": 0.7,
    "Acousticness": 1.2,
    "Instrumentalness": 1.0,
    "Liveness": 0.9,
    "Valence": 1.5,
    "Tempo": 0.8,
}


def run_mood_clustering(df: pd.DataFrame, n_clusters: int = 5):

    feature_df = df.dropna(subset=AUDIO_FEATURES).copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df[AUDIO_FEATURES])

    # Apply emotional weighting
    for i, col in enumerate(AUDIO_FEATURES):
        X_scaled[:, i] *= FEATURE_WEIGHTS[col]

    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    clusters = gmm.fit_predict(X_scaled)

    feature_df["mood_cluster"] = clusters

    # Merge cluster back
    df = df.merge(
        feature_df[["Track URI", "mood_cluster"]],
        on="Track URI",
        how="left",
    )

    return df, gmm.means_