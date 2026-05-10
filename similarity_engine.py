import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

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

def build_similarity_matrix(df):

    scaler = StandardScaler()

    X = scaler.fit_transform(df[AUDIO_FEATURES])

    similarity = cosine_similarity(X)

    return similarity


def get_similar_songs(df, similarity_matrix, track_name, n=5):

    idx = df[df["Track Name"] == track_name].index[0]

    scores = list(enumerate(similarity_matrix[idx]))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    scores = scores[1:n+1]

    indices = [i[0] for i in scores]

    return df.iloc[indices][
        ["Track Name","Artist Name(s)","mood"]
    ]