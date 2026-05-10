import numpy as np
import pandas as pd

MOOD_COORDINATES = {
    "Melancholic": (0.2, 0.2),
    "Chill": (0.3, 0.5),
    "Flow State": (0.4, 0.4),
    "Energetic": (0.8, 0.7),
    "Party": (0.9, 0.9),
}


def generate_mood_journey(df: pd.DataFrame, start_mood: str, end_mood: str, steps: int = 12):
    """
    Generate a playlist that transitions smoothly from one mood to another.
    """

    if start_mood not in MOOD_COORDINATES or end_mood not in MOOD_COORDINATES:
        raise ValueError("Invalid mood")

    start = np.array(MOOD_COORDINATES[start_mood])
    end = np.array(MOOD_COORDINATES[end_mood])

    # Create emotional path
    transition_points = np.linspace(start, end, steps)

    playlist = []
    working_df = df.copy()

    for point in transition_points:

        working_df["distance"] = np.sqrt(
            (working_df["Energy"] - point[0]) ** 2 +
            (working_df["Valence"] - point[1]) ** 2
        )

        track = working_df.sort_values("distance").iloc[0]

        playlist.append(track)

        working_df = working_df.drop(track.name)

    return pd.DataFrame(playlist)