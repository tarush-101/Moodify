from .mood_engine import AUDIO_FEATURES


def attach_mood_names(df, centroids):

    mood_labels = {}

    # Rank clusters by energy
    energy_values = [(i, c[AUDIO_FEATURES.index("Energy")]) 
                     for i, c in enumerate(centroids)]

    energy_sorted = sorted(energy_values, key=lambda x: x[1])

    # Lowest energy → Melancholic
    mood_labels[energy_sorted[0][0]] = "Melancholic"

    # Highest energy → Party
    mood_labels[energy_sorted[-1][0]] = "Party"

    # Second highest energy → Energetic
    mood_labels[energy_sorted[-2][0]] = "Energetic"

    # Remaining clusters
    remaining = set(range(len(centroids))) - set(mood_labels.keys())
    remaining = list(remaining)

    # Among remaining, highest acousticness → Chill
    acoustic_scores = [(i, centroids[i][AUDIO_FEATURES.index("Acousticness")])
                       for i in remaining]

    acoustic_sorted = sorted(acoustic_scores, key=lambda x: x[1])

    mood_labels[acoustic_sorted[-1][0]] = "Chill"

    # Last remaining → Flow State
    final_remaining = set(range(len(centroids))) - set(mood_labels.keys())
    mood_labels[list(final_remaining)[0]] = "Flow State"

    df["mood"] = df["mood_cluster"].map(mood_labels)

    return df