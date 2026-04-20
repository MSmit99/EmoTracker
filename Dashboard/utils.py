EMOTIONS = ['Happy', 'Sad', 'Angry', 'Fearful', 'Neutral']

EMOTION_COLORS = {
    'Happy': '#F59E0B',
    'Sad': '#3B82F6',
    'Angry': '#EF4444',
    'Fearful': '#8B5CF6',
    'Neutral': '#6B7280'
}


def combine_at(results):
    """
    Combine predictions from FER / SER / TER models
    using probability averaging.

    Input:
        [
            {
                "emotion": "Happy",
                "all_probs": {
                    "Happy": 70.0,
                    ...
                }
            },
            ...
        ]

    Returns:
        {
            "emotion": "Happy",
            "accuracy": 62.4,
            "color": "#F59E0B",
            "all_probs": {...}
        }
    """

    combined = {emotion: 0.0 for emotion in EMOTIONS}

    total_models = len(results)

    for result in results:
        probs = result["all_probs"]

        for emotion, value in probs.items():
            combined[emotion] += value / total_models

    combined = {
        emotion: round(score, 1)
        for emotion, score in combined.items()
    }

    best_emotion = max(combined, key=combined.get)

    return {
        "emotion": best_emotion,
        "accuracy": combined[best_emotion],
        "color": EMOTION_COLORS[best_emotion],
        "all_probs": combined
    }


def average_timeline(model_timelines):
    """
    Build full ensemble timeline.

    Input:
        [
            fer_timeline,
            ser_timeline,
            ter_timeline
        ]

    Returns:
        [
            second0 prediction,
            second1 prediction,
            ...
        ]
    """

    shortest = min(len(timeline) for timeline in model_timelines)

    ensemble = []

    for second in range(shortest):

        second_predictions = [
            timeline[second]
            for timeline in model_timelines
        ]

        merged = combine_at(second_predictions)
        merged["second"] = second

        ensemble.append(merged)

    return ensemble