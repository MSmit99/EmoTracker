EMOTIONS = ['Happy', 'Sad', 'Angry', 'Fearful', 'Neutral']


def combine_at(results):

    combined = {emotion: 0.0 for emotion in EMOTIONS}

    total_models = len(results)

    for result in results:
        if not result:
            continue
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
        "all_probs": combined
    }


def average_timeline(results):
    timelines = []
    
    for result in results:
        if not result:
            continue
        else:
            timelines.append(result['timeline'])
            
    shortest = min(len(timeline) for timeline in timelines)

    ensemble = []

    for second in range(shortest):

        second_predictions = [
            timeline[second]
            for timeline in timelines
        ]

        merged = combine_at(second_predictions)
        merged["second"] = second

        ensemble.append(merged)

    return ensemble