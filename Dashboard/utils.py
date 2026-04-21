EMOTIONS = ['Happy', 'Sad', 'Angry', 'Fearful', 'Neutral']


def combine_at(results):

    combined = {emotion: 0.0 for emotion in EMOTIONS}

    total_models = len(results)
    
    weights = [0.4,0.4,0.2]

    for i, result in enumerate(results):
        print(f"weights: {result}", flush=True)
        if not result:
            continue
        probs = result["all_probs"]
        weight = weights[i%3]
        for emotion, value in probs.items():
            combined[emotion] += value * weight / total_models
    
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