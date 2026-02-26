"""
Majority Voting Module for Speech Emotion Recognition
Weighted voting based on model accuracy scores.
"""

# Model accuracy weights — higher = more influence on final vote.
# Based on test accuracy of each model on the TurEV-DB dataset.
MODEL_WEIGHTS = {
    'catboost':           1.00,  # Highest accuracy
    'xgboost':            0.90,
    'lightgbm':           0.88,
    'gradient_boosting':  0.85,
    'rf':                 0.82,
    'dnn':                0.80,
    'cnn1d':              0.78,
    'mlp':                0.75,
    'svm':                0.72,
    'knn':                0.65,
}

# Robust variants share the same weight as their base model
for key in list(MODEL_WEIGHTS.keys()):
    MODEL_WEIGHTS[f"{key}_robust"] = MODEL_WEIGHTS[key]

EMOTIONS = ['happy', 'sad', 'angry', 'calm']


def calculate_majority_vote(model_predictions: dict) -> dict:
    """
    Weighted majority voting across multiple model predictions.

    Args:
        model_predictions: {
            'catboost': {'happy': 30.2, 'sad': 10.1, 'angry': 50.0, 'calm': 9.7},
            'xgboost':  {'happy': 20.0, 'sad': 40.0, 'angry': 25.0, 'calm': 15.0},
            ...
        }
        Values are percentages (0-100).

    Returns: {
        'final_emotion': 'angry',
        'confidence': 42.5,
        'all_scores': {'happy': ..., 'sad': ..., 'angry': ..., 'calm': ...},
        'model_details': [
            {'model': 'CatBoost', 'weight': 1.0, 'prediction': 'angry', 'confidence': 50.0, 'scores': {...}},
            ...
        ]
    }
    """
    # Accumulate weighted scores
    weighted_scores = {e: 0.0 for e in EMOTIONS}
    total_weight = 0.0
    model_details = []

    for model_key, scores in model_predictions.items():
        weight = MODEL_WEIGHTS.get(model_key, 0.5)
        total_weight += weight

        # Find model's top prediction
        top_emotion = max(scores, key=scores.get)
        top_confidence = scores[top_emotion]

        # Accumulate weighted scores for each emotion
        for emotion in EMOTIONS:
            val = scores.get(emotion, 0.0)
            weighted_scores[emotion] += val * weight

        # Prettify model name for frontend
        display_name = model_key.replace('_robust', ' (Robust)').replace('_', ' ').title()

        model_details.append({
            'model': display_name,
            'key': model_key,
            'weight': round(weight, 2),
            'prediction': top_emotion,
            'confidence': round(top_confidence, 2),
            'scores': {k: round(v, 2) for k, v in scores.items()}
        })

    # Normalize to 0-100%
    if total_weight > 0:
        for e in EMOTIONS:
            weighted_scores[e] = weighted_scores[e] / total_weight

    # Re-normalize so they sum to 100
    total = sum(weighted_scores.values())
    if total > 0:
        for e in EMOTIONS:
            weighted_scores[e] = (weighted_scores[e] / total) * 100

    final_emotion = max(weighted_scores, key=weighted_scores.get)
    confidence = weighted_scores[final_emotion]

    return {
        'final_emotion': final_emotion,
        'confidence': round(confidence, 2),
        'all_scores': {k: round(v, 2) for k, v in weighted_scores.items()},
        'model_details': sorted(model_details, key=lambda x: x['weight'], reverse=True)
    }
