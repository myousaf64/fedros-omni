def score(prediction: str, expected: str) -> float:
    return 1.0 if prediction.strip().lower() == expected.strip().lower() else 0.0
