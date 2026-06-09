import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def score_caption(text: str) -> dict:
    """
    Returns full VADER scores + a label.
    """
    scores = _analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return {
        "compound": round(compound, 4),
        "pos": round(scores["pos"], 4),
        "neg": round(scores["neg"], 4),
        "neu": round(scores["neu"], 4),
        "label": label,
    }


def batch_score(captions: list[str]) -> list[dict]:
    """Score a list of captions. Returns list of score dicts."""
    return [score_caption(c) for c in captions]


def validate_alignment_labels(df: pd.DataFrame) -> dict:
    """Sanity check for alignment_label vs. VADER sentiment."""

    results = []
    expected_sentiments = {
        "Energetic": "positive",
        "Calm": "neutral",
        "Professional": "neutral",
        "Playful": "positive",
    }

    for _, row in df.iterrows():
        s = score_caption(str(row["caption"]))
        mood = str(row.get("mood", "")).strip()
        expected = expected_sentiments.get(mood, "neutral")

        results.append(
            {
                "caption": row["caption"],
                "mood": mood,
                "alignment_label": row.get("alignment_label", ""),
                "vader_label": s["label"],
                "compound": s["compound"],
                "expected_sentiment": expected,
                "sentiment_matches_mood": s["label"] == expected,
            }
        )

    out_df = pd.DataFrame(results)
    agreement = out_df["sentiment_matches_mood"].mean()

    print(f"VADER–Mood agreement: {agreement:.1%}")
    print("  (expected > 60% for a well-labeled dataset)")
    print(out_df["vader_label"].value_counts().to_string())

    return {
        "agreement_rate": agreement,
        "details": out_df,
    }
