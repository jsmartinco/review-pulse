from src.absa.interpretability.attention import EVIDENCE_CAVEAT, align_attention


def test_attention_alignment_uses_visible_tokens_only():
    evidence = align_attention("Great food!", [0.2, 0.7, 0.1, 0.0])
    assert evidence == [
        {"token": "Great", "start": 0, "end": 5, "score": 0.2, "weight": 0.2},
        {"token": "food", "start": 6, "end": 10, "score": 0.7, "weight": 0.7},
        {"token": "!", "start": 10, "end": 11, "score": 0.1, "weight": 0.1},
    ]
    assert "causal explanation" in EVIDENCE_CAVEAT


def test_attention_alignment_stops_at_shorter_sequence():
    assert align_attention("Great food!", [0.2]) == [
        {"token": "Great", "start": 0, "end": 5, "score": 0.2, "weight": 0.2}
    ]


def test_attention_alignment_preserves_case_punctuation_and_offsets():
    review = "Desserts—excellent; service... slow?"
    evidence = align_attention(review, [0.1] * 10)
    assert "".join(review[item["start"] : item["end"]] for item in evidence) == (
        "Desserts—excellent;service...slow?"
    )
    assert [item["token"] for item in evidence] == [
        "Desserts",
        "—",
        "excellent",
        ";",
        "service",
        ".",
        ".",
        ".",
        "slow",
        "?",
    ]
