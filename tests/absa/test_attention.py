from src.absa.interpretability.attention import EVIDENCE_CAVEAT, align_attention


def test_attention_alignment_uses_visible_tokens_only():
    evidence = align_attention("Great food!", [0.2, 0.7, 0.1, 0.0])
    assert evidence == [{"token": "great", "weight": 0.2}, {"token": "food", "weight": 0.7}, {"token": "!", "weight": 0.1}]
    assert "not causal" in EVIDENCE_CAVEAT


def test_attention_alignment_stops_at_shorter_sequence():
    assert align_attention("Great food!", [0.2]) == [{"token": "great", "weight": 0.2}]
