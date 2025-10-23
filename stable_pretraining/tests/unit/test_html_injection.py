import pytest
from stable_pretraining.data.transforms import HTMLInjection


@pytest.mark.unit
def test_html_injection_deterministic_same_idx(tmp_path):
    tag_path = tmp_path / "tags.txt"
    tag_path.write_text("<x> </x>\n")

    # Create deterministic injection transform
    modifier = HTMLInjection(
        file_path=str(tag_path), location="random", token_proportion=0.5, seed=123
    )

    # Two samples with the same idx should yield identical results
    sample1 = {"text": "consistent sample text", "label": "lbl", "idx": 10}
    sample2 = {"text": "consistent sample text", "label": "lbl", "idx": 10}

    out1 = modifier(sample1)
    out2 = modifier(sample2)

    assert out1["text"] == out2["text"], "Same idx should produce identical injection"
    assert out1["label"] == out2["label"]


@pytest.mark.unit
def test_html_injection_deterministic_different_idx(tmp_path):
    tag_path = tmp_path / "tags.txt"
    tag_path.write_text("<x> </x>\n")

    modifier = HTMLInjection(
        file_path=str(tag_path), location="random", token_proportion=0.5, seed=123
    )

    # Two samples with different idx should yield different results
    sample1 = {"text": "different sample text", "label": "lbl", "idx": 1}
    sample2 = {"text": "different sample text", "label": "lbl", "idx": 2}

    out1 = modifier(sample1)
    out2 = modifier(sample2)

    assert out1["text"] != out2["text"], (
        "Different idx should produce different injections"
    )


@pytest.mark.unit
def test_html_injection_deterministic_reproducibility_across_runs(tmp_path):
    tag_path = tmp_path / "tags.txt"
    tag_path.write_text("<p> </p>\n")

    sample = {"text": "check reproducibility", "label": "lbl", "idx": 42}

    modifier1 = HTMLInjection(
        file_path=str(tag_path), location="random", token_proportion=0.5, seed=777
    )
    modifier2 = HTMLInjection(
        file_path=str(tag_path), location="random", token_proportion=0.5, seed=777
    )

    out1 = modifier1(sample)
    out2 = modifier2(sample)

    assert out1["text"] == out2["text"], (
        "Same base seed and idx must yield identical output"
    )
