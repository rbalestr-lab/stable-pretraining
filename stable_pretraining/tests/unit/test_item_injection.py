import pytest
from stable_pretraining.data.transforms import SpuriousTextInjection, AddSampleIdx


@pytest.mark.unit
def test_spurious_text_injection_deterministic_same_idx(tmp_path):
    # Create dummy spurious tokens file
    src_path = tmp_path / "spurious.txt"
    src_path.write_text("RED\nGREEN\nBLUE\n")

    text = "deterministic spurious injection test"
    transform = SpuriousTextInjection(
        text_key="text",
        source=str(src_path),
        location="random",
        token_proportion=0.5,
        seed=123,
    )

    sample1 = {"text": text, "label": "A", "idx": 5}
    sample2 = {"text": text, "label": "A", "idx": 5}

    out1 = transform(sample1)
    out2 = transform(sample2)

    assert out1["text"] == out2["text"], "Same idx should produce identical injection"
    assert out1["label"] == out2["label"]


@pytest.mark.unit
def test_spurious_text_injection_deterministic_different_idx(tmp_path):
    src_path = tmp_path / "spurious.txt"
    src_path.write_text("HELLO\nWORLD\n")

    text = "check for idx-dependent difference"
    transform = SpuriousTextInjection(
        text_key="text",
        source=str(src_path),
        location="random",
        token_proportion=0.5,
        seed=321,
    )

    sample1 = {"text": text, "label": "B", "idx": 1}
    sample2 = {"text": text, "label": "B", "idx": 2}

    out1 = transform(sample1)
    out2 = transform(sample2)

    assert out1["text"] != out2["text"], "Different idx should yield different outputs"


@pytest.mark.unit
def test_spurious_text_injection_reproducibility_across_runs(tmp_path):
    src_path = tmp_path / "tokens.txt"
    src_path.write_text("A\nB\nC\n")

    text = "reproducibility test for spurious injection"
    sample = {"text": text, "label": "C", "idx": 42}

    transform1 = SpuriousTextInjection(
        text_key="text",
        source=str(src_path),
        location="end",
        token_proportion=0.25,
        seed=999,
    )
    transform2 = SpuriousTextInjection(
        text_key="text",
        source=str(src_path),
        location="end",
        token_proportion=0.25,
        seed=999,
    )

    out1 = transform1(sample)
    out2 = transform2(sample)

    assert out1["text"] == out2["text"], (
        "Same seed and idx should yield identical injection"
    )


@pytest.mark.unit
def test_spurious_text_injection_with_addsampleidx(tmp_path):
    src_path = tmp_path / "spurious.txt"
    src_path.write_text("NOISE\nTAG\n")

    add_idx = AddSampleIdx()
    transform = SpuriousTextInjection(
        text_key="text",
        source=str(src_path),
        location="beginning",
        token_proportion=0.3,
        seed=777,
    )

    sample = {"text": "verify deterministic pipeline", "label": "Y"}
    sample = add_idx(sample)
    out = transform(sample)

    assert "idx" in sample, "AddSampleIdx should add an 'idx' field"
    assert any(t in out["text"] for t in ["NOISE", "TAG"]), (
        "Should inject spurious token"
    )
