import pytest
import tempfile
import os
from stable_pretraining.data.utils import write_random_dates
from stable_pretraining.data.transforms import SpuriousTextInjection


@pytest.mark.unit
def test_write_random_dates_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = os.path.join(tmpdir, "dates.txt")
        write_random_dates(out_file, n=10, seed=42, with_replacement=False)

        assert os.path.exists(out_file)
        with open(out_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        assert len(lines) == 10
        assert all("-" in d for d in lines)  # looks like YYYY-MM-DD


@pytest.mark.unit
def test_spurious_text_injection_reads_from_file_and_injects_correctly():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create fake spurious tokens file
        file_path = os.path.join(tmpdir, "tokens.txt")
        with open(file_path, "w") as f:
            f.write("A\nB\nC\n")

        # Create transform
        transform = SpuriousTextInjection(
            text_key="text",
            file_path=file_path,
            location="random",
            token_proportion=0.5,
            seed=42,
        )

        sample = {"text": "original text", "label": 0}
        output = transform(sample)

        assert isinstance(output["text"], str)
        assert output["text"] != "original text"
        assert any(tok in output["text"] for tok in ["A", "B", "C"])


@pytest.mark.unit
def test_spurious_text_injection_is_deterministic_with_seed():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "tokens.txt")
        with open(file_path, "w") as f:
            f.write("X\nY\nZ\n")

        # Create two transforms with the same seed
        t1 = SpuriousTextInjection(
            text_key="text",
            file_path=file_path,
            location="end",
            token_proportion=0.5,
            seed=123,
        )
        t2 = SpuriousTextInjection(
            text_key="text",
            file_path=file_path,
            location="end",
            token_proportion=0.5,
            seed=123,
        )

        sample = {"text": "base text", "label": 1}
        outputs1 = [t1(sample)["text"] for _ in range(5)]
        outputs2 = [t2(sample)["text"] for _ in range(5)]

        assert outputs1 == outputs2, "Should produce identical results with same seed"
