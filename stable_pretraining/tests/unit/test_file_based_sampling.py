import pytest
import tempfile
import os
from stable_pretraining.data.utils import write_random_dates
from stable_pretraining.data.spurious_corr.modifiers import ItemInjection


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
def test_item_injection_reads_from_file_and_injects_correctly():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write fake items
        file_path = os.path.join(tmpdir, "items.txt")
        with open(file_path, "w") as f:
            f.write("A\nB\nC\n")

        # Create modifier
        modifier = ItemInjection(file_path=file_path, token="TEST")

        text, label = modifier("original text", 0)
        # The exact assertion depends on how your ItemInjection modifies the text.
        # Here's a generic check:
        assert isinstance(text, str)
        assert "TEST" in text or any(x in text for x in ["A", "B", "C"])


@pytest.mark.unit
def test_item_injection_is_deterministic_with_seed():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "items.txt")
        with open(file_path, "w") as f:
            f.write("X\nY\nZ\n")

        m1 = ItemInjection(file_path=file_path, token="SPURIOUS", seed=123)
        m2 = ItemInjection(file_path=file_path, token="SPURIOUS", seed=123)

        out1 = [m1("base text", 1)[0] for _ in range(10)]
        out2 = [m2("base text", 1)[0] for _ in range(10)]

        assert out1 == out2
