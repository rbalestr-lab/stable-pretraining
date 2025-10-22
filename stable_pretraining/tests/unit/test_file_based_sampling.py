import pytest
import tempfile
import os
from stable_pretraining.data.utils import load_items_from_file, write_random_dates


@pytest.fixture
def temp_file():
    """Create a temporary file with predictable content."""
    content = "\n".join(f"item_{i}" for i in range(100))
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        f.write(content)
        f.flush()
        yield f.name
    os.remove(f.name)


@pytest.mark.unit
def test_load_items_from_file_no_replacement(temp_file):
    items = load_items_from_file(temp_file, seed=123, with_replacement=False)
    assert len(items) == 100
    assert len(set(items)) == 100  # unique items only


@pytest.mark.unit
def test_load_items_from_file_with_replacement(temp_file):
    items = load_items_from_file(temp_file, seed=123, with_replacement=True)
    assert len(items) == 100
    # replacement allows duplicates
    assert len(set(items)) <= 100


@pytest.mark.unit
def test_load_items_reproducibility_with_same_seed(temp_file):
    items1 = load_items_from_file(temp_file, seed=42, with_replacement=True)
    items2 = load_items_from_file(temp_file, seed=42, with_replacement=True)
    assert items1 == items2


@pytest.mark.unit
def test_load_items_different_seed_produces_different_sequence(temp_file):
    items1 = load_items_from_file(temp_file, seed=1, with_replacement=True)
    items2 = load_items_from_file(temp_file, seed=2, with_replacement=True)
    assert items1 != items2  # extremely unlikely to match


@pytest.mark.unit
def test_load_items_from_empty_file_raises():
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        path = f.name
    with pytest.raises(ValueError):
        load_items_from_file(path)
    os.remove(path)


@pytest.mark.unit
def test_write_random_dates_creates_valid_file():
    """Ensure write_random_dates writes unique or repeatable dates."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "dates.txt")
        write_random_dates(
            path,
            num_samples=50,
            year_range=(2020, 2020),
            seed=42,
            with_replacement=False,
        )
        with open(path, "r") as f:
            dates = [line.strip() for line in f if line.strip()]
        assert len(dates) == len(set(dates))
        assert all(date.startswith("2020-") for date in dates)
