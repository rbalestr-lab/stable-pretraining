import pytest
import tempfile
import os
from stable_pretraining.data.spurious_corr.generators import SpuriousFileItemGenerator


# Utility to create a temp file with test content
@pytest.fixture
def temp_file():
    content = "\n".join(f"item_{i}" for i in range(100))
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        f.write(content)
        f.flush()
        yield f.name
    os.remove(f.name)


@pytest.mark.unit
def test_no_duplicates_with_replacement_false(temp_file):
    gen = SpuriousFileItemGenerator(temp_file, seed=123, with_replacement=False)
    generated = set()
    for _ in range(100):
        item = gen()
        assert item not in generated
        generated.add(item)

    with pytest.raises(RuntimeError):
        gen()  # Should raise after exhausting all items


@pytest.mark.unit
def test_same_seed_produces_same_sequence_no_replacement(temp_file):
    g1 = SpuriousFileItemGenerator(temp_file, seed=42, with_replacement=False)
    g2 = SpuriousFileItemGenerator(temp_file, seed=42, with_replacement=False)

    items1 = [g1() for _ in range(100)]
    items2 = [g2() for _ in range(100)]

    assert items1 == items2


@pytest.mark.unit
def test_same_seed_produces_same_sequence_with_replacement(temp_file):
    g1 = SpuriousFileItemGenerator(temp_file, seed=42, with_replacement=True)
    g2 = SpuriousFileItemGenerator(temp_file, seed=42, with_replacement=True)

    items1 = [g1() for _ in range(100)]
    items2 = [g2() for _ in range(100)]

    assert items1 == items2


@pytest.mark.unit
def test_different_seed_produces_different_sequence_with_replacement(temp_file):
    g1 = SpuriousFileItemGenerator(temp_file, seed=1, with_replacement=True)
    g2 = SpuriousFileItemGenerator(temp_file, seed=2, with_replacement=True)

    items1 = [g1() for _ in range(100)]
    items2 = [g2() for _ in range(100)]

    assert items1 != items2  # Very unlikely to match by chance


@pytest.mark.unit
def test_raises_on_empty_file():
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        pass  # empty file
    with pytest.raises(ValueError):
        SpuriousFileItemGenerator(f.name)
    os.remove(f.name)


@pytest.mark.unit
def test_generator_raises_after_all_items_used(temp_file):
    gen = SpuriousFileItemGenerator(temp_file, seed=42, with_replacement=False)
    for _ in range(100):  # exhaust all items
        _ = gen()
    with pytest.raises(RuntimeError, match="All unique items have been generated."):
        gen()
