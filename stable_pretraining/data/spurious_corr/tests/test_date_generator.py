import pytest
from spurious_corr.generators import SpuriousDateGenerator


@pytest.mark.unit
def test_no_duplicates_with_replacement_false():
    gen = SpuriousDateGenerator(
        year_range=(2020, 2020), seed=123, with_replacement=False
    )
    generated = set()
    num_samples = 365
    for _ in range(num_samples):
        date = gen()
        assert date not in generated
        generated.add(date)


@pytest.mark.unit
def test_same_seed_produces_same_sequence_no_replacement():
    g1 = SpuriousDateGenerator(year_range=(1900, 2100), seed=42, with_replacement=False)
    g2 = SpuriousDateGenerator(year_range=(1900, 2100), seed=42, with_replacement=False)

    dates1 = [g1() for _ in range(10000)]
    dates2 = [g2() for _ in range(10000)]

    assert dates1 == dates2


@pytest.mark.unit
def test_same_seed_produces_same_sequence_with_replacement():
    g1 = SpuriousDateGenerator(year_range=(1900, 2100), seed=42, with_replacement=True)
    g2 = SpuriousDateGenerator(year_range=(1900, 2100), seed=42, with_replacement=True)

    dates1 = [g1() for _ in range(10000)]
    dates2 = [g2() for _ in range(10000)]

    assert dates1 == dates2


@pytest.mark.unit
def test_different_seed_produces_different_sequence_no_replacement():
    g1 = SpuriousDateGenerator(year_range=(1900, 2100), seed=42, with_replacement=False)
    g2 = SpuriousDateGenerator(year_range=(1900, 2100), seed=42, with_replacement=False)

    dates1 = [g1() for _ in range(10000)]
    dates2 = [g2() for _ in range(10000)]

    assert dates1 == dates2


@pytest.mark.unit
def test_different_seed_produces_different_sequence_with_replacement():
    g1 = SpuriousDateGenerator(year_range=(1900, 2100), seed=42, with_replacement=True)
    g2 = SpuriousDateGenerator(year_range=(1900, 2100), seed=42, with_replacement=True)

    dates1 = [g1() for _ in range(10000)]
    dates2 = [g2() for _ in range(10000)]

    assert dates1 == dates2
