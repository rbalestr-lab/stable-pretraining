import pytest
from pathlib import Path
from stable_pretraining.data.utils import write_random_dates


@pytest.mark.unit
def test_no_duplicates_with_replacement_false(tmp_path: Path):
    """Ensure write_random_dates produces unique dates when with_replacement=False."""
    output_file = tmp_path / "dates.txt"
    write_random_dates(
        output_file, n=365, year_range=(2020, 2020), seed=123, with_replacement=False
    )

    dates = [line.strip() for line in open(output_file)]
    assert len(dates) == len(set(dates)), "Duplicates found when with_replacement=False"


@pytest.mark.unit
def test_with_replacement_allows_duplicates(tmp_path: Path):
    """Ensure write_random_dates allows duplicates when with_replacement=True."""
    output_file = tmp_path / "dates.txt"
    write_random_dates(
        output_file, n=1000, year_range=(2020, 2020), seed=42, with_replacement=True
    )

    dates = [line.strip() for line in open(output_file)]
    # With replacement, duplicates should appear in large samples
    assert len(set(dates)) < len(dates), (
        "No duplicates found when with_replacement=True"
    )


@pytest.mark.unit
def test_same_seed_produces_same_output(tmp_path: Path):
    """Ensure deterministic output for same seed."""
    f1 = tmp_path / "dates1.txt"
    f2 = tmp_path / "dates2.txt"

    write_random_dates(
        f1, n=100, year_range=(1900, 1905), seed=42, with_replacement=True
    )
    write_random_dates(
        f2, n=100, year_range=(1900, 1905), seed=42, with_replacement=True
    )

    d1 = open(f1).read().splitlines()
    d2 = open(f2).read().splitlines()

    assert d1 == d2, "Outputs differ for same seed"


@pytest.mark.unit
def test_different_seed_produces_different_output(tmp_path: Path):
    """Ensure non-deterministic output for different seeds."""
    f1 = tmp_path / "dates1.txt"
    f2 = tmp_path / "dates2.txt"

    write_random_dates(
        f1, n=100, year_range=(1900, 1905), seed=42, with_replacement=True
    )
    write_random_dates(
        f2, n=100, year_range=(1900, 1905), seed=99, with_replacement=True
    )

    d1 = open(f1).read().splitlines()
    d2 = open(f2).read().splitlines()

    assert d1 != d2, "Outputs identical for different seeds"


@pytest.mark.unit
def test_number_of_lines_written(tmp_path: Path):
    """Ensure exactly n lines are written."""
    output_file = tmp_path / "dates.txt"
    n = 50
    write_random_dates(
        output_file, n=n, year_range=(2020, 2020), seed=0, with_replacement=False
    )
    lines = open(output_file).read().splitlines()
    assert len(lines) == n, f"Expected {n} lines, got {len(lines)}"
