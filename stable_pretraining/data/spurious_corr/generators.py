"""generators.py.

This module provides generator functions for creating spurious text injections.
These functions can be used directly or integrated with the ItemInjection modifier.
"""

import random
import calendar


class SpuriousDateGenerator:
    """Generates random date strings in YYYY-MM-DD format.

    Can be configured to allow or disallow duplicates.
    """

    def __init__(self, year_range=(1100, 2600), seed=None, with_replacement=False):
        """Initialize the generator.

        Args:
            year_range (tuple): A (start_year, end_year) tuple.
            seed (int, optional): Seed for reproducibility.
            with_replacement (bool): Whether to allow duplicates.
        """
        self.rng = random.Random(seed)
        self.with_replacement = with_replacement
        self.generated = set()
        self.possible_dates = self._generate_all_valid_dates(year_range)
        self.total_possible = len(self.possible_dates)

    def _generate_all_valid_dates(self, year_range):
        """Precompute all valid dates in the range.

        Args:
            year_range (tuple): A (start_year, end_year) tuple.

        Returns:
            list[str]: List of all valid dates in the range.
        """
        start_year, end_year = year_range
        dates = []
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                _, max_day = calendar.monthrange(year, month)
                for day in range(1, max_day + 1):
                    date_str = f"{year}-{month:02d}-{day:02d}"
                    dates.append(date_str)
        return dates

    def __call__(self):
        """Generate a random date string.

        Returns:
            str: A random date string.

        Raises:
            RuntimeError: If all unique dates have been generated (when with_replacement is False).
        """
        if self.with_replacement:
            return self.rng.choice(self.possible_dates)

        if len(self.generated) >= self.total_possible:
            raise RuntimeError("All unique dates have been generated.")

        while True:
            date = self.rng.choice(self.possible_dates)
            if date not in self.generated:
                self.generated.add(date)
                return date


class SpuriousFileItemGenerator:
    """Generates items from a file, optionally without replacement.

    Each non-empty line in the file is considered a distinct item.
    """

    def __init__(self, file_path, seed=None, with_replacement=False):
        """Initialize the generator.

        Args:
            file_path (str): Path to the file with one item per line.
            seed (int, optional): Seed for reproducibility.
            with_replacement (bool): Whether to allow duplicates.
        """
        self.rng = random.Random(seed)
        self.with_replacement = with_replacement
        self.generated = set()

        with open(file_path, "r", encoding="utf-8") as f:
            self.items = [line.strip() for line in f if line.strip()]

        if not self.items:
            raise ValueError("File is empty or contains only blank lines.")

        self.total_possible = len(self.items)

    def __call__(self):
        """Generate a random item from the file.

        Returns:
            str: A random item.

        Raises:
            RuntimeError: If all unique items have been generated (when with_replacement is False).
        """
        if self.with_replacement:
            return self.rng.choice(self.items)

        if len(self.generated) >= self.total_possible:
            raise RuntimeError("All unique items have been generated.")

        while True:
            item = self.rng.choice(self.items)
            if item not in self.generated:
                self.generated.add(item)
                return item
