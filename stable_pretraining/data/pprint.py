"""pprint.py.

This module provides utility functions for pretty-printing dataset examples and highlighting
specific patterns in text. These functions are useful for debugging and visualizing the modifications
applied to the dataset.
"""

import re
from termcolor import colored


class TextHighlight:
    """Class that grants the functionality to highlight different text and pretty print it."""

    def __init__(self, mode="dates", file_path=None):
        """A unified class for highlighting text.

        Args:
            mode (str): One of ["dates", "file", "html"] — determines the highlight type.
            file_path (str, optional): Path to a file with highlight patterns if mode="file" or "html".
        """
        self.file_path = file_path
        self.mode = mode

        if self.mode in ("html", "file") and not self.file_path:
            raise ValueError(f"file_path must be provided when mode='{self.mode}'")

        if self.mode == "dates":
            self.highlight_func = self._highlight_dates
        elif self.mode == "html":
            self.highlight_func = self._highlight_html()
        elif self.mode == "file":
            self.highlight_func = self._highlight_from_file()
        else:
            raise ValueError(
                f"Unknown highlight mode: {self.mode}, should be in 'dates', 'file', 'html'"
            )

    def pretty_print(self, text: str):
        """Prints a single text with optional highlighting.

        Args:
            text (str): The text to print.
            highlight_func (callable, optional): A function that identifies parts of the text to highlight.
                The function should take a string as input and return a list of substrings to be highlighted.
        """
        if self.highlight_func:
            matches = self.highlight_func(text)
            for match in matches:
                text = text.replace(match, colored(match, "green"))
        print(text)
        print("-" * 40)

    def pretty_print_dataset(self, dataset, n=5, label=None):
        """Prints up to n examples of the dataset with optional highlighting.

        If a label is provided, only examples with that label are printed.

        Args:
            dataset: A dataset containing text and labels.
            n (int): Maximum number of examples to print (default is 5).
            label (int, optional): If provided, only examples with this label are printed.
        """
        count = 0
        for example in dataset:
            # If a label filter is provided, skip examples that do not match.
            if label is not None and example["labels"] != label:
                continue

            print(f"Text {count + 1} (Label={example['labels']}):")
            self.pretty_print(example["text"])
            count += 1
            if count >= n:
                break

    def _highlight_dates(self, text):
        """Finds all date patterns in the text in the format YYYY-MM-DD.

        Args:
            text (str): The text to search.

        Returns:
            list: A list of date strings found in the text.
        """
        return re.findall(r"\d{4}-\d{2}-\d{2}", text)

    def _highlight_from_file(self):
        """Reads patterns from a file and returns a highlight function that highlights these patterns in the text.

        Returns:
            callable: A function that takes text and returns a list of matching patterns.
        """
        with open(self.file_path, "r", encoding="utf-8") as file:
            patterns = [line.strip() for line in file if line.strip()]

        def highlight_func(text):
            matches = []
            for pattern in patterns:
                if pattern in text:
                    matches.append(pattern)
            return matches

        return highlight_func

    def _highlight_html(self):
        """Reads HTML tag patterns from a file and returns a highlight function that highlights these tags in the text.

        Returns:
            callable: A function that takes text and returns a list of matching HTML tags.
        """
        with open(self.file_path, "r", encoding="utf-8") as file:
            patterns = [line.strip() for line in file if line.strip()]
            tags = []
            for line in patterns:
                tags.extend(line.split())

        def highlight_func(text):
            matches = []
            for tag in tags:
                if tag in text:
                    matches.append(tag)
            return matches

        return highlight_func
