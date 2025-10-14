"""utils.py.

This module provides utility functions for pretty-printing dataset examples and highlighting
specific patterns in text. These functions are useful for debugging and visualizing the modifications
applied to the dataset.
"""

import re
from termcolor import colored


def pretty_print(text: str, highlight_func=None):
    """Prints a single text with optional highlighting.

    Args:
        text (str): The text to print.
        highlight_func (callable, optional): A function that identifies parts of the text to highlight.
            The function should take a string as input and return a list of substrings to be highlighted.
    """
    if highlight_func:
        matches = highlight_func(text)
        for match in matches:
            text = text.replace(match, colored(match, "green"))
    print(text)
    print("-" * 40)


def pretty_print_dataset(dataset, n=5, highlight_func=None, label=None):
    """Prints up to n examples of the dataset with optional highlighting.

    If a label is provided, only examples with that label are printed.

    Args:
        dataset: A dataset containing text and labels.
        n (int): Maximum number of examples to print (default is 5).
        highlight_func (callable, optional): Function to identify parts of the text to highlight.
        label (int, optional): If provided, only examples with this label are printed.
    """
    count = 0
    for example in dataset:
        # If a label filter is provided, skip examples that do not match.
        if label is not None and example["labels"] != label:
            continue

        print(f"Text {count + 1} (Label={example['labels']}):")
        pretty_print(example["text"], highlight_func)
        count += 1
        if count >= n:
            break


def highlight_dates(text):
    """Finds all date patterns in the text in the format YYYY-MM-DD.

    Args:
        text (str): The text to search.

    Returns:
        list: A list of date strings found in the text.
    """
    return re.findall(r"\d{4}-\d{2}-\d{2}", text)


def highlight_from_file(file_path):
    """Reads patterns from a file and returns a highlight function that highlights these patterns in the text.

    Args:
        file_path (str): Path to the file containing patterns.

    Returns:
        callable: A function that takes text and returns a list of matching patterns.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        patterns = [line.strip() for line in file if line.strip()]

    def highlight_func(text):
        matches = []
        for pattern in patterns:
            if pattern in text:
                matches.append(pattern)
        return matches

    return highlight_func


def highlight_html(file_path):
    """Reads HTML tag patterns from a file and returns a highlight function that highlights these tags in the text.

    Args:
        file_path (str): Path to the file containing HTML tag patterns.

    Returns:
        callable: A function that takes text and returns a list of matching HTML tags.
    """
    with open(file_path, "r", encoding="utf-8") as file:
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
