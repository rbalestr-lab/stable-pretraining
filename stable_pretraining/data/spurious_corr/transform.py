"""transform.py.

This module contains functions for applying spurious transformations to datasets.
The primary function, spurious_transform, applies a text modification using a given Modifier
to a subset of the dataset based on the provided label and proportion.
"""

import random
from datasets import concatenate_datasets  # assuming HuggingFace datasets


def spurious_transform(
    label_to_modify: int, dataset, modifier, text_proportion: float, seed=None
):
    """Applies a transformation to a subset of texts in the dataset that have the specified label.

    Args:
        label_to_modify (int): The label of the text to modify.
        dataset: The dataset containing the text data.
        modifier: An instance of a Modifier subclass that modifies (text, label).
        text_proportion (float): Proportion of texts to transform using the modifier (between 0 and 1).
        seed (int, optional): Seed for random sampling reproducibility.

    Returns:
        Dataset: A new dataset with the transformations applied to examples with the given label.
    """
    dataset_to_modify = dataset.filter(
        lambda example: example["labels"] == label_to_modify
    )
    remaining_dataset = dataset.filter(
        lambda example: example["labels"] != label_to_modify
    )

    # Determine the exact number of examples to modify
    n_examples = len(dataset_to_modify)
    n_to_modify = round(n_examples * text_proportion)

    # Create seeded random generator
    rng = random.Random(seed)

    # Randomly select exactly n_to_modify indices from the filtered dataset
    indices = list(range(n_examples))
    selected_indices = set(rng.sample(indices, n_to_modify))

    def modify_text(example, idx):
        # Modify only if the current index is in the selected indices
        if idx in selected_indices:
            new_text, new_label = modifier(example["text"], example["labels"])
            example["text"] = new_text
            example["labels"] = new_label
        return example

    modified_dataset = dataset_to_modify.map(modify_text, with_indices=True)
    return concatenate_datasets([modified_dataset, remaining_dataset])
