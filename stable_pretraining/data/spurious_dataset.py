"""spurious_dataset.py.

Unified module for constructing spurious correlation datasets.

All spurious injections are now file-based via ItemInjection.from_file or similar.
"""

import random
from datasets import concatenate_datasets

# The modifiers come from your modifiers.py file
from transforms import CompositeModifier


class SpuriousDatasetBuilder:
    """Builds datasets with spurious correlations by applying Modifier objects."""

    def __init__(self, seed=None):
        """Constructor for the DatasetBuilder.

        Args:
        seed (int, optional): Seed for reproducibility.
        """
        self.rng = random.Random(seed)

    def _apply_modifier_to_subset(self, dataset, label_to_modify, modifier, proportion):
        """Apply a Modifier (or CompositeModifier) to a proportion of samples with a given label."""
        dataset_to_modify = dataset.filter(lambda ex: ex["labels"] == label_to_modify)
        remaining_dataset = dataset.filter(lambda ex: ex["labels"] != label_to_modify)

        n_examples = len(dataset_to_modify)
        n_to_modify = round(n_examples * proportion)
        indices = list(range(n_examples))
        selected_indices = set(self.rng.sample(indices, n_to_modify))

        def modify_example(example, idx):
            if idx in selected_indices:
                new_text, new_label = modifier(example["text"], example["labels"])
                example["text"] = new_text
                example["labels"] = new_label
            return example

        modified_subset = dataset_to_modify.map(modify_example, with_indices=True)
        return concatenate_datasets([modified_subset, remaining_dataset])

    def build_spurious_dataset(
        self, dataset, modifiers_config, label_to_modify, proportion
    ):
        """Construct a spurious dataset.

        Args:
            dataset: Hugging Face Dataset object with "text" and "labels".
            modifiers_config (list[Modifier] or Modifier): One or more modifiers to apply.
            label_to_modify (int): Which label group to modify.
            proportion (float): Proportion of examples within that label to modify (0-1).

        Returns:
            Dataset: Modified dataset.
        """
        if isinstance(modifiers_config, list):
            modifier = CompositeModifier(modifiers_config)
        else:
            modifier = modifiers_config

        return self._apply_modifier_to_subset(
            dataset, label_to_modify, modifier, proportion
        )
