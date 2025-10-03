import pytest
from spurious_corr.generators import SpuriousDateGenerator
from spurious_corr.modifiers import ItemInjection
from spurious_corr.transform import spurious_transform
import llm_research.data


@pytest.mark.download
def test_spurious_transform_proportion_multiple():
    dataset_name = "imdb"
    data = llm_research.data.from_name(dataset_name)
    train_dataset = data["train"].select(range(200))

    label_to_modify = 1

    with open("spurious_corr/data/countries.txt", "r", encoding="utf-8") as f:
        country_list = [line.strip() for line in f if line.strip()]

    modifier = ItemInjection.from_list(country_list, token_proportion=0.5, seed=23)

    originals = [ex for ex in train_dataset]

    for text_proportion in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
        transformed = spurious_transform(
            label_to_modify=label_to_modify,
            dataset=train_dataset,
            modifier=modifier,
            text_proportion=text_proportion,
            seed=42,
        )

        # Count modified examples (compare original vs transformed)
        modified_count = sum(
            1
            for orig, mod in zip(originals, transformed)
            if orig["labels"] == label_to_modify and orig["text"] != mod["text"]
        )

        total_to_modify = sum(1 for ex in originals if ex["labels"] == label_to_modify)
        expected = round(total_to_modify * text_proportion)

        print(
            f"[text_proportion={text_proportion}] Modified: {modified_count} / Expected: {expected}"
        )
        assert modified_count == expected, (
            f"Expected {expected}, but got {modified_count} at proportion {text_proportion}"
        )


@pytest.mark.download
def test_spurious_transform_reproducible():
    dataset_name = "imdb"
    data = llm_research.data.from_name(dataset_name)
    train_dataset = data["train"].select(range(200))

    date_generator_1 = SpuriousDateGenerator(seed=19, with_replacement=False)
    modifier_1 = ItemInjection.from_function(
        date_generator_1, token_proportion=0.5, seed=19
    )

    date_generator_2 = SpuriousDateGenerator(seed=19, with_replacement=False)
    modifier_2 = ItemInjection.from_function(
        date_generator_2, token_proportion=0.5, seed=19
    )

    transformed1 = spurious_transform(0, train_dataset, modifier_1, 0.3, seed=19)
    transformed2 = spurious_transform(0, train_dataset, modifier_2, 0.3, seed=19)

    texts1 = [ex["text"] for ex in transformed1]
    texts2 = [ex["text"] for ex in transformed2]

    assert texts1 == texts2, "Expected reproducible output with same seed"


@pytest.mark.download
def test_spurious_transform_different_seeds():
    dataset_name = "imdb"
    data = llm_research.data.from_name(dataset_name)
    train_dataset = data["train"].select(range(200))

    date_generator_1 = SpuriousDateGenerator(seed=19, with_replacement=False)
    modifier_1 = ItemInjection.from_function(
        date_generator_1, token_proportion=0.5, seed=19
    )

    date_generator_2 = SpuriousDateGenerator(seed=19, with_replacement=False)
    modifier_2 = ItemInjection.from_function(
        date_generator_2, token_proportion=0.5, seed=19
    )

    transformed1 = spurious_transform(0, train_dataset, modifier_1, 0.3, seed=19)
    transformed2 = spurious_transform(0, train_dataset, modifier_2, 0.3, seed=20)

    texts1 = [ex["text"] for ex in transformed1]
    texts2 = [ex["text"] for ex in transformed2]

    assert texts1 != texts2, "Expected different outputs with different seeds"
