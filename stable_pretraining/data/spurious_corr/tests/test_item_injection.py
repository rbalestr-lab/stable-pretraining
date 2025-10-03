import pytest
from stable_pretraining.data.spurious_corr.modifiers import ItemInjection


@pytest.mark.unit
def test_injection_proportion():
    text = "this is a test sentence with eight tokens"
    token_count = len(text.split())

    for proportion in [0.1, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.9, 1.0]:
        modifier = ItemInjection.from_list(
            ["X"], token_proportion=proportion, location="end", seed=42
        )
        modified_text, label = modifier(text, "original_label")
        injected_count = modified_text.count("X")
        expected_count = max(1, int(token_count * proportion))
        assert injected_count == expected_count, (
            f"Expected {expected_count}, got {injected_count}"
        )
        assert label == "original_label"


@pytest.mark.unit
def test_injection_single_token():
    """Test the injection of a single token into a text string."""
    text = "this is a test sentence with eight tokens"

    modifier = ItemInjection.from_list(
        ["X"], token_proportion=0, location="random", seed=42
    )
    modified_text, label = modifier(text, "original_label")
    injected_count = modified_text.count("X")
    expected_count = 1
    assert injected_count == expected_count, (
        f"Expected {expected_count}, got {injected_count}"
    )
    assert label == "original_label"


@pytest.mark.unit
def test_injection_location_beginning():
    text = "hello world"
    modifier = ItemInjection.from_list(["<X>"], location="beginning", seed=42)
    modified_text, _ = modifier(text, "label")
    assert modified_text.startswith("<X>"), "Injection should be at the beginning"


@pytest.mark.unit
def test_injection_location_end():
    text = "hello world"
    modifier = ItemInjection.from_list(["<Y>"], location="end", seed=42)
    modified_text, _ = modifier(text, "label")
    assert modified_text.endswith("<Y>"), "Injection should be at the end"


@pytest.mark.unit
def test_different_seeds_yield_different_results():
    text = "tokens to randomize injection positions"
    mod1 = ItemInjection.from_list(
        ["<A>"], token_proportion=0.5, location="random", seed=1
    )
    mod2 = ItemInjection.from_list(
        ["<A>"], token_proportion=0.5, location="random", seed=2
    )

    text1, _ = mod1(text, "label")
    text2, _ = mod2(text, "label")

    assert text1 != text2, "Different seeds should yield different injection positions"
