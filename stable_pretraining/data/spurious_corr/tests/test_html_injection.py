import pytest
from stable_pretraining.data.spurious_corr.modifiers import HTMLInjection


@pytest.mark.unit
def test_html_injection_proportion(tmp_path):
    # Create a dummy tag file with 3 full tag pairs
    tag_path = tmp_path / "tags.txt"
    tag_path.write_text("<b> </b>\n<i> </i>\n<u> </u>\n")

    text = "this is a test sentence with eight tokens"
    token_count = len(text.split())

    # We'll test across different proportions of token-level injections
    for proportion in [0.1, 0.25, 0.5, 0.75, 1.0]:
        modifier = HTMLInjection.from_file(
            str(tag_path), location="random", token_proportion=proportion, seed=42
        )
        modified_text, label = modifier(text, "label")

        # Count total opening and closing tags
        opening_tags = ["<b>", "<i>", "<u>"]
        closing_tags = ["</b>", "</i>", "</u>"]

        open_count = sum(modified_text.count(tag) for tag in opening_tags)
        close_count = sum(modified_text.count(tag) for tag in closing_tags)

        # Each injection should add 1 opening + up to 1 closing tag
        expected_injections = max(1, int(token_count * proportion))

        assert open_count >= expected_injections, (
            f"Expected at least {expected_injections} opening tags, got {open_count}"
        )
        assert close_count <= open_count, (
            "There shouldn't be more closing tags than opening tags"
        )
        assert label == "label"


@pytest.mark.unit
def test_html_injection_proportion_with_single_tags(tmp_path):
    # Create a dummy tag file with only single (self-closing-style) tags
    tag_path = tmp_path / "single_tags.txt"
    tag_path.write_text("<br>\n<hr>\n<custom>\n")

    text = "this is a test sentence with eight tokens"
    token_count = len(text.split())

    for proportion in [0.1, 0.25, 0.5, 0.75, 1.0]:
        modifier = HTMLInjection.from_file(
            str(tag_path), location="random", token_proportion=proportion, seed=42
        )
        modified_text, label = modifier(text, "label")

        # Only single tags used, so count just those
        single_tags = ["<br>", "<hr>", "<custom>"]
        injected_count = sum(modified_text.count(tag) for tag in single_tags)

        expected_injections = max(1, int(token_count * proportion))
        assert injected_count == expected_injections, (
            f"Expected {expected_injections} tags, got {injected_count}"
        )
        assert label == "label"


@pytest.mark.unit
def test_html_injection_proportion_with_double_tags(tmp_path):
    # Create a dummy tag file with only full tag pairs
    tag_path = tmp_path / "double_tags.txt"
    tag_path.write_text("<b> </b>\n<i> </i>\n<u> </u>\n")

    text = "this is a test sentence with eight tokens"
    token_count = len(text.split())

    for proportion in [0.1, 0.25, 0.5, 0.75, 1.0]:
        modifier = HTMLInjection.from_file(
            str(tag_path), location="random", token_proportion=proportion, seed=42
        )
        modified_text, label = modifier(text, "label")

        opening_tags = ["<b>", "<i>", "<u>"]
        closing_tags = ["</b>", "</i>", "</u>"]

        open_count = sum(modified_text.count(tag) for tag in opening_tags)
        close_count = sum(modified_text.count(tag) for tag in closing_tags)

        expected_injections = max(1, int(token_count * proportion))

        assert open_count == expected_injections, (
            f"Expected {expected_injections} opening tags, got {open_count}"
        )
        assert close_count == expected_injections, (
            f"Expected {expected_injections} closing tags, got {close_count}"
        )
        assert label == "label"


@pytest.mark.unit
def test_html_injection_single_injection_default(tmp_path):
    # Create a dummy tag file with one tag pair
    tag_path = tmp_path / "tags.txt"
    tag_path.write_text("<mark> </mark>\n")

    text = "a short sentence with six tokens"
    modifier = HTMLInjection.from_file(str(tag_path), location="random", seed=42)

    modified_text, label = modifier(text, "label")

    # Expect exactly one opening tag and at most one closing tag
    opening_tag = "<mark>"
    closing_tag = "</mark>"

    open_count = modified_text.count(opening_tag)
    close_count = modified_text.count(closing_tag)

    assert open_count == 1, f"Expected exactly one opening tag, got {open_count}"
    assert close_count <= 1, f"Expected at most one closing tag, got {close_count}"
    assert label == "label"


@pytest.mark.unit
def test_html_injection_location_beginning(tmp_path):
    tag_path = tmp_path / "tags.txt"
    tag_path.write_text("<x> </x>\n")
    text = "sample sentence"

    modifier = HTMLInjection.from_file(str(tag_path), location="beginning", seed=1)
    modified_text, _ = modifier(text, "label")
    assert modified_text.startswith("<x>"), "Opening tag should be at the beginning"


@pytest.mark.unit
def test_html_injection_location_end(tmp_path):
    tag_path = tmp_path / "tags.txt"
    tag_path.write_text("<y> </y>\n")
    text = "another sample"

    modifier = HTMLInjection.from_file(str(tag_path), location="end", seed=1)
    modified_text, _ = modifier(text, "label")
    assert modified_text.endswith("</y>") or "<y>" in modified_text, (
        "Tag should be appended at end"
    )


@pytest.mark.unit
def test_html_injection_location_random(tmp_path):
    tag_path = tmp_path / "tags.txt"
    tag_path.write_text("<z> </z>\n")
    text = "tokens in various spots"

    modifier = HTMLInjection.from_file(str(tag_path), location="random", seed=123)
    modified_text, _ = modifier(text, "label")
    assert "<z>" in modified_text or "</z>" in modified_text


@pytest.mark.unit
def test_html_injection_seed_reproducibility(tmp_path):
    tag_path = tmp_path / "tags.txt"
    tag_path.write_text("<s> </s>\n")

    text = "reproducibility is key"
    mod1 = HTMLInjection.from_file(
        str(tag_path), location="random", token_proportion=0.5, seed=42
    )
    mod2 = HTMLInjection.from_file(
        str(tag_path), location="random", token_proportion=0.5, seed=42
    )

    out1, _ = mod1(text, "label")
    out2, _ = mod2(text, "label")
    assert out1 == out2


@pytest.mark.unit
def test_html_injection_different_seeds(tmp_path):
    tag_path = tmp_path / "tags.txt"
    tag_path.write_text("<q> </q>\n")
    text = "inject differently based on seed"

    mod1 = HTMLInjection.from_file(
        str(tag_path), location="random", token_proportion=0.5, seed=1
    )
    mod2 = HTMLInjection.from_file(
        str(tag_path), location="random", token_proportion=0.5, seed=2
    )

    out1, _ = mod1(text, "label")
    out2, _ = mod2(text, "label")
    assert out1 != out2, "Different seeds should yield different outputs"


@pytest.mark.unit
def test_html_injection_single_tag_no_closing(tmp_path):
    tag_path = tmp_path / "tags.txt"
    tag_path.write_text("<br>\n")  # Single, self-closing-like tag

    text = "check for self-closing"
    modifier = HTMLInjection.from_file(str(tag_path), location="end", seed=99)
    modified_text, _ = modifier(text, "label")

    assert "<br>" in modified_text and "</" not in modified_text, (
        "Only one tag should appear"
    )
