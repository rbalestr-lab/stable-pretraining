"""Demonstration of the spurious_corr library capabilities."""

from stable_pretraining.data.spurious_corr.modifiers import (
    ItemInjection,
    HTMLInjection,
    CompositeModifier,
)
from stable_pretraining.data.spurious_corr.generators import SpuriousDateGenerator
from stable_pretraining.data.spurious_corr.utils import (
    pretty_print,
    pretty_print_dataset,
    highlight_from_file,
    highlight_from_list,
    highlight_html,
    highlight_dates,
)
from stable_pretraining.data.spurious_corr.transform import spurious_transform
from datasets import load_dataset


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def example_1_basic_date_injection():
    """Example 1: Basic date injection at different locations."""
    print_section("Example 1: Date Injection with SpuriousDateGenerator")
    text = "Machine learning models require careful evaluation and testing procedures."
    print(f"\nOriginal text: '{text}'\n")
    print("-" * 50)

    # Create date generator
    date_gen = SpuriousDateGenerator(year_range=(1900, 2100), seed=42)

    # Example 1a: Inject at beginning
    modifier_start = ItemInjection.from_function(
        injection_func=date_gen, location="beginning", token_proportion=0.4, seed=42
    )

    modified_text, _ = modifier_start(text, 1)
    print("1a. Date injection at BEGINNING:")
    pretty_print(modified_text, highlight_dates)
    print("-" * 50)

    # Example 1b: Inject at end
    modifier_end = ItemInjection.from_function(
        injection_func=date_gen, location="end", token_proportion=0.4, seed=43
    )

    modified_text, _ = modifier_end(text, 1)
    print("1b. Date injection at END:")
    pretty_print(modified_text, highlight_dates)
    print("-" * 50)

    # Example 1c: Inject at random locations
    modifier_random = ItemInjection.from_function(
        injection_func=date_gen, location="random", token_proportion=0.4, seed=44
    )

    modified_text, _ = modifier_random(text, 1)
    print("1c. Date injection at RANDOM positions:")
    pretty_print(modified_text, highlight_dates)
    print("-" * 50)


def example_2_file_based_injection():
    """Example 2: Inject tokens from files (countries, colors)."""
    print_section("Example 2: File-Based Token Injection")

    # Example 2a: Country injection
    country_modifier = ItemInjection.from_file(
        file_path="examples/data/countries.txt",
        location="random",
        token_proportion=0.3,
        seed=42,
    )

    country_highlighter = highlight_from_file("examples/data/countries.txt")
    text = "International trade agreements benefit global economic stability."
    modified_text, _ = country_modifier(text, 1)

    print("2a. Country injection:")
    pretty_print(modified_text, country_highlighter)
    print("-" * 50)

    # Example 2b: Color injection
    color_modifier = ItemInjection.from_file(
        file_path="examples/data/colors.txt",
        location="random",
        token_proportion=1,
        seed=42,
    )

    color_highlighter = highlight_from_file("examples/data/colors.txt")
    text = "The sunset painted the sky beautifully."
    modified_text, _ = color_modifier(text, 1)

    print("2b. Color injection:")
    pretty_print(modified_text, color_highlighter)
    print("-" * 50)

    # Example 2c: Custom word list
    custom_modifier = ItemInjection.from_list(
        items=["URGENT", "BREAKING", "EXCLUSIVE", "ALERT"],
        location="random",
        token_proportion=1,
        seed=42,
    )

    custom_highlighter = highlight_from_list(
        ["URGENT", "BREAKING", "EXCLUSIVE", "ALERT"]
    )
    text = "Weather forecast predicts rain tomorrow."
    modified_text, _ = custom_modifier(text, 1)

    print("2c. Custom urgent words:")
    pretty_print(modified_text, custom_highlighter)
    print("-" * 50)


def example_3_html_injection():
    """Example 3: HTML tag injection with different strategies."""
    print_section("Example 3: HTML Tag Injection")
    html_highlighter = highlight_html("examples/data/html_tags.txt")
    text = "This is an important announcement for all users."

    # Example 3a: Single HTML tag at beginning
    begin_modifier = HTMLInjection.from_file(
        file_path="examples/data/html_tags.txt", location="beginning", seed=42
    )

    modified_text, _ = begin_modifier(text, 1)
    print("3a. Beginning single HTML tag injection:")
    pretty_print(modified_text, html_highlighter)
    print("-" * 50)

    # Example 3b: Single HTML tag at random location
    random_modifier = HTMLInjection.from_file(
        file_path="examples/data/html_tags.txt", location="random", seed=43
    )

    modified_text, _ = random_modifier(text, 1)
    print("3b. Random single HTML tag injection:")
    pretty_print(modified_text, html_highlighter)
    print("-" * 50)

    # Example 3c: Single HTML tag at end
    end_modifier = HTMLInjection.from_file(
        file_path="examples/data/html_tags.txt", location="end", seed=44
    )

    modified_text, _ = end_modifier(text, 1)
    print("3c. End single HTML tag injection:")
    pretty_print(modified_text, html_highlighter)
    print("-" * 50)

    # Example 3d: Multiple HTML tags at random locations
    multi_random_modifier = HTMLInjection.from_file(
        file_path="examples/data/html_tags.txt",
        location="random",
        token_proportion=0.5,
        seed=45,
    )

    modified_text, _ = multi_random_modifier(text, 1)
    print("3d. Multiple random HTML tag injection:")
    pretty_print(modified_text, html_highlighter)
    print("-" * 50)


def example_4_multiple_injections():
    """Example 4: Multiple different injection types combined."""
    print_section("Example 4: Multiple Injection Types Combined")

    # Date at beginning
    date_modifier = ItemInjection.from_function(
        SpuriousDateGenerator(year_range=(2020, 2024), seed=42),
        location="beginning",
        token_proportion=0,
    )

    # Country in middle
    country_modifier = ItemInjection.from_file(
        file_path="examples/data/countries.txt",
        location="random",
        token_proportion=0,
        seed=43,
    )

    # Color at end
    color_modifier = ItemInjection.from_file(
        file_path="examples/data/colors.txt",
        location="end",
        token_proportion=0,
        seed=44,
    )

    # Combine all
    multi_modifier = CompositeModifier(
        [date_modifier, country_modifier, color_modifier]
    )

    text = "Economic analysis shows promising trends in renewable energy sectors."
    modified_text, _ = multi_modifier(text, 1)
    print("4. Multiple injection types:")
    print(modified_text)
    print("-" * 50)


def example_5_token_density_comparison():
    """Example 5: Compare different token proportion levels."""
    print_section("Example 5: Token Proportion Comparison")

    text = "Artificial intelligence and machine learning technologies are transforming industries."
    highlighter = highlight_dates

    token_proportions = [0, 0.3, 0.5, 0.8, 1.0]  # 0 injects a single token

    for density in token_proportions:
        modifier = ItemInjection.from_function(
            SpuriousDateGenerator(year_range=(2020, 2024), seed=42),
            location="random",
            token_proportion=density,
            seed=42,
        )

        modified_text, _ = modifier(text, 1)
        print(f"\nToken proportion {density}:")
        pretty_print(modified_text, highlighter)


def example_6_dataset_simulation():
    """Example 6: Simulate dataset-level spurious correlations."""
    print_section(
        "Example 6: Dataset-Level Spurious Correlation Simulation using spurious_transform"
    )

    # Load IMDB dataset
    dataset = load_dataset("imdb", split="train")  # Load full training dataset

    # Create date modifier
    date_modifier = ItemInjection.from_function(
        SpuriousDateGenerator(year_range=(2020, 2024), seed=42),
        location="random",
        token_proportion=0.1,
        seed=42,
    )

    print("Simulating spurious correlation: Add dates to positive reviews only\n")

    # Apply spurious transformation
    modified_dataset = spurious_transform(
        label_to_modify=1,  # Target positive reviews
        dataset=dataset,
        modifier=date_modifier,
        text_proportion=1.0,  # Apply to all positive reviews
        seed=42,
    )

    # Print examples using pretty_print_dataset
    print("Positive reviews (with injected dates):")
    pretty_print_dataset(modified_dataset, n=3, highlight_func=highlight_dates, label=1)

    print("\nNegative reviews (original):")
    pretty_print_dataset(modified_dataset, n=3, highlight_func=highlight_dates, label=0)


def main():
    """Run all examples demonstrating library capabilities."""
    example_1_basic_date_injection()
    example_2_file_based_injection()
    example_3_html_injection()
    example_4_multiple_injections()
    example_5_token_density_comparison()
    example_6_dataset_simulation()


if __name__ == "__main__":
    main()
