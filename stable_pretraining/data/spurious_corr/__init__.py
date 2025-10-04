"""spurious_corr package.

This package provides tools to apply and test the effect of various transformations
(such as injecting spurious text) on datasets for research and testing purposes.
It includes functionality for text transformations, various generators for spurious
text, and utilities for printing and highlighting text.
"""

from .modifiers import (
    Modifier as Modifier,
    CompositeModifier as CompositeModifier,
    ItemInjection as ItemInjection,
    HTMLInjection as HTMLInjection,
)
from .transform import spurious_transform as spurious_transform
from .generators import SpuriousDateGenerator as SpuriousDateGenerator
from .utils import (
    pretty_print as pretty_print,
    pretty_print_dataset as pretty_print_dataset,
    highlight_dates as highlight_dates,
    highlight_from_file as highlight_from_file,
    highlight_html as highlight_html,
)
