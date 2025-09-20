__all__ = [
    "__title__",
    "__summary__",
    "__version__",
    "__author__",
    "__license__",
    "__url__",
]

__title__ = "stable-pretraining"
__summary__ = "Self-Supervised Learning Research Library"
__author__ = "Randall Balestriero, Hugues Van Assel, Lucas Maes"
__license__ = "MIT"
__url__ = "https://rbalestr-lab.github.io/stable-pretraining.github.io"

try:
    from ._version import version as __version__
except Exception:
    try:
        from importlib.metadata import version as _pkg_version

        __version__ = _pkg_version("stable-pretraining")
    except Exception:
        raise ImportError("Could not determine stable-pretraining version")
