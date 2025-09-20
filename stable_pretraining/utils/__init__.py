"""Stable SSL utilities package.

This package provides various utilities for self-supervised learning experiments
including distributed training helpers, custom autograd functions, neural network
modules, stable linear algebra operations, data generation, visualization, and
configuration management.
"""

# Import from submodules for backward compatibility

from .batch_utils import get_data_from_batch_or_outputs
from .config import (
    adapt_resnet_for_lowres,
    execute_from_config,
    find_module,
    replace_module,
    rgetattr,
    rsetattr,
)
from .data_generation import (
    generate_dae_samples,
    generate_dm_samples,
    generate_ssl_samples,
    generate_sup_samples,
)
from .distance_metrics import (
    compute_pairwise_distances,
    compute_pairwise_distances_chunked,
)
from .distributed import (
    FullGatherLayer,
    all_gather,
    all_reduce,
    is_dist_avail_and_initialized,
)
from .inspection_utils import (
    broadcast_param_to_list,
    dict_values,
    get_required_fn_parameters,
)
from .nn_modules import EMA, ImageToVideoEncoder, Normalize, OrderedQueue, UnsortedQueue

__all__ = [
    # autograd
    "MyReLU",
    "OrderedCovariance",
    "Covariance",
    "ordered_covariance",
    "covariance",
    # config
    "execute_from_config",
    "adapt_resnet_for_lowres",
    "rsetattr",
    "rgetattr",
    "find_module",
    "replace_module",
    # data_generation
    "generate_dae_samples",
    "generate_sup_samples",
    "generate_dm_samples",
    "generate_ssl_samples",
    # distance_metrics
    "compute_pairwise_distances",
    "compute_pairwise_distances_chunked",
    # distributed
    "is_dist_avail_and_initialized",
    "all_gather",
    "all_reduce",
    "FullGatherLayer",
    # inspection_utils
    "get_required_fn_parameters",
    "dict_values",
    "broadcast_param_to_list",
    # linalg
    "stable_eigvalsh",
    "stable_eigh",
    "stable_svd",
    "stable_svdvals",
    # nn_modules
    "ImageToVideoEncoder",
    "Normalize",
    "UnsortedQueue",
    "OrderedQueue",
    "EMA",
    # visualization
    "imshow_with_grid",
    "visualize_images_graph",
    # batch_utils
    "get_data_from_batch_or_outputs",
]
