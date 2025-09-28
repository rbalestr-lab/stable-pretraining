"""Utility functions for handling batch and outputs dictionaries in callbacks."""

from typing import Any, Dict, Optional, Union, Iterable

from loguru import logger as logging


def get_data_from_batch_or_outputs(
    key: Union[Iterable[str], str],
    batch: Dict[str, Any],
    outputs: Optional[Dict[str, Any]] = None,
    caller_name: str = "Callback",
) -> Optional[Any]:
    """Get data from either outputs or batch dictionary.

    In PyTorch Lightning, the outputs parameter in callbacks contains the return
    value from training_step/validation_step, while batch contains the original
    input. Since forward methods may modify batch in-place but Lightning creates
    a copy for outputs, we need to check both.

    Args:
        key: The key(s) to look for in the dictionaries
        batch: The original batch dictionary
        outputs: The outputs dictionary from training/validation step
        caller_name: Name of the calling function/class for logging

    Returns:
        The data associated with the key, or None if not found
    """
    output_as_list = True
    if type(key) is str:
        key = [key]
        output_as_list = False
    out = []
    for k in key:
        # First check outputs (which contains the forward pass results)
        if outputs is not None and k in outputs:
            out.append(outputs[k])
        elif k in batch:
            out.append(batch[k])
        else:
            msg = (
                f"{caller_name}: Key '{k}' not found in batch or outputs. "
                f"Available batch keys: {list(batch.keys())}, "
                f"Available output keys: {list(outputs.keys()) if outputs else 'None'}"
            )
            logging.warning(msg)
            raise ValueError(msg)
    if output_as_list:
        return out
    return out[0]
