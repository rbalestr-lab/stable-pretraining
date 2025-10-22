import time
from huggingface_hub.utils import HfHubHTTPError
import requests
from loguru import logger as logging


def with_hf_retry_ratelimit(func, *args, delay=10, max_attempts=100, **kwargs):
    """Calls the given function with retry logic for HTTP 429 (Too Many Requests) errors.

    This function attempts to call ``func(*args, **kwargs)``. If a rate-limiting error (HTTP 429)
    is encountered—detected via exception type, status code, or error message—it will wait
    for the duration specified by the HTTP ``Retry-After`` header (if present), or fall back to
    the ``delay`` parameter, and then retry. Retries continue up to ``max_attempts`` times.
    Non-429 errors are immediately re-raised. If all attempts fail due to 429, the last
    exception is raised.

    Exceptions handled:
        - huggingface_hub.utils.HfHubHTTPError
        - requests.exceptions.HTTPError
        - OSError

    429 detection is performed by checking the exception's ``response.status_code`` (if available)
    or by searching for '429' or 'Too Many Requests' in the exception message.

    Args:
        func (callable): The function to call.
        *args: Positional arguments to pass to ``func``.
        delay (int, optional): Default wait time (in seconds) between retries if ``Retry-After``
            is not provided. Defaults to 10.
        max_attempts (int, optional): Maximum number of attempts before giving up. Defaults to 100.
        **kwargs: Keyword arguments to pass to ``func``.

    Returns:
        The return value of ``func(*args, **kwargs)`` if successful.

    Raises:
        Exception: The original exception if a non-429 error occurs, or if all attempts fail.

    Example:
        >>> from transformers import AutoModel
        >>> model = with_hf_retry_ratelimit(
        ...     AutoModel.from_pretrained,
        ...     "facebook/ijepa_vith14_1k",
        ...     delay=10,
        ...     max_attempts=5,
        ... )
    """
    attempts = 0
    while True:
        try:
            return func(*args, **kwargs)
        except (HfHubHTTPError, requests.exceptions.HTTPError, OSError) as e:
            # Try to extract status code and Retry-After
            status_code = None
            retry_after = delay
            if hasattr(e, "response") and e.response is not None:
                status_code = getattr(e.response, "status_code", None)
                retry_after = int(e.response.headers.get("Retry-After", delay))
            # Fallback: parse error message for 429
            if status_code == 429 or "429" in str(e) or "Too Many Requests" in str(e):
                attempts += 1
                if attempts >= max_attempts:
                    raise
                logging.warning(
                    f"429 received. Waiting {retry_after}s before retrying (attempt {attempts}/{max_attempts})..."
                )
                time.sleep(retry_after)
            else:
                raise
