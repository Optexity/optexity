import asyncio
import functools
import logging
import time


def timeit(func):
    logger = logging.getLogger(func.__module__)

    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                logger.info(
                    f"{func.__qualname__} took {time.perf_counter() - start:.4f}s"
                )

        return async_wrapper
    else:

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                logger.info(
                    f"{func.__qualname__} took {time.perf_counter() - start:.4f}s"
                )

        return sync_wrapper
