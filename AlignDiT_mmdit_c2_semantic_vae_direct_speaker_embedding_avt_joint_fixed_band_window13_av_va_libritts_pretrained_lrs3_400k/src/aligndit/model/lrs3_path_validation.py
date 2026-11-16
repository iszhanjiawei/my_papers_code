"""Bounded, ordered path checks during LRS3 dataset construction only."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial


@contextmanager
def path_checked_rows(rows, validate, *, workers=16):
    """Yield ``(row, check_paths)`` without moving the caller's error boundary.

    Only the read-only path validator runs in threads. The caller retains every
    identity, text, provenance and accounting check in its original order, then
    invokes ``check_paths()`` at the original path-validation point. At most
    ``2 * workers`` rows are pending; a later parse error is likewise surfaced
    only after earlier rows. All threads join before this context exits, including
    caller/worker exceptions, so none can survive into a DataLoader fork.

    ``workers=0`` retains serial execution for equivalence checks. This changes
    scheduling only: no path result, parent resolution or receipt is cached.
    """
    if type(workers) is not int or not 0 <= workers <= 32:
        raise ValueError("LRS3 path-validation workers must be an integer from 0 to 32")
    source = iter(rows)
    pool = None
    pending = deque()
    try:
        if workers == 0:
            yield ((row, partial(validate, row)) for row in source)
            return
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="lrs3-path-validation")
        exhausted = False
        parse_error = None

        def fill():
            nonlocal exhausted, parse_error
            while not exhausted and len(pending) < 2 * workers:
                try:
                    row = next(source)
                except StopIteration:
                    exhausted = True
                except Exception as error:
                    # Prefetch may discover invalid JSON in a later record. Do
                    # not let it mask an earlier identity/text/path/CTC failure.
                    exhausted, parse_error = True, error
                else:
                    pending.append((row, pool.submit(validate, row)))

        def ordered():
            fill()
            while pending:
                row, result = pending.popleft()
                yield row, result.result
                fill()
            if parse_error is not None:
                raise parse_error

        yield ordered()
    finally:
        for _, future in pending:
            future.cancel()
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        close = getattr(source, "close", None)
        if close is not None:
            close()
