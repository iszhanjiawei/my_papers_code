"""Bounded, ordered read-only validation for the LRS3 cache finalizer.

Only the owner aggregates results and publishes manifests. Spawned workers get
small immutable context and each row's own labels/exclusion/import receipt.
They never inherit publication locks or receive the full recovery/receipt maps.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, TimeoutError
from contextlib import contextmanager, nullcontext
import multiprocessing
import os
from pathlib import Path
import signal
import threading

from lrs3_cache_parallel import ShutdownRequested


class ValidationControl:
    """Record owner signals without interrupting pool/lock registration."""
    def __init__(self):
        self.signum = None

    @contextmanager
    def signals(self):
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        def requested(signum, _frame):
            if self.signum is None:
                self.signum = signum
        for sig in previous:
            signal.signal(sig, requested)
        try:
            yield self
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    def check(self):
        if self.signum is not None:
            raise ShutdownRequested(self.signum)


def assert_no_cache_lock_fds(root):
    """Fail closed if a spawned validator acquired an owner's lock descriptor."""
    identities = {(p.stat().st_dev, p.stat().st_ino)
                  for p in (Path(root) / 'state/locks').glob('*.lock')}
    for path in Path('/proc/self/fd').iterdir():
        try:
            value = os.fstat(int(path.name))
        except OSError:
            continue  # The descriptor used to enumerate /proc has already closed.
        if (value.st_dev, value.st_ino) in identities:
            raise RuntimeError('Spawned validator inherited a cache lock descriptor')


_PROCESS_CONTEXT = None
_PROCESS_STOP = None


def _initialize_process(context, stop):
    global _PROCESS_CONTEXT, _PROCESS_STOP
    # The owner coordinates shutdown. A foreground-group Ctrl-C must not tear
    # down workers halfway through their multiprocessing queue bookkeeping.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    assert_no_cache_lock_fds(context['root'])
    _PROCESS_CONTEXT, _PROCESS_STOP = context, stop


def _validate_batch(batch, context, stop):
    from lrs3_finalize_cache import validate
    results = []
    for row, label, exclusion, receipt in batch:
        if stop.is_set():
            raise RuntimeError('Validation cancelled by owner')
        sid = row['sample_id']
        results.append(validate(
            row, context['root'], context['vocabulary'],
            {sid: label} if label is not None else {},
            context['strict_provenance'], context['text_policy'],
            {sid: exclusion} if exclusion is not None else {},
            context['imported_source'], receipt))
    return results


def _process_batch(batch):
    return _validate_batch(batch, _PROCESS_CONTEXT, _PROCESS_STOP)


def ordered_validate(rows, *, root, vocabulary, recovered, strict_provenance,
                     text_policy, media_quality_exclusions, imported_source,
                     import_receipts, executor='thread', workers=8, batch_size=16,
                     control=None):
    """Yield unchanged validate() results in source order, with bounded work.

    Exhaust or explicitly close this generator before releasing owner locks.
    Its finally block cancels queued work and joins all validation workers, also
    when aggregation raises or the owner receives a cooperative TERM/INT.
    """
    if executor not in {'thread', 'process'}:
        raise ValueError('Unknown validation executor')
    if workers < 1 or batch_size < 1:
        raise ValueError('Validation workers and batch size must be positive')
    own_control = control is None
    control = control or ValidationControl()
    # The terminal contains all excluded IDs; validate_import_receipt only uses
    # the small source contract. Per-row exclusions travel with the work item.
    source = ({k: v for k, v in imported_source.items() if k != 'terminal'}
              if imported_source is not None else None)
    context = dict(root=Path(root), vocabulary=vocabulary,
                   strict_provenance=strict_provenance, text_policy=text_policy,
                   imported_source=source)
    recovered = recovered or {}
    exclusions = media_quality_exclusions or {}
    receipts = import_receipts or {}
    iterator = iter(rows)
    pending = deque()

    def next_batch():
        batch = []
        for _ in range(batch_size):
            try:
                row = next(iterator)
            except StopIteration:
                break
            sid = row['sample_id']
            batch.append((row, recovered.get(sid), exclusions.get(sid), receipts.get(sid)))
        return batch

    with control.signals() if own_control else nullcontext():
        control.check()
        if executor == 'process':
            spawn = multiprocessing.get_context('spawn')
            stop = spawn.Event()
            pool = ProcessPoolExecutor(max_workers=workers, mp_context=spawn,
                                       initializer=_initialize_process, initargs=(context, stop))
            submit = lambda batch: pool.submit(_process_batch, batch)
        else:
            stop = threading.Event()
            pool = ThreadPoolExecutor(max_workers=workers)
            submit = lambda batch: pool.submit(_validate_batch, batch, context, stop)
        try:
            for _ in range(2 * workers):
                control.check()
                batch = next_batch()
                if not batch:
                    break
                pending.append(submit(batch))
            while pending:
                control.check()
                future = pending[0]
                try:
                    result = future.result(timeout=.1)
                except TimeoutError:
                    # A validator's own TimeoutError is a real failure.
                    if future.done():
                        result = future.result()
                    else:
                        continue
                pending.popleft()
                control.check()
                batch = next_batch()
                if batch:
                    pending.append(submit(batch))
                for item in result:
                    control.check()
                    yield item
            control.check()
        finally:
            stop.set()
            for future in pending:
                future.cancel()
            # No lock is released until every active validator has exited and
            # ProcessPoolExecutor has joined/reaped its worker processes.
            pool.shutdown(wait=True, cancel_futures=True)
