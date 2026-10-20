import threading
from concurrent.futures import ThreadPoolExecutor


class Busy(RuntimeError):
    pass


class InferenceExecutor:
    """One TPU owner with bounded admission, shared fairly by images and streams."""

    def __init__(self, queue_size):
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tpu")
        self.slots = threading.BoundedSemaphore(queue_size + 1)

    def submit(self, function, *args):
        if not self.slots.acquire(blocking=False):
            raise Busy("inference queue is full; retry later")
        try:
            future = self.pool.submit(function, *args)
        except Exception:
            self.slots.release()
            raise
        future.add_done_callback(lambda _: self.slots.release())
        return future

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
