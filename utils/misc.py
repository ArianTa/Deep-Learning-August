import time
import datetime
from contextlib import contextmanager


@contextmanager
def measure_time(label,):
    """
    Context manager to measure time of computation.
    >>> with measure_time('Heavy computation'):
    >>>     do_heavy_computation()
    'Duration of [Heavy computation]: 0:04:07.765971'
    Parameters
    ----------
    label: str
        The label by which the computation will be referred
    """
    print("{}...".format(label))
    start = time.time()
    yield
    end = time.time()
    print(
        "Duration of [{}]: {}".format(
            label, datetime.timedelta(seconds=end - start),
        )
    )


class dummy_context_mgr:
    def __enter__(self,):
        return None

    def __exit__(
        self, exc_type, exc_value, traceback,
    ):
        return False


def format_label(label,):
    # label = label.split("_")[1]  # takes only the superclass
    return label
