import math
from typing import Generic, Iterable, TypeVar

from tqdm.contrib.concurrent import thread_map

from opensplat3d.utils.scene_utils import CameraInfo

T = TypeVar("T")


def split_hold(x: list[T], hold: float | int = 8) -> tuple[list[T], list[T]]:
    """
    Split a sequence into A and B sets based on the hold parameter.
    If hold is 0, all data goes to the first set.
    If hold is greater than 1, it uses a hold-out strategy where every nth element is used for the second set.
    If hold is a float, it represents the fraction of data to hold out for the second set (e.g., 0.2 means 20% of the data is used for the second set).
    """
    assert hold >= 0, "Hold must be a non-negative number"
    if hold == 0:
        A, B = x, []
    elif hold > 1:
        nth = int(hold)
        A = [c for idx, c in enumerate(x) if idx % nth != 0]
        B = [c for idx, c in enumerate(x) if idx % nth == 0]
    else:
        hold = float(hold)
        num_val = int(hold * len(x))
        A = x[:-num_val]
        B = x[-num_val:]
    return A, B


def sample(x: list[T], num: int = -1, nth: int = -1, dist: str = "uniform") -> list[T]:
    """
    Sample a list of items based on the specified distribution and parameters.
    - num: Number of items to sample. If -1, no limit is applied.
    - nth: Step size for sampling. If > 1, every nth item is sampled
    - dist: Distribution type, either "first" to take the first num items or "uniform" to sample uniformly (default).
    """
    if num > 0:
        if dist == "first":
            x = x[:num]
        elif dist == "uniform":
            if len(x) > num:
                x = x[:: math.ceil(len(x) / num)]

    if nth > 1:
        x = x[::nth]

    return x


class Reader(Generic[T]):
    def __init__(self, train_keys: Iterable[T], test_keys: Iterable[T]):
        self.train_keys = train_keys
        self.test_keys = test_keys

    def read_camera(self, key: T) -> CameraInfo:
        """
        Read camera information for a given index.
        This method should be implemented in subclasses to read camera information
        from the specific format or source.
        """
        raise NotImplementedError(
            "This method should be implemented in subclasses to read camera information."
        )

    def load_train(self, progbar: bool = False) -> list[CameraInfo]:
        return self._load(
            self.train_keys, desc="Loading training cameras", progbar=progbar
        )

    def load_test(self, progbar: bool = False) -> list[CameraInfo]:
        return self._load(
            self.test_keys, desc="Loading testing cameras", progbar=progbar
        )

    def _load(
        self, keys: Iterable[T], desc: str | None = None, progbar: bool = False
    ) -> list[CameraInfo]:
        return thread_map(self.read_camera, keys, desc=desc, disable=not progbar)
