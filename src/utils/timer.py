import time
from contextlib import contextmanager


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.0f} s")


def wait(sec: float):
    time.sleep(sec)


if __name__ == "__main__":
    # 例
    with timer("wait"):
        wait(1.0)
