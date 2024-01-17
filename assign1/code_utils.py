from contextlib import contextmanager
import time
@contextmanager
def measure_time():
    start_time = time.time()
    yield
    end_time = time.time()
    print(f"The execution time is {end_time - start_time} seconds")
