import datetime
import time

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"Function '{func.__name__}' started at {datetime.datetime.now()}")
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' ended at {datetime.datetime.now()}")
        print(f"Function '{func.__name__}' took {execution_time:.6f} seconds to execute.")
        return result
    return wrapper

def get_title(passage):
    return passage.split('\n')[0]