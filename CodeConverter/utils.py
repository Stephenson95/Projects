import time
from pydantic import ValidationError

def retry(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry a function with a specified number of retries and delay - this is to account for unexpected LLM behaviour"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except (ValidationError, Exception) as e:
                    #print(f"Error: {e}. Retrying {retries + 1}/{max_retries}...")
                    time.sleep(delay)
                    retries += 1
            raise Exception(f"Function failed after {max_retries} retries.")
        return wrapper
    return decorator