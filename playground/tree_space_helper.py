import sys
import os
import psutil
import numpy as np


def get_deep_size(obj, seen=None):
    """Recursively finds the actual size of an object in bytes."""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    # Handle NumPy arrays specifically (their data is separate from the object shell)
    if isinstance(obj, np.ndarray):
        return obj.nbytes

    # Recursively count members of dictionaries
    if isinstance(obj, dict):
        size += sum([get_deep_size(v, seen) for v in obj.values()])
        size += sum([get_deep_size(k, seen) for k in obj.keys()])

    # Recursively count members of iterables (list, tuple, set, etc.)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_deep_size(i, seen) for i in obj])

    # Handle custom classes/objects
    elif hasattr(obj, '__dict__'):
        size += get_deep_size(obj.__dict__, seen)

    return size


def print_memory_stats(label="Deep Memory Report", **objs):
    """Prints deep size of named objects and total process memory."""
    proc = psutil.Process(os.getpid())
    mem_info = proc.memory_full_info()

    print(f"\n{'='*15} {label} {'='*15}")

    for name, obj in objs.items():
        size_mb = get_deep_size(obj) / (1024**2)
        obj_type = type(obj).__name__
        print(f"{name:15} ({obj_type:10}): {size_mb:10.4f} MB")

    print(f"{'-'*50}")
    print(f"Total Process (RSS): {mem_info.rss / 1024**2:10.4f} MB (Physical)")
    print(f"Total Process (USS): {mem_info.uss / 1024**2:10.4f} MB (Unique)")
    print(f"{'='* (32 + len(label))}\n")
