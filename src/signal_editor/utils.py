import sys
import typing as t


def make_odd(n: int) -> int:
    return n + 1 if n % 2 == 0 else n


def make_even(n: int) -> int:
    return n - 1 if n % 2 == 1 else n


def target_pos(x: float, y: float) -> str:
    x = max(x, 0)
    time_seconds = x / 400
    hours = time_seconds // 3600
    minutes = (time_seconds % 3600) // 60
    seconds = time_seconds % 60
    return f"time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}\namplitude: {y:.4f}"


def print_attribute_types(obj: object, indent: int = 0, file: t.TextIO = sys.stdout) -> None:
    if hasattr(obj, "__dict__"):
        for attr_name, attr_value in vars(obj).items():
            print(
                "  " * indent + f"{attr_name}: {type(attr_value)}; Value: {attr_value}", file=file
            )
            if isinstance(attr_value, (list, tuple, dict)):
                print_attribute_types(attr_value, indent + 1, file=file)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            print("  " * indent + f"{key}: {type(value)}; Value: {value}", file=file)
            if hasattr(value, "__dict__") or isinstance(value, (list, tuple, dict)):
                print_attribute_types(value, indent + 1, file=file)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            print(
                "  " * indent + f"{item.__class__.__name__}: {type(item)}; Value: {item}", file=file
            )
            if hasattr(item, "__dict__") or isinstance(item, (list, tuple, dict)):
                print_attribute_types(item, indent + 1, file=file)
    else:
        print("  " * indent + f"{obj}: {type(obj)}", file=file)


def seconds_to_timestamp(seconds: float) -> str:
    # Extract hours, minutes, and remaining seconds
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_int = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)

    return f"{hours:02d}:{minutes:02d}:{seconds_int:02d}.{milliseconds:03d}"
