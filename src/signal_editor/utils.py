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


def seconds_to_timestamp(seconds: float) -> str:
    # Extract hours, minutes, and remaining seconds
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_int = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)

    return f"{hours:02d}:{minutes:02d}:{seconds_int:02d}.{milliseconds:03d}"
