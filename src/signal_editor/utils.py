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
    return (
        f"time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}\namplitude: {y:.4f}"
    )


