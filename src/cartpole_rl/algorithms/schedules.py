"""Linear schedules (exploration epsilon, PPO learning-rate decay)."""


def linear(start: float, end: float, duration: int, t: int) -> float:
    """``start`` at t=0, ``end`` from t=duration on, a straight line in between."""
    if duration <= 0:
        return end
    frac = min(1.0, t / duration)
    return start + frac * (end - start)
