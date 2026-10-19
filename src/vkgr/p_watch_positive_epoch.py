"""Deterministic P watch-positive epoch mixing and expected logQ frequencies."""
import random

from .p_watch_positive import logq_frequency_table

WATCH_EPOCH_SEED_BASE = 20261020


def epoch_tokens(n_like, n_aux, epoch_index):
    """Return the shuffled (source, row-index) epoch sequence; epoch_index is zero-based."""
    if type(n_like) is not int or n_like < 1 or type(n_aux) is not int or n_aux < 1:
        raise ValueError("nonempty like and auxiliary pools required")
    if type(epoch_index) is not int or epoch_index < 0:
        raise ValueError("invalid zero-based epoch index")
    count = min(n_like, n_aux)
    rng = random.Random(WATCH_EPOCH_SEED_BASE + epoch_index + 1)
    auxiliary = rng.sample(range(n_aux), count)
    tokens = [("like", index) for index in range(n_like)] + [("aux", index) for index in auxiliary]
    rng.shuffle(tokens)
    return tokens


def epoch_rows(like_rows, auxiliary_rows, tokens):
    result = []
    for token in tokens:
        if type(token) is not tuple or len(token) != 2:
            raise ValueError("invalid mixed-row token")
        source, index = token
        rows = like_rows if source == "like" else auxiliary_rows if source == "aux" else None
        if rows is None or type(index) is not int or not 0 <= index < len(rows):
            raise ValueError("mixed-row token outside source pool")
        result.append(rows[index])
    return result


def frequency_table(like_rows, auxiliary_rows):
    if any(type(row.get("target")) is not int for row in like_rows + auxiliary_rows):
        raise ValueError("invalid mixed target row")
    return logq_frequency_table([row["target"] for row in like_rows],
                                [row["target"] for row in auxiliary_rows])
