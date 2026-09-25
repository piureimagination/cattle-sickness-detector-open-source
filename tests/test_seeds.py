import os
import random

from src.utils.seeds import lock_seeds


def test_lock_seeds_sets_pythonhashseed_env_var():
    lock_seeds(123)
    assert os.environ["PYTHONHASHSEED"] == "123"


def test_lock_seeds_makes_random_reproducible():
    lock_seeds(7)
    first = [random.random() for _ in range(5)]

    lock_seeds(7)
    second = [random.random() for _ in range(5)]

    assert first == second


def test_lock_seeds_different_seeds_differ():
    lock_seeds(1)
    a = random.random()

    lock_seeds(2)
    b = random.random()

    assert a != b
