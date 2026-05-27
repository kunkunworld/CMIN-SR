from __future__ import annotations

import numpy as np


def shuffled_surrogate(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64).copy()
    rng.shuffle(y)
    return y


def block_shuffled_surrogate(x: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    blocks = [arr[i : i + block_size] for i in range(0, len(arr), block_size)]
    idx = np.arange(len(blocks))
    rng.shuffle(idx)
    return np.concatenate([blocks[i] for i in idx], axis=0)[: len(arr)]


def phase_randomized_surrogate(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    n = len(arr)
    centered = arr - np.mean(arr)
    fft = np.fft.rfft(centered)
    amps = np.abs(fft)
    phases = np.angle(fft)
    if len(phases) > 2:
        random_phase = rng.uniform(-np.pi, np.pi, size=len(phases) - 2)
        phases[1:-1] = random_phase
    sur = np.fft.irfft(amps * np.exp(1j * phases), n=n)
    sur = (sur - np.mean(sur)) / max(np.std(sur), 1e-8)
    sur = sur * max(np.std(arr), 1e-8) + np.mean(arr)
    return sur.astype(np.float64)
