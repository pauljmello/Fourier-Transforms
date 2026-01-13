import numpy as np

from fourier import fft, dft, continuous_ft


def get_config(mode):
    if mode == "simple":
        return simple_config()
    elif mode == "complex":
        return complex_config()
    elif mode == "random":
        return random_config()


def get_transform(mode):
    if mode == "fft":
        return fft
    elif mode == "dft":
        return dft
    elif mode == "continuous":
        return continuous_ft


def simple_config():
    N = 32
    ground_truth = [(1.0, 2 / N, 0.0)]
    return N, ground_truth


def complex_config():
    N = 64
    ground_truth = [(1.0, 2 / N, 0.0), (0.7, 4 / N, np.pi / 2), (0.5, 6 / N, np.pi / 4)]
    return N, ground_truth


def random_config():
    N = np.random.choice([32, 64, 128, 256])
    num_components = np.random.randint(1, 8)
    max_bin = int(0.4 * N)
    used_bins = set()
    ground_truth = []
    for i in range(num_components):
        k = np.random.randint(1, max_bin + 1)
        while k in used_bins:
            k = np.random.randint(1, max_bin + 1)
        used_bins.add(k)
        amp = np.random.uniform(0.1, 1.5)
        freq = k / N
        phase = np.random.uniform(-np.pi, np.pi)
        ground_truth.append((amp, freq, phase))
    return N, ground_truth
