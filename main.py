import os

import numpy as np

from utils import get_transform, get_config
from visualize import plot_analysis, plot_spectrum
from waves import composite, find_peaks, sinusoid

SIGNAL_CONFIG = "random"  # "simple", "complex", "random"

TRANSFORM_CONFIG = "continuous"  # "fft", "dft", "continuous"


def main():
    N, ground_truth = get_config(SIGNAL_CONFIG)
    transform = get_transform(TRANSFORM_CONFIG)

    signal = composite(N, ground_truth)
    freqs, amps, phases = transform(signal)
    detected = find_peaks(freqs, amps, phases)

    print(f"Signal = {SIGNAL_CONFIG}")
    print(f"Transform = {TRANSFORM_CONFIG}\n")
    print(f"N = {N}")

    truth_formatted = []
    for a, f, p in ground_truth:
        truth_formatted.append((round(float(a), 3), round(float(f), 4), round(float(p), 3)))
    print(f"Ground Truth: \t{truth_formatted}")

    detected_formatted = []
    for a, f, p in detected:
        detected_formatted.append((round(float(a), 3), round(float(f), 4), round(float(p), 3)))
    print(f"Observed: \t\t{detected_formatted}\n")

    components = []
    for a, f, p in detected:
        components.append(sinusoid(N, a, f, p))
    reconstruct = np.sum(components, axis=0)

    rmse = np.sqrt(np.mean((signal - reconstruct) ** 2))
    print(f"rmse: {rmse:.2e}")

    os.makedirs(f"results/{TRANSFORM_CONFIG}", exist_ok=True)

    fig = plot_analysis(N, ground_truth, detected, signal)
    fig.savefig(f"results/{TRANSFORM_CONFIG}/analysis.png")

    fig_spectrum = plot_spectrum(freqs, amps, phases)
    fig_spectrum.savefig(f"results/{TRANSFORM_CONFIG}/spectrum.png")

    return signal, ground_truth, detected


if __name__ == "__main__":
    main()
