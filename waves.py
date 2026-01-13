import numpy as np


def sinusoid(N, amp, freq, phase):
    t = np.arange(N)
    theta = 2.0 * np.pi * freq * t + phase
    return amp * np.sin(theta)


def composite(N, components):
    signal = np.zeros(N)
    for amp, freq, phase in components:
        signal += sinusoid(N, amp, freq, phase)  # superposition principle
    return signal


def find_peaks(freqs, amps, phases, threshold=0.1):
    max_amp = np.max(amps)
    mask = amps > threshold * max_amp
    peak_amps = amps[mask]
    peak_freqs = freqs[mask]
    peak_phases = phases[mask]

    order = np.argsort(peak_amps)[::-1]

    peaks = []
    for i in order:
        peaks.append((peak_amps[i], peak_freqs[i], peak_phases[i]))

    return peaks
