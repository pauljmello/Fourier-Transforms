import numpy as np


def get_spectrum(X, N):
    n_bins = N // 2 + 1
    freqs = np.arange(n_bins) / N

    amps = np.abs(X[:n_bins]) / N
    amps[1:n_bins - 1] *= 2.0
    if N % 2 == 0:
        amps[n_bins - 1] = np.abs(X[N // 2]) / N

    theta = np.angle(X[:n_bins])
    phi = theta + np.pi / 2
    phi = np.mod(phi + np.pi, 2 * np.pi) - np.pi

    return freqs, amps, phi


def next_power_of_2(n):
    return 1 << (n - 1).bit_length()


def fft_core(x):
    N = len(x)
    if N <= 1:
        return x.astype(complex)

    even = fft_core(x[0::2])
    odd = fft_core(x[1::2])

    X = np.zeros(N, dtype=complex)
    for k in range(N // 2):
        W_k = np.exp(-2j * np.pi * k / N)  # twiddle factor
        t = W_k * odd[k]
        X[k] = even[k] + t  # Cooley-Tukey butterfly
        X[k + N // 2] = even[k] - t  # Cooley-Tukey butterfly
    return X


def fft(signal):
    N_orig = len(signal)
    N_padded = next_power_of_2(N_orig)

    if N_padded > N_orig:
        signal_padded = np.zeros(N_padded)
        signal_padded[:N_orig] = signal
    else:
        signal_padded = np.asarray(signal, dtype=float)

    X = fft_core(signal_padded)
    return get_spectrum(X, N_padded)


def dft(signal):
    N = len(signal)
    X = np.zeros(N, dtype=complex)

    for k in range(N):
        for n in range(N):
            angle = -2.0 * np.pi * k * n / N
            X[k] += signal[n] * np.exp(1j * angle)  # DFT analysis equation

    return get_spectrum(X, N)


def continuous_ft(signal):
    N = len(signal)
    n_bins = N // 2 + 1
    X = np.zeros(n_bins, dtype=complex)
    n = np.arange(N)

    for k in range(n_bins):
        theta = 2.0 * np.pi * k * n / N
        real_sum = np.sum(signal * np.cos(theta))  # Euler's formula
        imag_sum = np.sum(signal * np.sin(theta))  # Euler's formula
        X[k] = real_sum - 1j * imag_sum

    freqs = np.arange(n_bins) / N
    amps = np.abs(X) / N
    amps[1:n_bins - 1] *= 2.0
    if N % 2 == 0 and n_bins > 1:
        amps[n_bins - 1] = np.abs(X[n_bins - 1]) / N

    phases = np.angle(X) + np.pi / 2
    phases = np.mod(phases + np.pi, 2 * np.pi) - np.pi

    return freqs, amps, phases
