import matplotlib.pyplot as plt
import numpy as np

from waves import sinusoid


def plot_analysis(N, truth, detected, signal):
    t = np.arange(N)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].plot(t, signal, "k-", linewidth=1.5)
    axes[0, 0].set_title("Composite Signal")
    axes[0, 0].set_xlabel("Sample N")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True, alpha=0.3)

    components = []
    for a, f, p in detected:
        components.append(sinusoid(N, a, f, p))
    reconstruct = np.sum(components, axis=0)

    axes[0, 1].plot(t, signal, "k-", alpha=0.6, linewidth=1.5, label="Original")
    axes[0, 1].plot(t, reconstruct, "r--", linewidth=1.5, label="Reconstructed")
    axes[0, 1].set_title("Reconstruction Comparison")
    axes[0, 1].set_xlabel("Sample N")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].legend(fontsize=10, loc="upper right")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title("Detected Components")
    for amp, freq, phase in detected:
        wave = sinusoid(N, amp, freq, phase)
        label = f"A={amp:.2f}, f={freq:.4f}, phi={phase:.2f}"
        axes[1, 0].plot(t, wave, alpha=0.7, linewidth=1.2, label=label)
    axes[1, 0].set_xlabel("Sample N")
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].legend(fontsize=8, loc="upper right")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title("Ground Truth Components")
    for amp, freq, phase in truth:
        wave = sinusoid(N, amp, freq, phase)
        label = f"A={amp:.2f}, f={freq:.4f}, phi={phase:.2f}"
        axes[1, 1].plot(t, wave, alpha=0.7, linewidth=1.2, label=label)
    axes[1, 1].set_xlabel("Sample N")
    axes[1, 1].set_ylabel("Amplitude")
    axes[1, 1].legend(fontsize=8, loc="upper right")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_spectrum(freqs, amps, phases, title="Frequency Spectrum"):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    axes[0].stem(freqs, amps, basefmt=" ")
    axes[0].set_title(f"{title}: Amplitude")
    axes[0].set_xlabel("Normalized Frequency")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    sig_mask = amps > 0.01 * np.max(amps)
    axes[1].stem(freqs[sig_mask], phases[sig_mask], basefmt=" ")
    axes[1].set_title(f"{title}: Phase (significant components)")
    axes[1].set_xlabel("Normalized Frequency")
    axes[1].set_ylabel("Phase in Radians")
    axes[1].set_ylim(-np.pi, np.pi)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
