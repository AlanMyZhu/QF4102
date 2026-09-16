import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def bs_diCall(S0, X, r, T, sigma, H, q=0):
    """Question 1(i)."""
    S0 = np.asarray(S0, dtype=float)
    H = np.asarray(H, dtype=float)

    lambda_ = (r - q + sigma**2 / 2) / sigma**2
    y_ = np.log(H**2 / (S0 * X)) / (sigma * np.sqrt(T)) + lambda_ * sigma * np.sqrt(T)

    c_di = S0 * np.exp(-q * T) * (H / S0) ** (2 * lambda_) * norm.cdf(y_) - X * np.exp(
        -r * T
    ) * (H / S0) ** (2 * lambda_ - 2) * norm.cdf(y_ - sigma * np.sqrt(T))
    return c_di


def bs_vanillaCall(S0, X, r, T, sigma, q=0):
    """Question 1(iii)."""
    S0 = np.asarray(S0, dtype=float)

    d1 = (np.log(S0 / X) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_value = S0 * np.exp(-q * T) * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)
    return call_value


def plot_diCall_vs_S0():
    """Questions 1(ii)-(iii)."""
    S0_values = np.linspace(1.41, 1.70, 30)
    X = 1.4
    r = 0.04
    T = 1
    sigma = 0.2
    H = 1.35
    q = 0.01

    di_call_values = bs_diCall(S0_values, X, r, T, sigma, H, q)
    vanilla_call_values = bs_vanillaCall(S0_values, X, r, T, sigma, q)

    plt.plot(S0_values, di_call_values, label="Down-and-in call")
    plt.plot(S0_values, vanilla_call_values, label="Vanilla call")
    plt.xlabel("Initial asset price, S0")
    plt.ylabel("Option value")
    plt.title("Call Values versus Initial Asset Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_diCall_vs_H():
    """Question 1(iv)."""
    S0 = 1.45
    X = 1.4
    r = 0.04
    T = 1
    sigma = 0.2
    H_values = np.linspace(1.10, 1.40, 31)
    q = 0.01

    di_call_values = bs_diCall(S0, X, r, T, sigma, H_values, q)
    vanilla_call_value = bs_vanillaCall(S0, X, r, T, sigma, q)

    plt.plot(H_values, di_call_values, label="Down-and-in call")
    plt.axhline(vanilla_call_value, color="orange", label="Vanilla call")
    plt.xlabel("Barrier, H")
    plt.ylabel("Option value")
    plt.title("Call Values versus Barrier")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    plot_diCall_vs_S0()
    plot_diCall_vs_H()


if __name__ == "__main__":
    main()
