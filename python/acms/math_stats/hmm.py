"""Math & Statistics Library for ACMS."""

import numpy as np
from scipy import stats, linalg
from typing import Optional, Dict, List, Tuple, Callable


class HMM:
    """Hidden Markov Model with Viterbi decoding and BIC selection.

    Implements the Baum-Welch (EM) algorithm for parameter estimation,
    Viterbi algorithm for state decoding, and BIC for model selection.
    """

    def __init__(self, n_states: int = 3):
        """Initialize HMM.

        Args:
            n_states: Number of hidden states.
        """
        self.n_states = n_states
        self.transition: Optional[np.ndarray] = None
        self.emission_params: Optional[List[Tuple[float, float]]] = None
        self.initial_probs: Optional[np.ndarray] = None
        self.log_likelihood: Optional[float] = None

    def fit(self, observations: np.ndarray, max_iter: int = 100, tol: float = 1e-6):
        """Fit HMM using Baum-Welch (EM) algorithm.

        Args:
            observations: 1-D observation sequence.
            max_iter: Maximum EM iterations.
            tol: Log-likelihood convergence tolerance.

        Returns:
            self
        """
        T = len(observations)
        N = self.n_states

        self.transition = np.ones((N, N)) / N + np.random.randn(N, N) * 0.01
        self.transition = np.abs(self.transition)
        self.transition /= self.transition.sum(axis=1, keepdims=True)

        self.initial_probs = np.ones(N) / N
        obs_sorted = np.sort(observations)
        quantiles = np.array_split(obs_sorted, N)
        self.emission_params = [(np.mean(q), np.std(q) + 1e-8) for q in quantiles]

        prev_ll = -np.inf
        ll = prev_ll
        for iteration in range(max_iter):
            alpha, beta, gamma, xi, ll = self._forward_backward(observations)

            self.initial_probs = gamma[0] / (gamma[0].sum() + 1e-300)

            for i in range(N):
                for j in range(N):
                    self.transition[i, j] = xi[:, i, j].sum() / (gamma[:, i].sum() + 1e-300)

            for i in range(N):
                weights = gamma[:, i]
                w_sum = weights.sum()
                if w_sum > 0:
                    new_mean = np.sum(weights * observations) / w_sum
                    new_std = np.sqrt(np.sum(weights * (observations - new_mean) ** 2) / w_sum) + 1e-8
                    self.emission_params[i] = (new_mean, new_std)

            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

        self.log_likelihood = ll
        return self

    def _forward_backward(self, observations: np.ndarray) -> Tuple:
        """Forward-backward algorithm for computing posterior state probabilities."""
        T = len(observations)
        N = self.n_states

        B = np.zeros((T, N))
        for t in range(T):
            for i in range(N):
                mu, sigma = self.emission_params[i]
                B[t, i] = stats.norm.pdf(observations[t], mu, sigma)

        alpha = np.zeros((T, N))
        c = np.zeros(T)

        alpha[0] = self.initial_probs * B[0]
        c[0] = alpha[0].sum()
        if c[0] > 0:
            alpha[0] /= c[0] + 1e-300
        else:
            alpha[0] = np.ones(N) / N
            c[0] = 1.0

        for t in range(1, T):
            for j in range(N):
                alpha[t, j] = B[t, j] * np.sum(alpha[t - 1] * self.transition[:, j])
            c[t] = alpha[t].sum()
            if c[t] > 0:
                alpha[t] /= c[t] + 1e-300
            else:
                alpha[t] = np.ones(N) / N
                c[t] = 1.0

        beta = np.zeros((T, N))
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            for i in range(N):
                beta[t, i] = np.sum(self.transition[i, :] * B[t + 1, :] * beta[t + 1, :])
            if c[t + 1] > 0:
                beta[t] /= c[t + 1] + 1e-300

        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

        xi = np.zeros((T - 1, N, N))
        for t in range(T - 1):
            for i in range(N):
                for j in range(N):
                    xi[t, i, j] = alpha[t, i] * self.transition[i, j] * B[t + 1, j] * beta[t + 1, j]
            xi[t] /= xi[t].sum() + 1e-300

        ll = np.sum(np.log(c + 1e-300))
        return alpha, beta, gamma, xi, ll

    def viterbi(self, observations: np.ndarray) -> np.ndarray:
        """Viterbi algorithm for most likely state sequence.

        Args:
            observations: 1-D observation sequence.

        Returns:
            Array of most likely states.
        """
        T = len(observations)
        N = self.n_states

        B = np.zeros((T, N))
        for t in range(T):
            for i in range(N):
                mu, sigma = self.emission_params[i]
                B[t, i] = stats.norm.pdf(observations[t], mu, sigma)

        V = np.zeros((T, N))
        backpointer = np.zeros((T, N), dtype=int)
        V[0] = np.log(self.initial_probs + 1e-300) + np.log(B[0] + 1e-300)

        for t in range(1, T):
            for j in range(N):
                prob = V[t - 1] + np.log(self.transition[:, j] + 1e-300)
                backpointer[t, j] = np.argmax(prob)
                V[t, j] = prob[backpointer[t, j]] + np.log(B[t, j] + 1e-300)

        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(V[-1])
        for t in range(T - 2, -1, -1):
            states[t] = backpointer[t + 1, states[t + 1]]
        return states

    @staticmethod
    def select_n_states(observations: np.ndarray, max_states: int = 6) -> Dict:
        """Select optimal number of states using BIC.

        BIC = -2 * log_likelihood + k * ln(T)

        Args:
            observations: 1-D observation sequence.
            max_states: Maximum number of states to test.

        Returns:
            Dict with optimal n_states and BIC values.
        """
        T = len(observations)
        bic_values = {}

        for n in range(2, max_states + 1):
            hmm = HMM(n_states=n)
            hmm.fit(observations)
            k = n * (n - 1) + n * 2 + (n - 1)
            bic = -2 * hmm.log_likelihood + k * np.log(T)
            bic_values[n] = bic

        best_n = min(bic_values, key=bic_values.get)
        return {"optimal_n_states": best_n, "bic_values": bic_values}


class RegimeDetection:
    """Regime detection using HMM with BIC/AIC model selection.

    Provides a high-level interface for detecting market regimes
    with automatic model selection.
    """

    @staticmethod
    def detect(returns: np.ndarray, max_states: int = 6,
               method: str = "bic") -> Dict:
        """Detect market regimes with automatic state selection.

        Args:
            returns: Return series.
            max_states: Maximum number of regimes to test.
            method: Model selection criterion ("bic" or "aic").

        Returns:
            Dict with optimal model, states, and model selection results.
        """
        if len(returns) < 100:
            return {"states": np.zeros(len(returns), dtype=int),
                    "optimal_n_states": 1, "model": None}

        best_score = float('inf')
        best_n = 2
        best_hmm = None
        scores = {}

        for n in range(2, max_states + 1):
            hmm = HMM(n_states=n)
            hmm.fit(returns)

            k = n * (n - 1) + n * 2 + (n - 1)
            T = len(returns)

            if method == "aic":
                score = -2 * hmm.log_likelihood + 2 * k
            else:  # BIC
                score = -2 * hmm.log_likelihood + k * np.log(T)

            scores[n] = score
            if score < best_score:
                best_score = score
                best_n = n
                best_hmm = hmm

        states = best_hmm.viterbi(returns) if best_hmm is not None else np.zeros(len(returns), dtype=int)

        return {
            "states": states,
            "optimal_n_states": best_n,
            "model": best_hmm,
            "scores": scores,
            "method": method,
        }

__all__ = ['HMM', 'RegimeDetection']
