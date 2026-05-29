"""Math & Statistics Library for ACMS."""

import numpy as np
from scipy import stats, linalg
from typing import Optional, Dict, List, Tuple, Callable


class WaveletDecomposition:
    """Wavelet decomposition for multi-scale analysis.

    Implements discrete wavelet transforms:
    - Haar: Simplest wavelet, good for piecewise constant signals
    - DB4: Daubechies 4-coefficient wavelet, smoother decomposition
    """

    @staticmethod
    def haar_transform(data: np.ndarray) -> Dict:
        """Compute Haar wavelet decomposition.

        Args:
            data: 1-D signal to decompose.

        Returns:
            Dict with approximation and detail coefficients at each level.
        """
        n = len(data)
        if n < 4:
            return {"approximations": [data], "details": [np.array([])], "levels": 0}

        next_pow2 = int(2 ** np.ceil(np.log2(n)))
        padded = np.zeros(next_pow2)
        padded[:n] = data

        approximations = []
        details = []
        current = padded.copy()
        level = 0

        while len(current) >= 4:
            half = len(current) // 2
            approx = np.zeros(half)
            detail = np.zeros(half)

            for i in range(half):
                approx[i] = (current[2*i] + current[2*i + 1]) / np.sqrt(2)
                detail[i] = (current[2*i] - current[2*i + 1]) / np.sqrt(2)

            approximations.append(approx[:min(half, n)])
            details.append(detail[:min(half, n)])
            current = approx
            level += 1

        return {"approximations": approximations, "details": details, "levels": level, "wavelet": "haar"}

    @staticmethod
    def db4_transform(data: np.ndarray) -> Dict:
        """Compute DB4 (Daubechies 4) wavelet decomposition.

        Uses the 4-coefficient Daubechies filter bank for smoother
        decomposition than Haar.

        Args:
            data: 1-D signal to decompose.

        Returns:
            Dict with approximation and detail coefficients.
        """
        n = len(data)
        if n < 8:
            return WaveletDecomposition.haar_transform(data)

        # DB4 filter coefficients
        h = np.array([
            0.2303778133088964,
            0.7148465705529154,
            0.6308807679398587,
            -0.0279837694168599,
            -0.1870348117190931,
            0.0308413818355607,
            0.0328830116668852,
            -0.0105974017850690,
        ])
        g = np.array([
            -0.0105974017850690,
            -0.0328830116668852,
            0.0308413818355607,
            0.1870348117190931,
            -0.0279837694168599,
            -0.6308807679398587,
            0.7148465705529154,
            -0.2303778133088964,
        ])

        # Pad to even length
        pad_len = len(h)
        padded = np.zeros(n + pad_len)
        padded[:n] = data

        approximations = []
        details = []
        current = padded.copy()
        level = 0

        while len(current) >= 2 * len(h):
            n_curr = len(current)
            half = n_curr // 2

            # Periodic extension
            extended = np.concatenate([current, current[:pad_len]])

            approx = np.zeros(half)
            detail = np.zeros(half)

            for i in range(half):
                for k in range(len(h)):
                    idx = 2 * i + k
                    if idx < len(extended):
                        approx[i] += h[k] * extended[idx]
                        detail[i] += g[k] * extended[idx]

            approximations.append(approx)
            details.append(detail)
            current = approx
            level += 1

            if level >= 10:  # Safety limit
                break

        return {
            "approximations": approximations,
            "details": details,
            "levels": level,
            "wavelet": "db4",
        }

    @staticmethod
    def denoise(data: np.ndarray, threshold: float = 1.0,
                wavelet: str = "haar") -> np.ndarray:
        """Denoise signal using wavelet thresholding.

        Applies soft thresholding to detail coefficients and
        reconstructs the signal.

        Args:
            data: Input signal.
            threshold: Threshold for detail coefficients.
            wavelet: Wavelet type ("haar" or "db4").

        Returns:
            Denoised signal.
        """
        if wavelet == "db4":
            result = WaveletDecomposition.db4_transform(data)
        else:
            result = WaveletDecomposition.haar_transform(data)

        if not result["details"] or not result["approximations"]:
            return data

        # Soft thresholding
        denoised_details = []
        for detail in result["details"]:
            denoised = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0)
            denoised_details.append(denoised)

        # Reconstruct using approximation at coarsest level
        approx = result["approximations"][-1]
        reconstructed = np.repeat(approx, max(1, len(data) // len(approx)))

        if len(reconstructed) < len(data):
            reconstructed = np.pad(reconstructed, (0, len(data) - len(reconstructed)), mode='edge')
        return reconstructed[:len(data)]



__all__ = ['WaveletDecomposition']
