"""Math & Statistics Library for ACMS.

Implements:
- Black-Scholes option pricing with Greeks
- Almgren-Chriss optimal execution
- Hidden Markov Model with BIC/AIC model selection (RegimeDetection)
- Cointegration tests (Engle-Granger, Johansen)
- K-means clustering
- PCA
- GARCH(1,1) volatility model with MLE, forecasting, standardized residuals
- Kalman Filter for dynamic state estimation, hedge ratio, trend extraction
- Variance Ratio Test (Lo-MacKinlay) with multiple holding periods
- Phillips-Perron unit root test with Newey-West standard errors
- Copula models: Gaussian copula, Student-t copula
- Wavelet decomposition: Haar and DB4 wavelets
- Bootstrap methods: percentile, BCa, block bootstrap
- Granger causality test with lag selection
- Hurst exponent: R/S analysis with Anis-Lloyd correction, bootstrapped CI
- Statistical utilities: autocorrelation, partial autocorrelation, Jarque-Bera, ADF
"""

from acms.math_stats.black_scholes import BlackScholes
from acms.math_stats.execution import AlmgrenChriss
from acms.math_stats.hurst import HurstExponent
from acms.math_stats.garch import GARCH11
from acms.math_stats.kalman import KalmanFilter
from acms.math_stats.hmm import HMM, RegimeDetection
from acms.math_stats.tests import VarianceRatioTest, PhillipsPerronTest
from acms.math_stats.copula import GaussianCopula, StudentTCopula
from acms.math_stats.wavelet import WaveletDecomposition
from acms.math_stats.bootstrap import Bootstrap
from acms.math_stats.clustering import KMeansClustering, PCA, autocorrelation, partial_autocorrelation, jarque_bera, augmented_dickey_fuller
from acms.math_stats.statistical_tests import GrangerCausalityTest, EngleGranger, Johansen

__all__ = [
    "BlackScholes", "AlmgrenChriss", "HurstExponent", "GARCH11",
    "KalmanFilter", "HMM", "RegimeDetection", "VarianceRatioTest",
    "PhillipsPerronTest", "GaussianCopula", "StudentTCopula",
    "WaveletDecomposition", "Bootstrap", "GrangerCausalityTest",
    "EngleGranger", "Johansen", "KMeansClustering", "PCA",
    "autocorrelation", "partial_autocorrelation", "jarque_bera",
    "augmented_dickey_fuller",
]
