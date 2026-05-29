"""Hierarchical Risk Parity portfolio optimization."""

from typing import List

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


class HierarchicalRiskParity:
    """Hierarchical Risk Parity (HRP) - Lopez de Prado algorithm.

    Addresses the instability of Markowitz optimization by:
    1. Clustering assets by correlation
    2. Allocating capital top-down through the dendrogram
    3. Using inverse-variance weighting within clusters
    """

    def optimize(self, returns_matrix: np.ndarray) -> dict:
        """Compute HRP portfolio allocation."""
        if returns_matrix.shape[1] < 2:
            return {"weights": np.array([1.0]), "linkage": None}

        corr = np.corrcoef(returns_matrix.T)
        cov = np.cov(returns_matrix.T)
        n = corr.shape[0]

        corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
        np.fill_diagonal(corr, 1.0)

        dist = squareform(1 - np.abs(corr), checks=False)
        if len(dist) == 0 or np.any(np.isnan(dist)):
            return {"weights": np.ones(n) / n, "linkage": None}

        link = linkage(dist, method='ward')

        sort_ix = self._get_quasi_diag(link)
        sort_ix = [i for i in sort_ix if i < n]

        weights = self._recursive_bisection(cov, sort_ix)
        return {"weights": weights, "linkage": link}

    @staticmethod
    def _get_quasi_diag(link: np.ndarray) -> List[int]:
        """Extract sorted list of original items from linkage matrix."""
        n = link.shape[0] + 1
        sort_ix = [int(link[-1, 0]), int(link[-1, 1])]

        max_id = 2 * n - 2
        while max(sort_ix) >= n:
            new_sort = []
            for i in sort_ix:
                if int(i) >= n:
                    idx = int(i) - n
                    if idx < link.shape[0]:
                        new_sort.append(int(link[idx, 0]))
                        new_sort.append(int(link[idx, 1]))
                    else:
                        new_sort.append(i)
                else:
                    new_sort.append(i)
            sort_ix = new_sort
            if max(sort_ix) < n:
                break
        return [int(i) for i in sort_ix]

    @staticmethod
    def _recursive_bisection(cov: np.ndarray, sort_ix: List[int]) -> np.ndarray:
        """Allocate weights through recursive bisection."""
        n = len(sort_ix)
        weights = np.ones(n)
        clusters = [sort_ix]

        while clusters:
            new_clusters = []
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue
                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]

                cov_left = cov[np.ix_(left, left)]
                var_left = np.diag(cov_left)
                inv_var_left = 1.0 / (var_left + 1e-10)
                w_left = inv_var_left / np.sum(inv_var_left)
                v_left = w_left @ cov_left @ w_left

                cov_right = cov[np.ix_(right, right)]
                var_right = np.diag(cov_right)
                inv_var_right = 1.0 / (var_right + 1e-10)
                w_right = inv_var_right / np.sum(inv_var_right)
                v_right = w_right @ cov_right @ w_right

                alpha = 1.0 - v_left / (v_left + v_right) if (v_left + v_right) > 0 else 0.5

                for i in left:
                    weights[sort_ix.index(i)] *= alpha
                for i in right:
                    weights[sort_ix.index(i)] *= (1 - alpha)

                if len(left) > 1:
                    new_clusters.append(left)
                if len(right) > 1:
                    new_clusters.append(right)
            clusters = new_clusters

        return weights


__all__ = [
    "HierarchicalRiskParity",
]
