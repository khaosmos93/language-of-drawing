"""5) Graph representation:  G=(V,E), Laplacian L = D - W; Fiedler vector painted to pixels."""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from skimage.segmentation import slic

from .base import Representation


class GraphRep(Representation):
    name = "graph"
    equation = r"L = D - W,\quad Lv = \lambda v"
    cmap = "twilight"
    n_segments = 300
    sigma = 0.15  # similarity bandwidth

    def compute(self, img: np.ndarray) -> dict:
        H, W = img.shape
        # SLIC needs RGB or 2D image; we feed grayscale (channel_axis=None).
        seg = slic(
            img.astype(np.float32),
            n_segments=self.n_segments,
            compactness=0.1,
            channel_axis=None,
            start_label=0,
        )
        n = int(seg.max()) + 1
        # mean intensity per superpixel
        sums = np.bincount(seg.ravel(), weights=img.ravel().astype(np.float64), minlength=n)
        cnts = np.bincount(seg.ravel(), minlength=n).astype(np.float64)
        mu = sums / np.maximum(cnts, 1)
        # adjacency: pairs of distinct labels that touch horizontally/vertically
        a = seg[:, :-1].ravel(); b = seg[:, 1:].ravel()
        c = seg[:-1, :].ravel(); d = seg[1:, :].ravel()
        ii = np.concatenate([a, c])
        jj = np.concatenate([b, d])
        mask = ii != jj
        ii, jj = ii[mask], jj[mask]
        # undirected unique pairs
        lo = np.minimum(ii, jj); hi = np.maximum(ii, jj)
        pair = lo.astype(np.int64) * n + hi.astype(np.int64)
        pair = np.unique(pair)
        i_e = (pair // n).astype(np.int32)
        j_e = (pair % n).astype(np.int32)
        w_e = np.exp(-((mu[i_e] - mu[j_e]) ** 2) / (2 * self.sigma ** 2))
        # symmetric weight matrix
        rows = np.concatenate([i_e, j_e])
        cols = np.concatenate([j_e, i_e])
        data = np.concatenate([w_e, w_e]).astype(np.float64)
        Wm = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        deg = np.asarray(Wm.sum(axis=1)).ravel()
        D = sparse.diags(deg)
        L = (D - Wm).astype(np.float64)
        # Fiedler = 2nd smallest eigenvector. Use shift-invert near 0; fall back if not converged.
        try:
            vals, vecs = eigsh(L, k=min(3, n - 1), sigma=1e-6, which="LM")
        except Exception:
            vals, vecs = eigsh(L + 1e-6 * sparse.eye(n), k=min(3, n - 1), which="SM")
        order = np.argsort(vals)
        fiedler_idx = order[1] if len(order) > 1 else order[0]
        fiedler = vecs[:, fiedler_idx]
        return {"seg": seg, "fiedler": fiedler.astype(np.float32), "n": n}

    def to_field(self, raw: dict) -> np.ndarray:
        return self._norm01(raw["fiedler"][raw["seg"]])
