"""
Density balancing T_bal: empirical CDF of one coordinate, pooled over the nodes of the training meshes.

The eCDF is estimated with a histogram on a fixed range [lo, hi], linearly interpolated between the bin edges,
mixed with the uniform CDF (weight eps) so that it is strictly increasing, and extended linearly outside the range.
The output is rescaled to [lo, hi], so T_bal maps the coordinate onto itself while equalising the node density.
"""
import numpy as np


class StrictECDF:
    def __init__(self, lo, hi, n_bins=100_000, eps=1e-6):
        self.lo, self.hi, self.n_bins, self.eps = float(lo), float(hi), n_bins, eps
        self.edges = np.linspace(self.lo, self.hi, n_bins + 1)
        self.counts = np.zeros(n_bins, np.int64)

    def update(self, v):
        """Add the coordinates v of one mesh to the histogram."""
        self.counts += np.histogram(np.clip(v, self.lo, self.hi), bins=self.edges)[0]
        return self

    def finalize(self):
        cdf = np.r_[0.0, np.cumsum(self.counts)] / self.counts.sum()
        self.cdf = (1 - self.eps) * cdf + self.eps * (self.edges - self.lo) / (self.hi - self.lo)
        self.n_empty_bins = int((self.counts == 0).sum())
        return self

    def __call__(self, v):
        v = np.asarray(v, float)
        F = np.interp(v, self.edges, self.cdf)
        below, above = v < self.lo, v > self.hi                        # linear extension with the end slopes
        s0 = (self.cdf[1] - self.cdf[0]) / (self.edges[1] - self.edges[0])
        s1 = (self.cdf[-1] - self.cdf[-2]) / (self.edges[-1] - self.edges[-2])
        F[below] = s0 * (v[below] - self.lo)
        F[above] = 1.0 + s1 * (v[above] - self.hi)
        return self.lo + F * (self.hi - self.lo)

    def state(self):
        return dict(lo=self.lo, hi=self.hi, edges=self.edges, cdf=self.cdf, eps=self.eps, n_empty_bins=self.n_empty_bins)

    @classmethod
    def from_state(cls, s):
        e = cls.__new__(cls)
        e.lo, e.hi, e.edges, e.cdf, e.eps, e.n_empty_bins = s["lo"], s["hi"], s["edges"], s["cdf"], s["eps"], s["n_empty_bins"]
        e.n_bins = len(e.edges) - 1
        return e
