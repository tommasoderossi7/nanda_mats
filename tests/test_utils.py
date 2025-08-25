import numpy as np

from src.utils import (
	softmax,
	kl_divergence,
	top_k_overlap,
	bootstrap_mean_confidence_interval,
)


def test_softmax_basic():
	x = np.array([0.0, 1.0, 2.0])
	p = softmax(x)
	assert np.allclose(p.sum(), 1.0)
	assert np.allclose(np.round(p, 6), np.array([0.090031, 0.244728, 0.665241]))


def test_kl_divergence():
	p = np.array([0.1, 0.2, 0.7])
	q = np.array([0.2, 0.2, 0.6])
	kl = kl_divergence(p, q)
	assert float(np.round(kl, 6)) == 0.038591


def test_top_k_overlap():
	p = np.array([0.1, 0.8, 0.05, 0.05])
	q = np.array([0.7, 0.1, 0.1, 0.1])
	assert top_k_overlap(p, q, k=2) == 1.0


def test_bootstrap_ci():
	vals = [0.0, 1.0, 2.0, 3.0]
	m, lo, hi = bootstrap_mean_confidence_interval(vals, num_bootstrap=200, alpha=0.1, seed=0)
	assert abs(m - 1.5) < 1e-8
	assert lo < m < hi
