import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple

import numpy as np


def set_global_seed(seed: int) -> None:
	"""
	Set seeds for Python, NumPy, and ensure deterministic behavior where possible.

	>>> set_global_seed(12345)
	>>> rng_vals = np.random.RandomState(12345).rand(3)
	>>> (len(rng_vals), round(float(rng_vals[0]), 6))
	(3, 0.929616)
	"""
	random.seed(seed)
	np.random.seed(seed)
	try:
		import torch  # type: ignore

		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(seed)
	except Exception:
		pass


def read_jsonl(path: str | Path) -> List[dict]:
	"""
	Read a JSONL file into a list of dicts.

	>>> from tempfile import NamedTemporaryFile
	>>> tmp = NamedTemporaryFile(delete=False, mode="w", suffix=".jsonl")
	>>> _ = tmp.write('{"a": 1}\n{"b": 2}\n')
	>>> tmp.close()
	>>> items = read_jsonl(tmp.name)
	>>> [list(x.items())[0] for x in items]
	[('a', 1), ('b', 2)]
	"""
	path = Path(path)
	rows: List[dict] = []
	with path.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			rows.append(json.loads(line))
	return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
	"""
	Write an iterable of dict rows to JSONL.

	>>> from tempfile import NamedTemporaryFile
	>>> tmp = NamedTemporaryFile(delete=False, mode="w", suffix=".jsonl")
	>>> tmp.close()
	>>> write_jsonl(tmp.name, [{"x": 1}, {"y": 2}])
	>>> read_jsonl(tmp.name)
	[{'x': 1}, {'y': 2}]
	"""
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as f:
		for row in rows:
			f.write(json.dumps(row, ensure_ascii=False) + "\n")


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
	"""
	Numerically stable softmax.

	>>> x = np.array([0.0, 1.0, 2.0])
	>>> np.round(softmax(x), 6).tolist()
	[0.090031, 0.244728, 0.665241]
	"""
	shifted = logits - np.max(logits, axis=axis, keepdims=True)
	exp = np.exp(shifted)
	den = np.sum(exp, axis=axis, keepdims=True)
	return exp / den


def kl_divergence(p: np.ndarray, q: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
	"""
	Compute KL(p || q) along a given axis. p and q must be probability distributions.

	>>> p = np.array([0.1, 0.2, 0.7])
	>>> q = np.array([0.2, 0.2, 0.6])
	>>> float(np.round(kl_divergence(p, q), 6))
	0.038591
	"""
	p = np.clip(p, eps, 1.0)
	q = np.clip(q, eps, 1.0)
	p = p / np.sum(p, axis=axis, keepdims=True)
	q = q / np.sum(q, axis=axis, keepdims=True)
	kl = np.sum(p * (np.log(p) - np.log(q)), axis=axis)
	return kl


def top_k_overlap(p: np.ndarray, q: np.ndarray, k: int = 10) -> float:
	"""
	Compute the top-k index overlap between two distributions over the last axis.

	>>> p = np.array([0.1, 0.8, 0.05, 0.05])
	>>> q = np.array([0.7, 0.1, 0.1, 0.1])
	>>> top_k_overlap(p, q, k=2)
	0.5
	"""
	if p.ndim != 1 or q.ndim != 1:
		raise ValueError("p and q must be 1D distributions")
	if p.shape[0] != q.shape[0]:
		raise ValueError("p and q must have same length")
	k = min(k, p.shape[0])
	idx_p = np.argsort(-p)[:k]
	idx_q = np.argsort(-q)[:k]
	return float(len(set(idx_p.tolist()) & set(idx_q.tolist())) / k)


def bootstrap_mean_confidence_interval(values: Sequence[float], num_bootstrap: int = 1000, alpha: float = 0.05, seed: int | None = 0) -> Tuple[float, float, float]:
	"""
	Bootstrap the mean and return (mean, ci_lower, ci_upper).

	>>> vals = [0.0, 1.0, 2.0, 3.0]
	>>> m, lo, hi = bootstrap_mean_confidence_interval(vals, num_bootstrap=200, alpha=0.1, seed=0)
	>>> round(m, 6), 0.5 < lo < hi < 2.5
	(1.5, True)
	"""
	arr = np.asarray(values, dtype=float)
	if arr.size == 0:
		raise ValueError("values must be non-empty")
	if seed is not None:
		np.random.seed(seed)
	means = []
	for _ in range(num_bootstrap):
		indices = np.random.randint(0, arr.size, size=arr.size)
		means.append(float(arr[indices].mean()))
	means = np.array(means)
	ci_low = float(np.quantile(means, alpha / 2))
	ci_high = float(np.quantile(means, 1 - alpha / 2))
	return float(arr.mean()), ci_low, ci_high


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Utility demos")
	parser.add_argument("--demo", choices=["softmax", "kl", "topk", "bootstrap"], default="softmax")
	args = parser.parse_args()

	if args.demo == "softmax":
		arr = np.array([0.0, 1.0, 2.0])
		print(np.round(softmax(arr), 6))
	elif args.demo == "kl":
		p = np.array([0.1, 0.2, 0.7])
		q = np.array([0.2, 0.2, 0.6])
		print(round(float(kl_divergence(p, q)), 6))
	elif args.demo == "topk":
		p = np.array([0.1, 0.8, 0.05, 0.05])
		q = np.array([0.7, 0.1, 0.1, 0.1])
		print(top_k_overlap(p, q, k=2))
	elif args.demo == "bootstrap":
		vals = [0.0, 1.0, 2.0, 3.0]
		print(bootstrap_mean_confidence_interval(vals, num_bootstrap=200, alpha=0.1, seed=0))
