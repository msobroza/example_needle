"""Tests for :mod:`needle.scoring.strategies`."""

from __future__ import annotations

import numpy as np
import pytest

from needle.scoring import (
    CosineScorer,
    MaxSimScorer,
    ScoringStrategy,
    get_scorer,
    minmax_normalize,
)


def test_maxsim_matches_manual_formula():
    query = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    document = np.array([[1.0, 0.0], [0.0, 2.0], [0.5, 0.5], [3.0, 0.0]])

    scorer = MaxSimScorer()
    expected = float((query @ document.T).max(axis=1).sum())

    assert scorer.score(query, document) == pytest.approx(expected)


def test_cosine_identical_unit_vectors_is_one():
    scorer = CosineScorer()
    vec = np.array([0.6, 0.8])  # unit norm
    assert scorer.score(vec, vec) == pytest.approx(1.0)


def test_cosine_orthogonal_is_zero():
    scorer = CosineScorer()
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert scorer.score(a, b) == pytest.approx(0.0, abs=1e-9)


def test_get_scorer_selects_strategy():
    assert isinstance(get_scorer(True), MaxSimScorer)
    assert isinstance(get_scorer(False), CosineScorer)
    assert isinstance(get_scorer(True), ScoringStrategy)


def test_minmax_normalize_basic():
    out = minmax_normalize(np.array([0.0, 5.0, 10.0]))
    assert out.tolist() == [0.0, 0.5, 1.0]


def test_minmax_normalize_constant_is_zeros():
    out = minmax_normalize(np.array([3.0, 3.0, 3.0]))
    assert out.tolist() == [0.0, 0.0, 0.0]


def test_score_all_returns_array_of_right_length():
    scorer = CosineScorer()
    query = np.array([1.0, 0.0])
    documents = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([1.0, 1.0]),
    ]
    scores = scorer.score_all(query, documents)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (3,)


def test_score_all_maxsim_length():
    scorer = MaxSimScorer()
    query = np.array([[1.0, 0.0], [0.0, 1.0]])
    documents = [
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[0.5, 0.5]]),
    ]
    scores = scorer.score_all(query, documents)
    assert scores.shape == (2,)
