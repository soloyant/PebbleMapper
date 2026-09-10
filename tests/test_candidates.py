"""Rival placements survive the refusal that mentions them.

When a quadrat matches two places about equally well the matcher refuses it --
correctly, because it cannot choose on evidence. It then threw both placements
away, leaving the user a sentence and nothing to act on except running the whole
search again with a better seed.

On a cobble beach every square metre resembles every other, so ambiguity is the
ordinary failure rather than the exotic one. And it is the one failure a person
can usually settle in a second by looking at the ortho: the machine has only
two nearly-equal inlier counts, which is exactly what "ambiguous" means.
"""
from __future__ import annotations

import numpy as np
import pytest

from functions import georef as G


def test_a_quality_carries_candidates_and_defaults_to_none():
    q = G.MatchQuality()
    assert q.candidates == []
    assert q.as_dict()["candidates"] == []


def test_candidates_survive_as_dict():
    q = G.MatchQuality(n_inliers=30)
    q.candidates = [{"matrix": [[1, 0, 5], [0, -1, 6], [0, 0, 1]],
                     "n_inliers": 30, "residual_m": 0.01,
                     "centre": (5.0, 6.0), "from_seed_m": 0.4}]
    d = q.as_dict()
    assert len(d["candidates"]) == 1
    assert d["candidates"][0]["n_inliers"] == 30


def test_the_ambiguity_refusal_hands_back_what_it_found():
    """Read against the source: the branch that refuses for ambiguity must
    populate candidates, or the message inviting the user to look at them is
    a lie."""
    import inspect
    src = inspect.getsource(G.locate_quadrat)
    i = src.index("matched {len(accepted)} separate places")
    block = src[i:i + 2600]
    assert "q.candidates" in block, \
        "the ambiguity refusal no longer returns the candidates"
    assert "rival_matches" in block, \
        "the rivals are being counted but not kept"


def test_a_candidate_states_where_it_is_and_how_good_it_is():
    """A list of transforms is not a choice. To pick one, a person needs its
    position, its inlier count, and how far it sits from the seed."""
    import inspect
    src = inspect.getsource(G.locate_quadrat)
    i = src.index("cands.append")
    block = src[i:i + 700]
    for key in ("matrix", "n_inliers", "residual_m", "centre", "from_seed_m"):
        assert f'"{key}"' in block, f"a candidate does not report {key}"


def test_candidates_are_best_first():
    import inspect
    src = inspect.getsource(G.locate_quadrat)
    assert 'cands.sort(key=lambda c: -c["n_inliers"])' in src


def test_a_survey_result_carries_them_too():
    """They have to reach the table the user is actually looking at."""
    from functions.seeds import QuadratCheck
    c = QuadratCheck("a.jpg", "not_located", "ambiguous", 3, float("nan"),
                     "list", None, (1.0, 2.0),
                     [{"n_inliers": 9, "centre": (1.0, 2.0)}])
    assert c.candidates and c.candidates[0]["n_inliers"] == 9
    assert QuadratCheck("b.jpg", "located").candidates == []
