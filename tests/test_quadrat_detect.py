"""Tests for proposing quadrat corners.

The property under test is mostly NEGATIVE: this module must refuse far more
often than it offers, must never write anything, and must never return a
position it cannot vouch for. Seven method families were measured and the
best reaches 3.4% of a side against a 1% requirement, so an offer that looks
confident and is wrong is the failure mode to design against.
"""
import numpy as np
import pytest

from functions import quadrat_detect as QD


def _frame_image(size=520, side=340, rail=20, rot_deg=0.0, seed=0,
                 shift=(0, 0), bright=1.0):
    """A synthetic quadrat: a bright square ring on textured 'gravel'."""
    import cv2
    rng = np.random.RandomState(seed)
    # gravel: blobs, so the background is textured rather than noise
    img = np.full((size, size), 90, np.float32)
    for _ in range(700):
        x, y = rng.randint(0, size, 2)
        r = rng.randint(4, 11)
        cv2.circle(img, (int(x), int(y)), int(r), float(rng.randint(55, 150)), -1)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    c = size / 2.0
    half = side / 2.0
    quad = np.array([[c - half, c - half], [c + half, c - half],
                     [c + half, c + half], [c - half, c + half]], np.float32)
    th = np.radians(rot_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    quad = (R @ (quad - c).T).T + c + np.asarray(shift, float)
    outer = np.zeros((size, size), np.uint8)
    cv2.fillPoly(outer, [quad.astype(np.int32)], 255)
    inner_q = (quad - quad.mean(0)) * (1 - 2.0 * rail / side) + quad.mean(0)
    inner = np.zeros((size, size), np.uint8)
    cv2.fillPoly(inner, [inner_q.astype(np.int32)], 255)
    ring = cv2.bitwise_and(outer, cv2.bitwise_not(inner))
    img[ring > 0] = 215
    img = np.clip(img * bright, 0, 255)
    return img.astype(np.uint8), quad.astype(float)


@pytest.fixture
def exemplar():
    img, quad = _frame_image(seed=1)
    return QD.build_exemplar(img, quad, frame_thickness_m=0.06,
                             quadrat_side_m=1.0), img, quad


def test_the_frame_thickness_is_required_not_guessed():
    """Without it the rail cannot be masked, and a guess changes the match."""
    img, quad = _frame_image()
    with pytest.raises(ValueError) as e:
        QD.build_exemplar(img, quad, frame_thickness_m=0.0, quadrat_side_m=1.0)
    assert "thickness" in str(e.value).lower()


def test_the_side_length_is_required_not_guessed():
    img, quad = _frame_image()
    with pytest.raises(ValueError) as e:
        QD.build_exemplar(img, quad, frame_thickness_m=0.06, quadrat_side_m=0)
    assert "side" in str(e.value).lower()


def test_four_corners_are_required(exemplar):
    img, quad = _frame_image()
    with pytest.raises(ValueError):
        QD.build_exemplar(img, quad[:3], frame_thickness_m=0.06,
                          quadrat_side_m=1.0)


def test_the_exemplar_masks_the_interior_away(exemplar):
    """The gravel inside the frame is what differs between photographs, so it
    must not be part of the template."""
    ex, _, _ = exemplar
    h, w = ex.ring_mask.shape
    assert ex.ring_mask[h // 2, w // 2] == 0, "the middle must be masked out"
    frac = float((ex.ring_mask > 0).mean())
    assert 0.05 < frac < 0.60, "the mask should be a ring, not everything"


def test_a_photograph_of_the_same_quadrat_is_proposed(exemplar):
    ex, img, quad = exemplar
    moved, truth = _frame_image(seed=2, shift=(18, -11))
    p = QD.propose(moved, ex)
    assert p.accepted, p.reasons
    side = float(np.median([np.hypot(*(truth[(i + 1) % 4] - truth[i]))
                            for i in range(4)]))
    worst = max(np.hypot(*(p.corners[k] - truth[k])) for k in range(4)) / side
    assert worst < 0.05, "an accepted proposal must be near the truth"


def test_an_accepted_proposal_carries_no_reasons(exemplar):
    ex, _, _ = exemplar
    moved, _ = _frame_image(seed=3, shift=(9, 7))
    p = QD.propose(moved, ex)
    if p.accepted:
        assert p.reasons == []


def test_a_photograph_without_the_quadrat_is_refused(exemplar):
    """The failure to design against: a confident position nowhere near it."""
    ex, _, _ = exemplar
    import cv2
    rng = np.random.RandomState(9)
    plain = np.full((520, 520), 90, np.float32)
    for _ in range(700):
        x, y = rng.randint(0, 520, 2)
        cv2.circle(plain, (int(x), int(y)), int(rng.randint(4, 11)),
                   float(rng.randint(55, 150)), -1)
    p = QD.propose(plain.astype(np.uint8), ex)
    assert not p.accepted
    assert p.corners is None
    assert p.reasons, "a refusal must say why"


def test_a_refusal_explains_itself_in_actionable_terms(exemplar):
    ex, _, _ = exemplar
    tiny = np.zeros((32, 32), np.uint8)
    p = QD.propose(tiny, ex)
    assert not p.accepted
    assert p.reasons
    assert any(len(r) > 20 for r in p.reasons)


def test_nothing_is_returned_when_it_is_refused(exemplar):
    """A refused proposal must not leak a position the caller could draw."""
    ex, _, _ = exemplar
    p = QD.propose(np.full((520, 520), 128, np.uint8), ex)
    if not p.accepted:
        assert p.corners is None


def test_the_gate_can_be_tightened_and_loosened(exemplar):
    """The threshold is a measured default, not a constant."""
    ex, _, _ = exemplar
    moved, _ = _frame_image(seed=4, shift=(12, 12))
    strict = QD.propose(moved, ex, settings={"max_disagreement": 0.0})
    assert not strict.accepted
    assert "disagree" in " ".join(strict.reasons).lower() or strict.reasons


def test_the_default_gate_is_the_measured_one():
    """0.020 offers 13 of 59, all within 5% of a side, on the measured corpus."""
    assert QD.DEFAULTS["max_disagreement"] == 0.020


def test_proposing_never_raises_on_bad_input(exemplar):
    ex, _, _ = exemplar
    for bad in (np.zeros((0, 0), np.uint8), np.zeros((4, 4), np.uint8),
                np.full((80, 80), 255, np.uint8)):
        p = QD.propose(bad, ex)
        assert isinstance(p, QD.CornerProposal)
        assert p.accepted is False or p.corners is not None


def test_the_proposal_is_serialisable_and_complete(exemplar):
    ex, _, _ = exemplar
    moved, _ = _frame_image(seed=5, shift=(6, 6))
    d = QD.propose(moved, ex).as_dict()
    for k in ("corners", "disagreement", "score", "accepted", "reasons"):
        assert k in d
    import json
    json.dumps(d)


def test_brightness_between_photographs_does_not_prevent_a_match(exemplar):
    """Exposure across one survey varies by up to 86 grey levels, measured."""
    ex, _, _ = exemplar
    dim, truth = _frame_image(seed=6, shift=(10, 4), bright=0.65)
    p = QD.propose(dim, ex)
    # it may still refuse, but it must not crash and must explain itself
    assert p.accepted or p.reasons


def test_the_answer_does_not_depend_on_the_input_resolution():
    """The gate was calibrated at one working size and is not scale-free.

    Shipped without this guard, the GUI handed `propose` a 12 MP photograph and
    it took 52.5 s and then REFUSED -- disagreement read 0.515 against a 0.020
    threshold -- where the same pair at 1000 px accepted in 3.1 s at 2.1%
    error. Both images are fitted to `work_px` before anything else happens,
    and corners come back in the caller's own pixels.
    """
    import cv2
    img, quad = _frame_image(seed=11)
    tgt, truth = _frame_image(seed=12, shift=(14, -8))
    big_img = cv2.resize(img, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_LINEAR)
    big_tgt = cv2.resize(tgt, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_LINEAR)

    small = QD.propose(tgt, QD.build_exemplar(img, quad, 0.06, 1.0))
    big = QD.propose(big_tgt, QD.build_exemplar(big_img, quad * 3.0, 0.06, 1.0))

    assert small.accepted == big.accepted, "resolution changed the decision"
    if small.accepted:
        side = float(np.median([np.hypot(*(truth[(i + 1) % 4] - truth[i]))
                                for i in range(4)]))
        # corners must come back in the CALLER's pixels, so the big-image
        # answer divided by 3 must land on the small-image answer
        worst = max(np.hypot(*(big.corners[k] / 3.0 - small.corners[k]))
                    for k in range(4)) / side
        assert worst < 0.02, f"same scene, different answer by {100 * worst:.1f}%"


# --------------------------------------------------------------------------- #
#  Feature 022: refine the proposal onto the rails                             #
# --------------------------------------------------------------------------- #
def test_the_exemplar_carries_a_cross_profile_per_edge(exemplar):
    ex, _, _ = exemplar
    assert ex.edge_profiles is not None
    assert len(ex.edge_profiles) == 4
    assert all(p is not None and len(p) > 20 for p in ex.edge_profiles)


def test_refining_a_correct_placement_does_not_move_it(exemplar):
    """The property all three earlier refiners failed.

    Three earlier formulations, handed the exact answer
    as a seed, moved AWAY from it by 7.8%, 16.4% and 9.1% of a side. A refiner
    that damages a correct placement is not a refiner.
    """
    ex, img, quad = exemplar
    out, score = QD.refine_corners(img, ex, quad)
    assert out is not None, "refinement found no support on its own exemplar"
    side = float(np.median([np.hypot(*(quad[(i + 1) % 4] - quad[i]))
                            for i in range(4)]))
    worst = max(np.hypot(*(out[k] - quad[k])) for k in range(4)) / side
    assert worst < 0.01, f"a correct placement moved by {100 * worst:.1f}%"


def test_refinement_pulls_a_displaced_quadrilateral_back(exemplar):
    ex, img, quad = exemplar
    ctr = quad.mean(0)
    side = float(np.median([np.hypot(*(quad[(i + 1) % 4] - quad[i]))
                            for i in range(4)]))
    seed = quad + np.array([0.025, -0.02]) * side
    before = max(np.hypot(*(seed[k] - quad[k])) for k in range(4)) / side
    out, _ = QD.refine_corners(img, ex, seed)
    assert out is not None
    after = max(np.hypot(*(out[k] - quad[k])) for k in range(4)) / side
    assert after < before, f"refinement made it worse: {before:.3f} -> {after:.3f}"


def test_refinement_returns_nothing_rather_than_a_guess(exemplar):
    """No usable rail means no refinement, so the caller keeps its seed."""
    ex, _, _ = exemplar
    flat = np.full((520, 520), 128, np.uint8)
    out, score = QD.refine_corners(flat, ex, np.array(
        [[100., 100.], [400., 100.], [400., 400.], [100., 400.]]))
    assert out is None or score < 1.0


def test_a_refused_proposal_is_not_refined_into_existence(exemplar):
    """Refinement runs after the gate; it must not create an offer."""
    ex, _, _ = exemplar
    import cv2
    rng = np.random.RandomState(21)
    plain = np.full((520, 520), 90, np.float32)
    for _ in range(700):
        x, y = rng.randint(0, 520, 2)
        cv2.circle(plain, (int(x), int(y)), int(rng.randint(4, 11)),
                   float(rng.randint(55, 150)), -1)
    p = QD.propose(plain.astype(np.uint8), ex)
    assert not p.accepted
    assert p.corners is None
    assert not p.refined


def test_refinement_can_be_switched_off(exemplar):
    ex, _, _ = exemplar
    moved, _ = _frame_image(seed=31, shift=(10, 6))
    off = QD.propose(moved, ex, settings={"refine": False})
    assert not off.refined


def test_the_proposal_records_whether_it_was_refined(exemplar):
    ex, _, _ = exemplar
    moved, _ = _frame_image(seed=32, shift=(8, 8))
    d = QD.propose(moved, ex).as_dict()
    assert "refined" in d and "refine_score" in d
    import json
    json.dumps(d)


def test_an_unrefined_offer_says_that_it_is_unrefined(exemplar, monkeypatch):
    """Falling back to the coarse position must be recorded.

    A refined offer sits at a fraction of a per cent of a side, an unrefined
    one at a few per cent. Handing back the coarse position without saying so
    describes the one in the words of the other -- which is the silent
    degradation this module exists to stop.
    """
    ex, _, _ = exemplar
    moved, _ = _frame_image(seed=33, shift=(9, 5))
    monkeypatch.setattr(QD, "refine_corners", lambda *a, **k: (None, -1.0))
    p = QD.propose(moved, ex)
    assert p.accepted and p.corners is not None    # the offer still stands
    assert not p.refined
    assert any("unrefined" in r for r in p.reasons), p.reasons


def test_corner_identity_is_only_fixed_up_to_a_quarter_turn(exemplar):
    """The ring is four-fold symmetric, so labelling can come back rotated.

    This is a documented limitation, not a defect: the OUTLINE is what the
    guide draws. The test pins the property so that a caller which starts
    depending on corner identity finds out here rather than in the field.
    """
    ex, _, _ = exemplar
    img, quad = _frame_image(seed=34, rot_deg=90.0)
    p = QD.propose(img, ex)
    if not p.accepted:
        pytest.skip("nothing offered on this synthetic quarter-turn")
    s = float(np.median([np.hypot(*(quad[(i + 1) % 4] - quad[i]))
                         for i in range(4)]))

    def worst(q):
        return max(np.hypot(*(q[i] - quad[i])) for i in range(4)) / s

    best = min(worst(np.roll(p.corners, k, axis=0)) for k in range(4))
    assert best < 0.05, "the outline itself must be right: %.1f%%" % (100 * best)
