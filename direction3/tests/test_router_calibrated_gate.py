from paramAgent import (
    _normalized_entropy,
    _top1_top2_margin,
    _calibrated_router_accept,
    _rank_of_index,
    _apply_prompt_dominance_clamp,
    _fallback_blend_alpha,
)


def test_normalized_entropy_peaked_is_low():
    assert _normalized_entropy([1.0, 0.0, 0.0]) == 0.0


def test_normalized_entropy_uniform_is_high():
    ent = _normalized_entropy([1.0, 1.0, 1.0])
    assert 0.99 <= ent <= 1.0


def test_top1_top2_margin():
    assert _top1_top2_margin([0.9, 0.4, 0.2]) == 0.5


def test_calibrated_router_accept_requires_conf_margin_entropy():
    accept, diag = _calibrated_router_accept(
        router_conf=0.30,
        router_mix=[0.34, 0.33, 0.33],
        fused_scores=[0.78, 0.41, 0.2],
        conf_threshold=0.6,
        min_margin=0.05,
        max_entropy=0.95,
    )
    assert accept is True
    assert diag["router_entropy"] > 0.95
    assert diag["top1_top2_margin"] >= 0.05


def test_calibrated_router_rejects_small_margin():
    accept, _ = _calibrated_router_accept(
        router_conf=0.99,
        router_mix=[0.8, 0.1, 0.1],
        fused_scores=[0.51, 0.50, 0.1],
        conf_threshold=0.6,
        min_margin=0.05,
        max_entropy=0.95,
    )
    assert accept is False


def test_rank_of_index_is_one_based():
    assert _rank_of_index([0.8, 0.6, 0.2], 0) == 1
    assert _rank_of_index([0.8, 0.6, 0.2], 2) == 3


def test_prompt_dominance_clamp_enforces_min_prompt_weight():
    w0, w1, w2 = _apply_prompt_dominance_clamp([0.2, 0.4, 0.4], min_prompt_weight=0.55)
    assert w0 >= 0.55
    assert abs((w0 + w1 + w2) - 1.0) < 1e-8


def test_fallback_blend_alpha_for_margin_reject_is_zero():
    alpha = _fallback_blend_alpha(
        router_conf=0.97,
        conf_threshold=0.6,
        rejected=True,
        reject_due_to_margin=True,
    )
    assert alpha == 0.0


def test_fallback_blend_alpha_reject_non_margin_is_soft_capped():
    alpha = _fallback_blend_alpha(
        router_conf=0.97,
        conf_threshold=0.6,
        rejected=True,
        reject_due_to_margin=False,
    )
    assert alpha <= 0.35
