from paramAgent import _router_shortlist_indices


def test_router_shortlist_prefers_anchor_over_noisy_negative():
    prompt_sims = [0.91, 0.84, 0.83, 0.77, 0.45]
    reflection_sims = [0.05, 0.89, 0.10, 0.20, 0.88]
    negative_penalties = [0.90, 0.05, 0.80, 0.10, 0.05]

    indices = _router_shortlist_indices(
        prompt_sims=prompt_sims,
        reflection_sims=reflection_sims,
        negative_penalties=negative_penalties,
        reflection_available=True,
        shortlist_k=3,
    )

    assert indices == [1, 4, 3]


def test_router_shortlist_returns_full_range_when_k_large():
    indices = _router_shortlist_indices(
        prompt_sims=[0.4, 0.2],
        reflection_sims=[0.3, 0.1],
        negative_penalties=[0.0, 0.0],
        reflection_available=False,
        shortlist_k=8,
    )
    assert indices == [0, 1]
