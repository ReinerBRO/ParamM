from paramAgent import _select_candidate_index_by_feedback


def test_select_candidate_prefers_passing():
    idx = _select_candidate_index_by_feedback(
        is_passing_list=[False, True, False],
        feedback_list=["Tests failed: assert a", "All tests passed", "Tests failed: assert a assert b"],
    )
    assert idx == 1


def test_select_candidate_prefers_fewer_assert_failures():
    idx = _select_candidate_index_by_feedback(
        is_passing_list=[False, False, False],
        feedback_list=[
            "Tests failed: assert a assert b assert c",
            "Tests failed: assert a",
            "Tests failed: assert a assert b",
        ],
    )
    assert idx == 1
