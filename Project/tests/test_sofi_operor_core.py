import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import sofi_operor_core as core


# 모든 테스트 전에 글로벌 상태 초기화
def setup_function(function):
    core.GLOBAL_TRACE_LOG.entries.clear()
    core.PREV_PHASE_STATE = None


def test_call_llm_echo_structure():
    """call_llm이 에러 없이 문자열을 반환하고,
    user_prompt의 일부를 포함하는지 확인한다.
    (외부 API는 아직 연결되지 않았으므로 에코 구조만 검증)
    """
    msg = "테스트 프롬프트입니다.\n두 번째 줄."
    out = core.call_llm("system prompt", msg)
    assert isinstance(out, str)
    assert "테스트 프롬프트입니다." in out


def test_agent_step_basic_flow():
    """agent_step이 기본 입력에서 정상적으로 문자열 응답을 생성하고
    TraceLog에 1개의 엔트리가 쌓이는지 검증한다.
    """
    reply = core.agent_step(
        "Sofience–Operor 구조 테스트입니다.",
        env_state={"need_level": 0.7, "supply_level": 0.3},
    )

    # 1) 응답 타입 확인
    assert isinstance(reply, str)
    assert "채널" in reply or "Sofience–Operor" in reply

    # 2) TraceLog가 1개 엔트리를 가지는지
    assert len(core.GLOBAL_TRACE_LOG.entries) == 1
    entry = core.GLOBAL_TRACE_LOG.entries[0]

    # 3) TraceEntry 필드들이 기본적인 형태를 갖추는지
    assert isinstance(entry.turn_id, str)
    assert isinstance(entry.context, dict)
    assert isinstance(entry.goal, dict)
    assert isinstance(entry.result, dict)
    assert isinstance(entry.delta_phi_vec, dict)
    assert set(entry.delta_phi_vec.keys()) == {"semantic", "ethical", "strategic"}


def test_delta_phi_vector_range():
    """compute_delta_phi_vector가 0~1 범위의 값을 반환하는지 확인."""
    prev = core.PhaseState(
        goal_text="이전 목표입니다.",
        plan_id="plan_conservative",
        alignment_score=0.8,
        ethical_risk=0.2,
        channel="main",
    )
    curr = core.PhaseState(
        goal_text="새로운 목표입니다.",
        plan_id="plan_aggressive",
        alignment_score=0.4,
        ethical_risk=0.6,
        channel="main",
    )

    vec = core.compute_delta_phi_vector(prev, curr, goal_prev_text=prev.goal_text)
    assert set(vec.keys()) == {"semantic", "ethical", "strategic"}
    for v in vec.values():
        assert 0.0 <= v <= 1.0


def test_recursive_alignment_search_terminates():
    """recursive_alignment_search가 최대 깊이에서 종료되고,
    ScoredPlan 또는 None을 반환하는지만 확인.
    (무한 루프 방지용 안전망 테스트)
    """
    ctx = core.build_context(
        user_input="테스트용 입력입니다.",
        env_state={},
        trace_log=core.GLOBAL_TRACE_LOG,
    )
    goal = core.compose_goal(ctx)

    result = core.recursive_alignment_search(
        ctx,
        goal,
        depth=0,
        max_depth=1,
    )

    assert (result is None) or isinstance(result, core.ScoredPlan)
