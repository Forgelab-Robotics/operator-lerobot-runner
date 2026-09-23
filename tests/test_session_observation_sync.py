from types import SimpleNamespace

from lerobot_inference.inference.session_endpoint import LeRobotServeSessionEndpoint
from lerobot_inference.inference.session_runner import LeRobotSessionPolicyRunner


def test_queued_actions_wait_for_complete_new_observation():
    outputs = []
    consumed = []
    node = SimpleNamespace(send_output=lambda name, value: outputs.append((name, value)))
    policy = SimpleNamespace(
        is_observation_needed=lambda: False,
        generate_action=lambda observation, aliases: consumed.append(observation) or [0.0] * 7,
    )
    runner = LeRobotSessionPolicyRunner(
        node, policy=policy, endpoint=LeRobotServeSessionEndpoint(policy_id="test"),
        joint_order=["state"], image_input_id_to_alias={"image/main": "image", "image/second": "image2"},
        build_action=lambda value: SimpleNamespace(to_arrow=lambda: value),
        auto_start=True, require_fresh_observation=True,
    )
    def event(name):
        runner.process_event({"type": "INPUT", "id": name, "value": object()})
    try:
        event("tick")
        event("proprio_state"); event("image/main"); event("tick")
        assert consumed == []
        event("image/second"); event("tick")
        assert len(consumed) == 1
        for _ in range(10): event("tick")
        assert len(consumed) == 1
        event("image/main"); event("image/second"); event("tick")
        assert len(consumed) == 1
        event("proprio_state"); event("tick")
        assert len(consumed) == 2
        runner._stop_session()
        runner.state.phase = "running"
        event("tick")
        assert len(consumed) == 2
    finally:
        runner.process_event({"type": "STOP"})
