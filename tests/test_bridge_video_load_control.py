"""v29 real-codec integration and four-arm accounting, no real network."""
from dataclasses import replace
import json

import pytest

from colabcola.bridge_video_controller import ControllerConfig
from colabcola.bridge_video_impairment import LinkConfig
from colabcola.bridge_video_load import LoadControlConfig, LoadRateReceiver
from colabcola.bridge_video_load_control import compare_load_control
from colabcola.bridge_video_rate import RateConfig, RateReceiver, stream_with_rate_control
from colabcola.bridge_video_stream import packet_bytes


ARMS = ("two_rate_recovery", "age_only_recovery", "wire_load_recovery",
        "matched_restart_high_rate")


@pytest.fixture
def codec():
    av = pytest.importorskip("av", reason="optional PyAV not installed")
    pytest.importorskip("numpy", reason="NumPy not installed")
    try:
        av.CodecContext.create("libx264", "w")
    except Exception:
        pytest.skip("PyAV build does not include libx264")


def _args(**overrides):
    args = dict(width=96, height=64, fps=30, frames=36, gop=24,
                rates=RateConfig(silence_timeout_ms=150), adapt=True,
                link=LinkConfig(delay_ms=15, jitter_ms=0, bandwidth_kbps=0,
                                max_age_ms=0, max_inflight=16))
    args.update(overrides)
    return args


def test_stream_requires_a_fresh_matching_adaptive_policy():
    args = _args()
    for invalid in (True, {}, "policy"):
        with pytest.raises(ValueError, match="rate_policy"):
            stream_with_rate_control(**args, rate_policy=invalid)
    with pytest.raises(ValueError, match="rate_policy"):
        stream_with_rate_control(**{**args, "adapt": False},
                                 rate_policy=RateReceiver(args["rates"]))
    with pytest.raises(ValueError, match="rate_policy"):
        stream_with_rate_control(**args, rate_policy=RateReceiver(
            replace(args["rates"], high_kbps=700)))
    reused = RateReceiver(args["rates"])
    reused.source_frames_encoded = 1
    with pytest.raises(ValueError, match="rate_policy"):
        stream_with_rate_control(**args, rate_policy=reused)


def test_comparison_configuration_bounds_are_checked_before_encoding():
    for override in ({"load": False}, {"rates": {}}, {"recovery": 1}, {"link": "link"}):
        with pytest.raises(ValueError, match="configuration"):
            compare_load_control(**override)
    with pytest.raises(ValueError, match="budgets"):
        compare_load_control(rates=RateConfig(max_switches=13),
                             recovery=ControllerConfig(max_requests=8))
    for buffers in ([], (), (80, 40), (80, 80), (True,), (1001,)):
        with pytest.raises(ValueError, match="buffer"):
            compare_load_control(buffers_ms=buffers)


def test_explicit_legacy_policy_is_identical_to_the_default_stream(codec):
    args = _args(link=LinkConfig(delay_ms=40, jitter_ms=0, bandwidth_kbps=0,
                                 max_inflight=16, max_age_ms=0, drop_seqs=(5, 8)))
    recovery = ControllerConfig(feedback_ms=20, retry_ms=80)
    legacy = stream_with_rate_control(**args, recovery=recovery)
    explicit = stream_with_rate_control(**args, recovery=recovery,
                                        rate_policy=RateReceiver(args["rates"]))
    assert legacy == explicit


def test_sender_observes_queue_rejected_units_and_same_rate_recovery_idrs(codec, monkeypatch):
    rates = RateConfig(max_switches=0, silence_timeout_ms=150)
    policy = LoadRateReceiver(rates, LoadControlConfig(), fps=30)
    observed = []
    original = policy.observe_encoded

    def capture(packet, source_ms, target_kbps):
        observed.append(packet)
        original(packet, source_ms, target_kbps)

    monkeypatch.setattr(policy, "observe_encoded", capture)
    packets, trace, info = stream_with_rate_control(**_args(
        frames=24, rates=rates,
        link=LinkConfig(delay_ms=200, jitter_ms=0, bandwidth_kbps=0,
                        max_age_ms=0, max_inflight=1)),
        recovery=ControllerConfig(feedback_ms=20, max_requests=1), rate_policy=policy)
    assert observed == packets
    assert policy.source_frames_encoded == 24
    offered = sum(len(packet_bytes(p)) for p in observed)
    rejected = {seq for seq, reason in trace.dropped if reason == "queue_full"}
    assert rejected
    assert offered > trace.wire_sent_bytes
    assert offered - trace.wire_sent_bytes == sum(len(packet_bytes(packets[s])) for s in rejected)
    assert any(not x["rate_change"] for x in info["encoder_restarts"])
    assert info["switches_applied"] == 0


def test_clean_link_no_binding_wire_cap_preserves_all_four_streams(codec):
    report = compare_load_control(
        width=96, height=64, frames=36,
        link=LinkConfig(delay_ms=15, jitter_ms=0, bandwidth_kbps=0,
                        max_age_ms=0, max_inflight=16))
    assert report["wire_cap_binding_decisions"] == 0
    assert all(value is None for value in report["first_policy_difference_seqs"].values())
    assert all(report[name] == report[ARMS[0]] for name in ARMS)
    assert report["wire_load_recovery"]["actually_decoded_frames"] == 36


def test_congestion_comparison_accounts_for_offered_load_and_replays_every_idr(codec):
    report = compare_load_control(
        frames=120, link=LinkConfig(delay_ms=40, jitter_ms=0, bandwidth_kbps=0,
                                    bandwidth_steps=((0, 1200), (600, 300), (2200, 1200)),
                                    max_inflight=32, max_age_ms=1000))
    controller = report["controllers"]["wire_load_recovery"]
    wire = report["wire_load_recovery"]
    assert controller["switches_applied"] <= 12
    assert len(controller["encoder_restarts"]) <= 20
    assert report["restart_control"]["restart_seqs"] == [
        x["source_seq"] for x in controller["encoder_restarts"]]
    assert wire["encoded_idr_seqs"] == report["matched_restart_high_rate"]["encoded_idr_seqs"]
    common = set.intersection(*(
        {seq for first, last in report[name]["decoded_runs"] for seq in range(first, last + 1)}
        for name in ARMS))
    assert report["quality_comparison"]["common_frame_seqs"] == sorted(common)
    for name in ARMS:
        arm = report[name]
        assert arm["delivered_frames"] + arm["link_drops"] == 120
        assert arm["offered_wire_bytes"] == arm["wire_bytes"] + arm["queue_rejected_wire_bytes"]
        assert arm["offered_framing_bytes"] == 120 * 30
        assert sum(s["offered_wire_bytes"] for s in arm["target_segments"]) == arm["offered_wire_bytes"]
        assert sum(s["source_frames"] for s in arm["target_segments"]) == 120
        assert arm["quality_decoded_frames"] == arm["actually_decoded_frames"]
        assert arm["matched_frame_count"] == len(common)
    if report["wire_cap_binding_decisions"] == 0:
        assert report["age_only_recovery"] == wire
    json.dumps(report, allow_nan=False)
