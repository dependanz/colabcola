"""Burst-window arithmetic and controlled four-arm codec comparisons."""
import json

import pytest

from colabcola.bridge_video_burst import _burst_metrics, compare_encoder_bursts
from colabcola.bridge_video_controller import ControllerConfig
from colabcola.bridge_video_impairment import LinkConfig
from colabcola.bridge_video_rate import EncoderBurstConfig, RateConfig
from colabcola.bridge_video_stream import VideoAccessUnit


ARMS = ("legacy_live", "tight_live", "tight_replay_legacy_schedule", "legacy_replay_tight_schedule")


@pytest.fixture
def codec():
    av = pytest.importorskip("av", reason="optional PyAV not installed")
    pytest.importorskip("numpy", reason="NumPy not installed")
    try:
        av.CodecContext.create("libx264", "w")
    except Exception:
        pytest.skip("PyAV build does not include libx264")


def test_burst_comparison_rejects_invalid_settings_and_windows_before_encoding():
    for invalid in ({"tight": False}, {"load": {}}, {"rates": 1}, {"recovery": "recovery"}):
        with pytest.raises(ValueError, match="configuration"):
            compare_encoder_bursts(**invalid)
    for invalid in ([], (), (0,), (1, 1), (9, 3), (True,), (61,)):
        with pytest.raises(ValueError, match="burst windows"):
            compare_encoder_bursts(burst_frames=invalid)
    for invalid in ([], (), (40, 40), (80, 40), (True,), (1001,)):
        with pytest.raises(ValueError, match="buffer depths"):
            compare_encoder_bursts(buffers_ms=invalid)
    with pytest.raises(ValueError, match="budgets"):
        compare_encoder_bursts(rates=RateConfig(max_switches=13),
                               recovery=ControllerConfig(max_requests=8))


def test_burst_windows_split_at_same_rate_restarts_and_use_actual_partial_duration():
    packets = [VideoAccessUnit(seq, seq * 100_000_000, seq * 100_000_000,
                               b"x" * size, keyframe=seq in (0, 2, 4))
               for seq, size in enumerate((100, 200, 300, 400, 500))]
    info = {"encoder_restarts": [{"source_seq": 2, "target_kbps": 20},
                                  {"source_seq": 4, "target_kbps": 20}]}
    result = _burst_metrics(packets, info, fps=10, high_kbps=200,
                           burst=EncoderBurstConfig(vbv_ms=1), windows=(1, 3, 9))
    segments = result["encoder_segments"]
    assert [(s["start_seq"], s["end_seq_exclusive"]) for s in segments] == [(0, 2), (2, 4), (4, 5)]
    assert segments[0]["configured_vbv_kbits"] == 20
    assert segments[0]["configured_vbv_equivalent_ms"] == 100
    window = segments[0]["initial_windows"][1]
    assert window == {"requested_frames": 3, "measured_frames": 2, "complete": False,
                      "duration_ms": 200.0, "payload_bytes": 300, "wire_bytes": 360,
                      "payload_kbps": 12.0, "offered_wire_kbps": 14.4,
                      "payload_to_target_ratio": 0.06}
    assert segments[2]["initial_windows"][0]["complete"] is True
    assert segments[2]["initial_windows"][1]["measured_frames"] == 1
    peak = result["peak_source_windows"][1]
    assert peak["full_windows"] == 3
    assert peak["peak"] == {"start_seq": 2, "end_seq_exclusive": 5,
                            "wire_bytes": 1290, "offered_wire_kbps": 34.4,
                            "mean_target_kbps": 20.0, "framing_kbps": 2.4}
    assert result["peak_source_windows"][2]["peak"] is None
    assert result["encoded_idr_payload_bytes"] == 900
    assert result["encoded_idr_wire_bytes"] == 990
    json.dumps(result, allow_nan=False)


def test_identical_vbv_setting_makes_all_four_arm_summaries_identical(codec):
    report = compare_encoder_bursts(
        width=96, height=64, frames=36, tight=EncoderBurstConfig(vbv_ms=1000),
        link=LinkConfig(delay_ms=15, jitter_ms=0, bandwidth_kbps=0,
                        max_inflight=16, max_age_ms=0))
    assert report["first_controller_action_difference_seq"] is None
    assert all(report[name] == report[ARMS[0]] for name in ARMS)
    assert report["legacy_live"]["actually_decoded_frames"] == 36


def test_crossover_preserves_actions_and_accounts_for_real_decoding_and_all_bytes(codec):
    report = compare_encoder_bursts(
        frames=120, link=LinkConfig(delay_ms=40, jitter_ms=0, bandwidth_kbps=0,
                                    bandwidth_steps=((0, 1200), (600, 300), (2200, 1200)),
                                    max_inflight=32, max_age_ms=1000))
    for live, replay in report["offline_controls"]["same_schedule_pairs"]:
        a, b = report[live], report[replay]
        assert a["encoded_idr_seqs"] == b["encoded_idr_seqs"]
        assert [(s["start_seq"], s["target_kbps"]) for s in a["encoder_segments"]] == [
            (s["start_seq"], s["target_kbps"]) for s in b["encoder_segments"]]
    common = set.intersection(*(
        {seq for start, end in report[name]["decoded_runs"] for seq in range(start, end + 1)}
        for name in ARMS))
    assert report["quality_comparison"]["common_frame_seqs"] == sorted(common)
    for name in ARMS:
        arm = report[name]
        assert arm["delivered_frames"] + arm["link_drops"] == 120
        assert arm["offered_wire_bytes"] == arm["wire_bytes"] + arm["queue_rejected_wire_bytes"]
        assert arm["quality_decoded_frames"] == arm["actually_decoded_frames"]
        assert arm["matched_frame_count"] == len(common)
        assert len(arm["encoder_segments"]) <= 21
        for segment in arm["encoder_segments"]:
            for window in segment["initial_windows"]:
                assert window["complete"] == (window["requested_frames"] == window["measured_frames"])
                assert window["measured_frames"] <= segment["end_seq_exclusive"] - segment["start_seq"]
    json.dumps(report, allow_nan=False)
