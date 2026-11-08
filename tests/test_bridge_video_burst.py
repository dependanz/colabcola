"""v30 encoder-buffer isolation and offline target/restart replay."""
from __future__ import annotations

from dataclasses import replace
import math
from types import SimpleNamespace

import pytest

from colabcola import bridge_video_rate as rate_module
from colabcola.bridge_video_controller import ControllerConfig
from colabcola.bridge_video_impairment import LinkConfig, decode_delivered_h264
from colabcola.bridge_video_load import LoadControlConfig, LoadRateReceiver
from colabcola.bridge_video_rate import (
    EncoderBurstConfig, RateConfig, RateReceiver, stream_with_rate_control,
)


@pytest.fixture
def codec():
    av = pytest.importorskip("av", reason="optional PyAV not installed")
    pytest.importorskip("numpy", reason="NumPy not installed")
    try:
        av.CodecContext.create("libx264", "w")
    except Exception:
        pytest.skip("PyAV build does not include libx264")


def _args(**changes):
    args = dict(
        width=96, height=64, fps=30, frames=60, gop=24, adapt=False,
        link=LinkConfig(delay_ms=40, jitter_ms=0, bandwidth_kbps=0,
                        max_inflight=16, max_age_ms=0),
        rates=RateConfig(control_delay_ms=20, recovered_age_ms=0, min_switch_ms=0),
    )
    args.update(changes)
    return args


def _schedule(info):
    return tuple((x["source_seq"], x["target_kbps"]) for x in info["encoder_restarts"])


def test_burst_duration_requires_a_bounded_integer():
    assert EncoderBurstConfig().vbv_ms == 100
    for invalid in (0, -1, 1001, True, 100.0, math.nan, math.inf, "100", None):
        with pytest.raises(ValueError):
            EncoderBurstConfig(vbv_ms=invalid)


@pytest.mark.parametrize("duration, target, fps, expected_kbits", [
    (100, 650, 30, 65),
    (100, 173, 30, 18),  # Round a 17.3-kbit request up to whole kilobits.
    (1, 20, 60, 1),
    (1, 650, 30, 22),  # One nominal frame costs more than the duration request.
    (100, 650, 1, 650),
    (1000, 20_000, 60, 20_000),
])
def test_buffer_size_respects_integer_and_one_frame_floors(duration, target, fps, expected_kbits):
    assert EncoderBurstConfig(vbv_ms=duration).buffer_kbits(target, fps) == expected_kbits


def test_buffer_size_rejects_invalid_rate_or_source_cadence():
    config = EncoderBurstConfig()
    for target in (19, 20_001, True, 650.0, math.nan, "650"):
        with pytest.raises(ValueError):
            config.buffer_kbits(target, 30)
    for fps in (0, 61, True, 30.0, math.inf, "30"):
        with pytest.raises(ValueError):
            config.buffer_kbits(650, fps)


def test_encoder_configuration_changes_only_vbv_buffer_size():
    encoders = []

    def create(name, mode):
        assert (name, mode) == ("libx264", "w")
        encoder = SimpleNamespace()
        encoder.open = lambda: None
        encoders.append(encoder)
        return encoder

    av = SimpleNamespace(CodecContext=SimpleNamespace(create=create))
    legacy = rate_module._encoder(av, 96, 64, 30, 24, 173)
    explicit = rate_module._encoder(
        av, 96, 64, 30, 24, 173, burst=EncoderBurstConfig(vbv_ms=1000))
    tight = rate_module._encoder(av, 96, 64, 30, 24, 173, burst=EncoderBurstConfig(vbv_ms=100))
    assert len(encoders) == 3
    assert legacy.options == explicit.options
    legacy_params = dict(part.split("=", 1) for part in legacy.options["x264-params"].split(":"))
    tight_params = dict(part.split("=", 1) for part in tight.options["x264-params"].split(":"))
    assert legacy_params.pop("vbv-bufsize") == "173"
    assert tight_params.pop("vbv-bufsize") == "18"
    assert tight_params == legacy_params
    assert "vbv-init" not in tight_params
    assert {k: v for k, v in tight.options.items() if k != "x264-params"} == {
        k: v for k, v in legacy.options.items() if k != "x264-params"}
    attributes = ("width", "height", "pix_fmt", "time_base", "framerate", "thread_count", "bit_rate")
    assert all(getattr(tight, key) == getattr(legacy, key) == getattr(explicit, key)
               for key in attributes)
    assert tight.bit_rate == 173_000


def test_invalid_burst_and_replay_configuration_is_rejected_before_encoding(monkeypatch):
    def unexpected_encoder(*args, **kwargs):
        raise AssertionError("invalid experiment reached encoder creation")

    monkeypatch.setattr(rate_module, "_encoder", unexpected_encoder)
    for invalid in (False, 100, {}, "100"):
        with pytest.raises(ValueError):
            stream_with_rate_control(**_args(), burst=invalid)
    for schedule in (
        [], True, "schedule", ([8, 95],), ((8,),), ((8, 95, 0),),
        ((0, 95),), ((60, 95),), ((True, 95),), ((8.0, 95),),
        ((8, True),), ((8, 95.0),), ((8, 19),), ((8, 651),),
        ((8, 95), (8, 95)), ((11, 95), (8, 95)),
        tuple((seq, 95) for seq in range(1, 22)),
    ):
        with pytest.raises(ValueError, match="replay_schedule"):
            stream_with_rate_control(**_args(), replay_schedule=schedule)


def test_replay_cannot_mix_with_live_feedback_or_another_restart_schedule():
    for schedule in ((), ((8, 95),)):
        for changes in (
            {"adapt": True},
            {"recovery": ControllerConfig(max_requests=0)},
            {"rate_policy": RateReceiver(_args()["rates"])},
            {"restart_seqs": (9,)},
        ):
            args = _args()
            args.update(changes)
            with pytest.raises(ValueError):
                stream_with_rate_control(**args, replay_schedule=schedule)


def test_explicit_one_second_buffer_preserves_the_legacy_stream_and_report(codec):
    args = _args(adapt=True)
    args["link"] = replace(args["link"], drop_seqs=(5, 8, 27))
    recovery = ControllerConfig(feedback_ms=20, retry_ms=80, max_requests=5)
    old_packets, old_trace, old_info = stream_with_rate_control(**args, recovery=recovery)
    packets, trace, info = stream_with_rate_control(
        **args, recovery=recovery, burst=EncoderBurstConfig(vbv_ms=1000))
    assert packets == old_packets
    assert trace == old_trace
    assert {k: v for k, v in info.items() if k != "encoder_burst"} == old_info
    assert info["encoder_burst"] == {"vbv_ms": 1000}


def test_empty_offline_replay_preserves_fixed_rate_stream_and_labels_the_replay(codec):
    baseline, baseline_trace, baseline_info = stream_with_rate_control(**_args())
    packets, trace, info = stream_with_rate_control(**_args(), replay_schedule=())
    assert packets == baseline
    assert trace == baseline_trace
    assert info["encoder_restarts"] == []
    assert info["replay_schedule"] == []
    assert "offline" in info["replay_note"].lower()
    assert {k: v for k, v in info.items()
            if k not in ("encoder_restarts", "replay_schedule", "replay_note")} == baseline_info


@pytest.mark.parametrize("vbv_ms", [1000, 100])
def test_closed_loop_load_policy_replays_exactly_with_the_same_burst_setting(codec, vbv_ms):
    rates = RateConfig(
        congested_age_ms=30, recovered_age_ms=0, congestion_samples=3,
        control_delay_ms=20, min_switch_ms=0, max_switches=8)
    args = _args(rates=rates)
    args["link"] = replace(args["link"], drop_seqs=(5, 8, 27))
    burst = EncoderBurstConfig(vbv_ms=vbv_ms)
    policy = LoadRateReceiver(rates, LoadControlConfig(), fps=30)
    closed_packets, closed_trace, closed_info = stream_with_rate_control(
        **{**args, "adapt": True}, burst=burst, rate_policy=policy,
        recovery=ControllerConfig(feedback_ms=20, retry_ms=80, max_requests=5))
    schedule = _schedule(closed_info)
    assert closed_info["switches_applied"] >= 2
    assert closed_info["recovery"]["idr_restarts_applied"] >= 2
    assert dict(closed_trace.dropped)[8] == "erased"
    assert any(not x["rate_change"] and x["recovery_request_ids"]
               for x in closed_info["encoder_restarts"])
    packets, trace, info = stream_with_rate_control(
        **args, burst=burst, replay_schedule=schedule)
    assert packets == closed_packets
    assert trace == closed_trace
    assert _schedule(info) == schedule
    assert info["replay_schedule"] == [list(item) for item in schedule]
    assert info["requests_sent"] == 0
    assert info["switches_applied"] == 0  # Replay is not live receiver feedback.
    assert "recovery" not in info
    assert [p.seq for p in packets if p.keyframe] == [p.seq for p in closed_packets if p.keyframe]
    decoded = decode_delivered_h264(trace, 96, 64)
    assert decoded["decoded_frames"] > 0
    assert decoded["decoded_runs"][-1][-1] == args["frames"] - 1


def test_changed_burst_replay_preserves_targets_all_idrs_and_real_decoding(codec):
    args = _args(width=128, height=72)
    schedule = ((8, 95), (11, 95), (30, 650))
    old_packets, _, old_info = stream_with_rate_control(
        **args, replay_schedule=schedule, burst=EncoderBurstConfig(vbv_ms=1000))
    packets, trace, info = stream_with_rate_control(
        **args, replay_schedule=schedule, burst=EncoderBurstConfig(vbv_ms=100))
    assert _schedule(info) == _schedule(old_info) == schedule
    assert [x["rate_change"] for x in info["encoder_restarts"]] == [True, False, True]
    assert [p.seq for p in packets if p.keyframe] == [0, 8, 11, 30, 54]
    assert [p.seq for p in packets if p.keyframe] == [p.seq for p in old_packets if p.keyframe]
    # Encoder settings are different from startup, before any replayed action.
    assert packets[0].data != old_packets[0].data
    assert len(packets) == args["frames"]
    assert [p.seq for p in packets] == list(range(args["frames"]))
    decoded = decode_delivered_h264(trace, 128, 72)
    assert decoded["decoded_frames"] == 60
    assert decoded["decoded_runs"] == [[0, 59]]
