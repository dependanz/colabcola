"""v33 integration: frozen real H.264, complete-only decoding and paid bytes."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

from colabcola import bridge_video_packetization as packetization
from colabcola.bridge_video_datagram import DATAGRAM_HEADER, DatagramConfig, datagram_bytes, fragment_unit
from colabcola.bridge_video_feedback import fixed_deadline_playout
from colabcola.bridge_video_impairment import MAX_WIRE_BYTES, LinkConfig, decode_delivered_h264
from colabcola.bridge_video_packet_transport import PacketSenderConfig
from colabcola.bridge_video_rate import EncoderBurstConfig, RateConfig, stream_with_rate_control
from colabcola.bridge_video_sender_service import SenderServiceConfig
from colabcola.bridge_video_stream import PACKET_HEADER, packet_bytes


WIDTH, HEIGHT, FPS, FRAMES = 128, 72, 30, 12
ARM_NAMES = ("batch_immediate", "batch_fixed", "batch_service",
             "packet_burst", "packet_fixed", "packet_service")


@pytest.fixture(scope="module")
def codec():
    av = pytest.importorskip("av", reason="optional PyAV not installed")
    pytest.importorskip("numpy", reason="NumPy not installed")
    try:
        av.CodecContext.create("libx264", "w")
    except Exception:
        pytest.skip("PyAV build does not include libx264")


def _link(**changes):
    return LinkConfig(**{
        "delay_ms": 20, "jitter_ms": 0, "bandwidth_kbps": 0,
        "max_inflight": 120, "max_age_ms": 0, **changes,
    })


@pytest.fixture(scope="module")
def frozen_source(codec):
    packets, _, _ = stream_with_rate_control(
        width=WIDTH, height=HEIGHT, fps=FPS, frames=FRAMES, gop=6,
        link=_link(), rates=RateConfig(), adapt=False,
        burst=EncoderBurstConfig(vbv_ms=100))
    assert [packet.seq for packet in packets if packet.keyframe] == [0, 6]
    return packets


@pytest.fixture(scope="module")
def clean_comparison(codec):
    # A small finite-rate, jittered control. Large byte/state limits remove
    # admission and local expiry as alternative causes of missing frames.
    return packetization.compare_packetization(
        width=WIDTH, height=HEIGHT, fps=FPS, frames=FRAMES, gop=6,
        link=_link(delay_ms=40, jitter_ms=8, bandwidth_kbps=1200),
        max_inflight_bytes=MAX_WIRE_BYTES, sender_max_pending=120,
        sender_max_age_ms=0, buffers_ms=(40, 120))


def _replay(packets, arm="packet_burst", **changes):
    grouping, pacing = packetization.ARMS[arm]
    options = dict(
        fps=FPS, link=_link(),
        sender=PacketSenderConfig(grouping, pacing, max_pending=120, max_age_ms=0),
        max_inflight_bytes=MAX_WIRE_BYTES,
    )
    return packetization.run_packet_replay(packets, **{**options, **changes})


def _wire_accounting(summary):
    values = summary["accounting"]
    assert values["generated_wire_bytes"] == values["local_discarded_wire_bytes"] + values["attempted_wire_bytes"]
    assert values["attempted_wire_bytes"] == values["admission_rejected_wire_bytes"] + values["admitted_wire_bytes"]
    assert values["admitted_wire_bytes"] == values["erased_wire_bytes"] + values["received_wire_bytes"]
    assert values["received_wire_bytes"] == values["reassembled_wire_bytes"] + values["unusable_received_wire_bytes"]
    assert values["received_wire_bytes"] == sum(row["received_wire_bytes"] for row in summary["frames"])
    assert values["unusable_received_wire_bytes"] == sum(row["unusable_received_wire_bytes"] for row in summary["frames"])
    assert summary["sender"]["pending_frames"] == summary["receiver"]["pending_frames"] == 0
    json.dumps(summary, allow_nan=False)


def test_six_arm_comparison_freezes_source_headers_idrs_and_matched_quality(clean_comparison):
    report = clean_comparison
    assert tuple(report["arms"]) == ARM_NAMES
    assert tuple(report["arm_configs"]) == ARM_NAMES
    reference = report["arms"]["batch_immediate"]
    for summary in report["arms"].values():
        for field in ("encoded_source_sha256", "datagram_source_sha256", "encoded_idr_seqs",
                      "encoded_payload_bytes", "original_framed_bytes", "datagram_header_bytes",
                      "datagram_wire_bytes", "datagram_count", "frame_datagram_counts"):
            assert summary[field] == reference[field]
        assert summary["original_framed_bytes"] == summary["encoded_payload_bytes"] + FRAMES * PACKET_HEADER.size
        assert summary["datagram_header_bytes"] == summary["datagram_count"] * DATAGRAM_HEADER.size
        assert summary["datagram_wire_bytes"] == summary["original_framed_bytes"] + summary["datagram_header_bytes"]
        assert summary["largest_datagram_bytes"] <= report["datagram_config"]["max_datagram_bytes"]
        assert summary["reassembled_frames"] == summary["actually_decoded_frames"] == FRAMES
        assert summary["matched_frame_count"] == FRAMES
        assert summary["matched_frame_rgb_psnr_db"] == reference["matched_frame_rgb_psnr_db"]
        _wire_accounting(summary)
    assert report["quality_comparison"]["common_frame_seqs"] == list(range(FRAMES))
    json.dumps(report, allow_nan=False)


def test_all_six_replays_reconstruct_exact_original_au_bytes_and_clocks(frozen_source):
    datagram = DatagramConfig(256)
    framed = b"".join(packet_bytes(packet) for packet in frozen_source)
    expected_wire = b"".join(datagram_bytes(fragment, datagram)
                             for packet in frozen_source for fragment in fragment_unit(packet, datagram))
    for arm in ARM_NAMES:
        trace, summary = _replay(frozen_source, arm, datagram=datagram)
        assert trace.source_count == FRAMES
        assert [item.packet for item in trace.delivered] == frozen_source
        assert summary["encoded_source_sha256"] == hashlib.sha256(framed).hexdigest()
        assert summary["datagram_source_sha256"] == hashlib.sha256(expected_wire).hexdigest()
        assert summary["datagram_wire_bytes"] == len(expected_wire)
        for item in trace.delivered:
            assert item.source_ms == item.packet.seq * 1000.0 / FPS
            assert item.packet.pts_ns == item.packet.capture_ns == round(item.packet.seq * 1_000_000_000 / FPS)
            assert item.wire_size == len(packet_bytes(item.packet))
        _wire_accounting(summary)


def test_clean_batch_and_packet_burst_have_equal_au_completion_times(clean_comparison):
    batch = clean_comparison["arms"]["batch_immediate"]
    burst = clean_comparison["arms"]["packet_burst"]
    assert batch["link"]["rejected_packets"] == burst["link"]["rejected_packets"] == 0
    assert batch["link"]["erased_packets"] == burst["link"]["erased_packets"] == 0
    assert batch["link"]["attempted_groups"] == FRAMES
    assert burst["link"]["attempted_groups"] > FRAMES
    assert [row["reassembly_completion_ms"] for row in batch["frames"]] == pytest.approx(
        [row["reassembly_completion_ms"] for row in burst["frames"]], rel=0, abs=1e-9)
    assert batch["accounting"] == burst["accounting"]
    assert batch["decoded_runs"] == burst["decoded_runs"] == [[0, FRAMES - 1]]
    assert batch["playout"] == burst["playout"]


def test_every_arm_preserves_full_presentation_horizon_when_no_packet_is_admitted(frozen_source):
    for arm in ARM_NAMES:
        trace, summary = _replay(frozen_source, arm, max_inflight_bytes=1)
        assert trace.source_count == summary["source_frames"] == FRAMES
        assert trace.delivered == ()
        assert summary["receiver"]["unseen_frames"] == summary["receiver"]["missing_frames"] == FRAMES
        assert summary["receiver"]["expired_frames"] == 0
        assert [row["source_seq"] for row in summary["frames"]] == list(range(FRAMES))
        observed = decode_delivered_h264(trace, WIDTH, HEIGHT)
        playout = fixed_deadline_playout(trace, observed, fps=FPS, buffer_ms=120)
        assert playout["on_time_frames"] == playout["late_decoded_frames"] == 0
        assert playout["undecodable_frames"] == playout["initial_blank_slots"] == FRAMES
        assert playout["longest_unfresh_run_ms"] == round(FRAMES * 1000.0 / FPS, 2)
        _wire_accounting(summary)


def test_first_fragment_erasure_never_decodes_partial_au_and_recovers_at_frozen_idr(frozen_source):
    config = DatagramConfig()
    erased = fragment_unit(frozen_source[2], config)[0]
    expected_unusable = sum(len(datagram_bytes(fragment, config))
                            for fragment in fragment_unit(frozen_source[2], config)[1:])
    assert expected_unusable > 0
    for arm in ARM_NAMES:
        trace, summary = _replay(frozen_source, arm, datagram=config,
                                 erase_first_seqs=(2,), reassembly_max_age_ms=350)
        assert [item.packet.seq for item in trace.delivered] == [seq for seq in range(FRAMES) if seq != 2]
        assert all(item.packet == frozen_source[item.packet.seq] for item in trace.delivered)
        observed = decode_delivered_h264(trace, WIDTH, HEIGHT)
        assert observed["decoded_runs"] == [[0, 1], [6, FRAMES - 1]]
        assert observed["dependency_skipped"] == 3
        assert observed["idr_resynchronizations"] == 1
        assert summary["encoded_idr_seqs"] == [0, 6]
        assert summary["link"]["erased_packets"] == 1
        assert summary["accounting"]["erased_wire_bytes"] == len(datagram_bytes(erased, config))
        assert summary["accounting"]["unusable_received_wire_bytes"] == expected_unusable
        row = summary["frames"][2]
        assert row["received_packets"] == row["generated_packets"] - 1
        assert row["receiver_state"] == "expired"
        assert row["reassembly_completion_ms"] is None
        assert row["received_wire_bytes"] == row["unusable_received_wire_bytes"] == expected_unusable
        _wire_accounting(summary)


def test_partial_admission_accounts_for_received_but_unusable_wire(frozen_source):
    trace, summary = _replay(
        frozen_source, "packet_burst", link=_link(delay_ms=200),
        max_inflight_bytes=1200, reassembly_max_age_ms=350)
    first = summary["frames"][0]
    assert first["generated_packets"] > 1
    assert first["received_packets"] == 1
    assert first["admission_rejected_packets"] == first["generated_packets"] - 1
    assert first["received_wire_bytes"] == first["unusable_received_wire_bytes"] == 1200
    assert first["receiver_state"] == "expired"
    assert 0 not in [item.packet.seq for item in trace.delivered]
    assert summary["accounting"]["admission_rejected_wire_bytes"] > 0
    assert summary["accounting"]["unusable_received_wire_bytes"] > 0
    received_unfinished = sum(row["received_wire_bytes"] for row in summary["frames"]
                              if row["receiver_state"] != "complete")
    assert summary["accounting"]["unusable_received_wire_bytes"] == received_unfinished
    assert summary["accounting"]["erased_wire_bytes"] == 0
    _wire_accounting(summary)


def test_downstream_rejection_cannot_change_sender_schedule_or_abort_later_fragments(frozen_source):
    for arm in ARM_NAMES:
        _, accepted = _replay(frozen_source, arm)
        _, rejected = _replay(frozen_source, arm, max_inflight_bytes=1,
                              link=_link(bandwidth_steps=((0, 90), (200, 2200))))
        assert rejected["sender"] == accepted["sender"]
        assert rejected["encoded_source_sha256"] == accepted["encoded_source_sha256"]
        assert rejected["datagram_source_sha256"] == accepted["datagram_source_sha256"]
        assert rejected["link"]["rejected_packets"] == rejected["datagram_count"]
        assert rejected["sender"]["remaining_debt_bits"] == accepted["sender"]["remaining_debt_bits"]
        attempted = [tuple(row["fragment_id"]) for row in rejected["link"]["packets"]]
        assert attempted == [(seq, index) for seq, count in enumerate(rejected["frame_datagram_counts"])
                             for index in range(count)]
        _wire_accounting(rejected)


def test_replay_rejects_noncontiguous_or_retimed_sources_before_any_dispatch(frozen_source, monkeypatch):
    def unexpected_dispatch(*args, **kwargs):
        raise AssertionError("invalid source reached the sender")

    monkeypatch.setattr(packetization.PacketSenderQueue, "enqueue", unexpected_dispatch)
    invalid = [[], frozen_source * 11, frozen_source[1:],
               [replace(frozen_source[0], seq=1), *frozen_source[1:]],
               [replace(frozen_source[0], pts_ns=1), *frozen_source[1:]],
               [replace(frozen_source[0], capture_ns=1), *frozen_source[1:]],
               [replace(frozen_source[0], data=b""), *frozen_source[1:]]]
    for source in invalid:
        with pytest.raises(ValueError):
            _replay(source)


def test_comparison_and_cli_reject_invalid_controls_before_encoding(monkeypatch, capsys):
    def unexpected_encoder(*args, **kwargs):
        raise AssertionError("invalid configuration reached the encoder")

    monkeypatch.setattr(packetization, "stream_with_rate_control", unexpected_encoder)
    for changes in (
        {"buffers_ms": ()}, {"buffers_ms": (80, 40)}, {"buffers_ms": (40, 40)},
        {"buffers_ms": (True,)}, {"link": {}}, {"datagram": {}}, {"service": {}},
        {"service": SenderServiceConfig(queue_boost=False)},
        {"max_inflight_bytes": 0}, {"max_inflight_bytes": MAX_WIRE_BYTES + 1},
        {"sender_max_pending": 121}, {"sender_max_age_ms": -1},
        {"reassembly_max_age_ms": float("nan")}, {"reassembly_max_bytes": 0},
        {"erase_first_seqs": (120,)}, {"link": _link(drop_seqs=(2,))},
    ):
        with pytest.raises(ValueError):
            packetization.compare_packetization(**changes)
    for args in (
        ["--frames", "121"], ["--max-datagram-bytes", "255"],
        ["--link-max-inflight-bytes", "0"], ["--sender-max-pending", "121"],
        ["--reassembly-max-age-ms", "nan"], ["--erase-first-seqs", "120"],
        ["--drain-ms", "0"], ["--bandwidth-steps", "400:1200"],
    ):
        with pytest.raises(SystemExit) as exc:
            packetization.main(args)
        assert exc.value.code == 2
        captured = capsys.readouterr()
        assert "error:" in captured.err and captured.out == ""


def test_cli_preserves_typed_controls_and_serializes_finite_report(clean_comparison, monkeypatch, capsys):
    received = {}

    def comparison(**kwargs):
        received.update(kwargs)
        return clean_comparison

    monkeypatch.setattr(packetization, "compare_packetization", comparison)
    assert packetization.main([
        "--frames", "12", "--gop", "6", "--bandwidth-steps", "", "--bandwidth-kbps", "1200",
        "--max-datagram-bytes", "256", "--link-max-inflight-bytes", "8192",
        "--sender-max-pending", "5", "--sender-max-age-ms", "80",
        "--reassembly-max-age-ms", "350", "--reassembly-max-bytes", "32768",
        "--erase-first-seqs", "2,7", "--drain-ms", "40", "--service-cap-factor", "3",
        "--buffers-ms", "40,120",
    ]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == json.loads(json.dumps(clean_comparison, allow_nan=False))
    assert captured.err == ""
    assert received["frames"] == 12 and received["gop"] == 6
    assert received["link"].bandwidth_steps == () and received["link"].bandwidth_kbps == 1200
    assert received["datagram"] == DatagramConfig(256)
    assert received["service"] == SenderServiceConfig(drain_ms=40, max_rate_factor=3)
    assert received["max_inflight_bytes"] == 8192
    assert received["sender_max_pending"] == 5 and received["sender_max_age_ms"] == 80
    assert received["reassembly_max_age_ms"] == 350 and received["reassembly_max_bytes"] == 32768
    assert received["erase_first_seqs"] == (2, 7) and received["buffers_ms"] == (40, 120)
