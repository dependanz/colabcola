"""Offline frame-protocol and live H.264 socketpair streaming regressions."""
from __future__ import annotations

import shutil
import socket

import pytest

from colabcola.bridge_video_stream import (
    AccessUnitParser, MAGIC, VERSION, MAX_PACKET, PACKET_HEADER,
    VideoAccessUnit, compare_stream, packet_bytes, receive_packet,
    frame_stage_times_ms,
)


def test_annex_b_access_units_are_reassembled_across_read_boundaries():
    first = b"\x00\x00\x00\x01\x09\x10\x00\x00\x01\x65\x01\x02"
    second = b"\x00\x00\x00\x01\x09\x30\x00\x00\x01\x41\x03"
    parser = AccessUnitParser()
    assert parser.feed(first[:7]) == []
    assert parser.feed(first[7:] + second[:5]) == [first]
    assert parser.feed(second[5:]) == []
    assert parser.finish() == [second]
    assert parser.finish() == []


def test_local_video_packet_roundtrip_with_timestamps_and_keyframe_flag():
    sender, receiver = socket.socketpair()
    try:
        original = VideoAccessUnit(
            seq=2, pts_ns=166666667, capture_ns=123456789,
            data=b"\x00\x00\x00\x01\x09\x10", keyframe=True,
        )
        sender.sendall(packet_bytes(original))
        assert receive_packet(receiver) == original
        sender.shutdown(socket.SHUT_WR)
        assert receive_packet(receiver) is None
    finally:
        sender.close()
        receiver.close()


@pytest.mark.parametrize("seq,pts,stamp,data", [
    (-1, 10, 20, b"x"),
    (1, -1, 20, b"x"),
    (1, 10, -1, b"x"),
    (1, 10, 20, b""),
    (1, 10, 20, b"x" * (MAX_PACKET + 1)),
    (True, 10, 20, b"x"),
])
def test_invalid_video_packets_rejected_before_socket_transfer(seq, pts, stamp, data):
    with pytest.raises(ValueError):
        packet_bytes(VideoAccessUnit(seq, pts, stamp, data))


def test_receiver_rejects_oversized_frame_header_before_payload_allocation():
    sender, receiver = socket.socketpair()
    try:
        sender.sendall(PACKET_HEADER.pack(
            MAGIC, VERSION, 0, 1, 0, 123, MAX_PACKET + 1,
        ))
        with pytest.raises(ValueError, match="oversized"):
            receive_packet(receiver)
    finally:
        sender.close()
        receiver.close()


def test_frame_latency_stage_durations_sum_to_total_age():
    # 50ms before access unit, .4ms for local socket, 30ms for decoder.
    stages = frame_stage_times_ms(1_000_000_000, 1_050_000_000,
                                  1_050_400_000, 1_080_400_000)
    assert stages == pytest.approx((50, .4, 30))
    assert sum(stages) == pytest.approx(80.4)
    for times in ((4, 3, 5, 6), (-1, 3, 5, 6), (1, 2, 4, 3)):
        with pytest.raises(ValueError):
            frame_stage_times_ms(*times)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="optional FFmpeg unavailable")
def test_persistent_encoder_and_decoder_recover_frames_before_source_finishes():
    report = compare_stream(width=96, height=64, frames=24, fps=12, pattern="square")
    assert report["decoded_frames"] == 24
    # This is the meaningful property absent from the previous batch test:
    # the decoder emits frames while the source is still producing them.
    assert report["decoded_before_source_finished"] >= 3
    assert 4 <= report["live_decode_fps"] <= 30
    assert report["capture_to_decoded_p50_ms"] > 0
    for stage in ("capture_to_packet_p50_ms", "packet_to_socket_p50_ms",
                  "socket_to_decoded_p50_ms"):
        assert report[stage] >= 0
    assert report["capture_to_decoded_frame_periods"] > 0
    assert report["keyframes"] >= 1
    assert report["h264_bytes"] > 0
    assert report["scope"].startswith("paced offline local")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="optional FFmpeg unavailable")
def test_low_delay_decoder_mode_is_explicit_and_decodes_all_frames():
    report = compare_stream(width=96, height=64, frames=24, fps=12,
                            pattern="square", decoder_low_delay=True)
    assert report["decoder_low_delay"] is True
    assert report["decoded_frames"] == 24
    assert report["decoded_before_source_finished"] > 0
    assert report["capture_to_decoded_p50_ms"] > 0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="optional FFmpeg unavailable")
def test_stream_scene_pattern_roundtrip_small():
    report = compare_stream(width=64, height=48, frames=8, fps=12, pattern="scene")
    assert report["decoded_frames"] == 8
    assert report["keyframes"] >= 1
