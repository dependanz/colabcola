"""v33w: offline freshness/latency trade-off selection (no GPU/network)."""
from __future__ import annotations

import struct
import zlib

import pytest

from colabcola.bridge_video_colab_source import (
    RING_SLOTS, RemoteH264PullSource, RemoteVideoSourceError,
    read_next_buffered_rgb,
)
from colabcola.bridge_video_comm_protocol import decode_video_reply
from colabcola.bridge_comm import KernelCommBridge

RUN = "8" * 32
W, H = 640, 360
IDR = b"\x00\x00\x00\x01\x65\x88\x84"
P = b"\x00\x00\x00\x01\x41\x88\x84"


def publish(directory, seq, ack=0):
    rgb = bytes((seq % 251, 15, 20)) * (W * H)
    ppm = b"P6\n640 360\n255\n" + rgb
    packed = struct.pack("!4sIIII", b"CCB1", seq, ack, W, H)
    packed += zlib.compress(ppm, level=1)
    (directory / f"frame-ring-{seq % RING_SLOTS:02d}.bin").write_bytes(packed)


def source(directory, lag):
    def encoder_factory(w, h, fps):
        assert (w, h, fps) == (W, H, 30)
        def encode(pixels, coded_seq):
            assert len(pixels) == W * H * 3
            return (IDR if coded_seq == 0 else P), coded_seq == 0
        return encode
    return RemoteH264PullSource(
        directory, RUN, fps=30, sequence_buffered=True,
        max_buffer_lag_frames=lag, encoder_factory=encoder_factory)


@pytest.mark.parametrize("lag,expected,skipped", [
    (0, 5, 4), (1, 4, 3), (2, 3, 2),
])
def test_tradeoff_selects_within_intentional_freshness_bound(tmp_path, lag, expected, skipped):
    publish(tmp_path, 0)
    video = source(tmp_path, lag)
    original = video.pull(1)
    assert original["renderer_seq"] == 0
    for seq in range(1, 6):
        publish(tmp_path, seq, ack=seq)
    second = video.pull(2)
    assert second["renderer_seq"] == expected
    assert second["renderer_buffer_lag"] == lag
    assert second["renderer_frames_skipped"] == skipped
    assert second["input_ack"] == expected
    unit, ack = decode_video_reply(second, run_id=RUN, expected_pull_seq=2)
    assert unit.seq == 1 and ack == expected

    # With no new publication, bounded history either has another distinct
    # frame available or acknowledges no-new without advancing the codec.
    next_reply = video.pull(3)
    if lag == 0:
        assert next_reply["kind"] == "video_no_new"
        assert video.coded_seq == 2
    else:
        assert next_reply["kind"] == "video_access_unit"
        assert next_reply["renderer_seq"] == expected + 1
        assert decode_video_reply(next_reply, run_id=RUN, expected_pull_seq=3)[0].seq == 2


@pytest.mark.parametrize("bad", [-1, 3, True, 1.0, "1"])
def test_invalid_lag_fails_closed_offline(tmp_path, bad):
    with pytest.raises(RemoteVideoSourceError):
        read_next_buffered_rgb(tmp_path, after_seq=None, max_lag_frames=bad)
    with pytest.raises(RemoteVideoSourceError):
        RemoteH264PullSource(tmp_path, RUN, fps=30,
                             sequence_buffered=True, max_buffer_lag_frames=bad,
                             encoder_factory=lambda *_: lambda pixels, seq: (IDR, True))
    with pytest.raises(ValueError):
        KernelCommBridge(None, None, RUN, 30, video_codec="h264",
                         sequence_buffered=True, max_buffer_lag_frames=bad)


def test_nonbuffered_mode_rejects_nondefault_lag(tmp_path):
    with pytest.raises(ValueError):
        KernelCommBridge(None, None, RUN, 30, video_codec="h264",
                         sequence_buffered=False, max_buffer_lag_frames=1)
    with pytest.raises(RemoteVideoSourceError):
        RemoteH264PullSource(tmp_path, RUN, fps=30,
                             sequence_buffered=False, max_buffer_lag_frames=1,
                             encoder_factory=lambda *_: lambda pixels, seq: (IDR, True))


@pytest.mark.parametrize("lag", [0, 1, 2])
def test_cli_passes_explicit_lag_to_external_buffered_mode(monkeypatch, lag):
    from colabcola.cli import main
    from colabcola import bridge_colab
    seen = []
    monkeypatch.setattr(bridge_colab, "run_colab", lambda args: seen.append(args) or 0)
    args = ["--profile", "example", "bridge", "colab",
            "--attach-existing", "--sequence-buffered",
            "--renderer-lag-frames", str(lag),
            "--transport", "comm", "--video-codec", "h264", "--inflight", "1",
            "--fps", "30", "--poll-hz", "30",
            "--benchmark-h264-capacity", "--preflight-only",
            "--max-seconds", "300"]
    assert main(args) == 0
    assert seen[-1].renderer_lag_frames == lag
    assert seen[-1].sequence_buffered


def test_cli_rejects_nonbuffered_and_owned_mode_lag(monkeypatch):
    from colabcola.cli import main
    from colabcola import bridge_colab
    monkeypatch.setattr(bridge_colab, "run_colab",
                        lambda _: pytest.fail("invalid lag reached provider"))
    baseline = ["--profile", "example", "bridge", "colab",
                "--transport", "comm", "--video-codec", "h264",
                "--fps", "30", "--poll-hz", "30",
                "--benchmark-h264-capacity", "--preflight-only"]
    with pytest.raises(SystemExit):
        main([*baseline, "--attach-existing", "--renderer-lag-frames", "1"])
    with pytest.raises(SystemExit):
        main([*baseline, "--sequence-buffered", "--renderer-lag-frames", "1"])
    with pytest.raises(SystemExit):
        main([*baseline, "--attach-existing", "--sequence-buffered",
              "--renderer-lag-frames", "3"])
