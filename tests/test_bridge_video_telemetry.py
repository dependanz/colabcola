"""v33h: deterministic host-side display/transfer metric contract."""
import pytest

from colabcola.bridge_video_comm_session import application_video_rates, local_elapsed_us


def test_application_rates_are_distinct_and_exclude_network_overhead():
    label = application_video_rates(
        h264_bytes=1024, ccv1_bytes=1536, base64_chars=2048, elapsed=1)
    assert label == "H.264 1.0 KiB/s · CCV1 1.5 KiB/s · Base64 2.0 KiB/s"
    # Counts are byte-equivalent lengths of the ASCII Base64 string only.
    assert "WebSocket" not in label


@pytest.mark.parametrize("args", [
    {"h264_bytes": -1, "ccv1_bytes": 0, "base64_chars": 0, "elapsed": 1},
    {"h264_bytes": 0, "ccv1_bytes": True, "base64_chars": 0, "elapsed": 1},
    {"h264_bytes": 0, "ccv1_bytes": 0, "base64_chars": 0, "elapsed": 0},
])
def test_bad_application_accounting_rejected(args):
    with pytest.raises(ValueError):
        application_video_rates(**args)


def test_high_resolution_host_only_timing():
    assert local_elapsed_us(1_000_000, 1_002_500) == 2.5
    assert local_elapsed_us(12, 12) == 0
    for a, b in ((-1, 4), (9, 8), (True, 3), (2, 3.4)):
        with pytest.raises(ValueError):
            local_elapsed_us(a, b)
