"""Frame envelope encoding tests (pure functions, no sockets)."""

import json
import struct

import numpy as np

from camera.interfaces.types import Frame, FrameStamp

from backend.services.camera_gateway import encode_frame_message


def make_frame(seq: int = 1, image=None) -> Frame:
    stamp = FrameStamp(
        sequence=seq,
        monotonic_ns=1_000 + seq,
        wallclock_ns=1_700_000_000_000_000_000 + seq,
        fps=29.9,
    )
    if image is None:
        return Frame(stamp=stamp, width=64, height=48, image=None)
    return Frame(stamp=stamp, width=image.shape[1], height=image.shape[0], image=image)


def split_envelope(blob: bytes):
    (header_length,) = struct.unpack("<I", blob[:4])
    header = json.loads(blob[4 : 4 + header_length].decode("utf-8"))
    payload = blob[4 + header_length :]
    return header, payload


class TestFrameEnvelope:
    def test_layout_header_then_jpeg(self):
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        blob = encode_frame_message(make_frame(image=image), jpeg_quality=80)

        header, payload = split_envelope(blob)
        assert header["seq"] == 1
        assert header["monoNs"] == 1_001
        assert header["wallNs"] == 1_700_000_000_000_000_001
        assert abs(header["fps"] - 29.9) < 1e-9
        assert (header["w"], header["h"]) == (64, 48)
        assert header["enc"] == "jpeg"

        assert len(payload) > 0
        assert payload[:2] == b"\xff\xd8", "payload must start with JPEG SOI marker"

    def test_metadata_only_tick_has_empty_payload(self):
        blob = encode_frame_message(make_frame())
        header, payload = split_envelope(blob)
        assert header["enc"] == "none"
        assert payload == b""
        assert len(blob) == 4 + len(json.dumps(header, separators=(",", ":")))

    def test_sequence_preserved_per_frame(self):
        first = encode_frame_message(make_frame(seq=1))
        second = encode_frame_message(make_frame(seq=2))
        h1, _ = split_envelope(first)
        h2, _ = split_envelope(second)
        assert h2["seq"] == h1["seq"] + 1
