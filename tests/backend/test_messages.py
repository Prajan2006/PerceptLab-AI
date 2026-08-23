"""Protocol message parsing and DTO mapping tests."""

import pytest

from camera.interfaces.types import CaptureDeviceInfo, Resolution, SourceKind

from backend.models.messages import (
    PROTOCOL_VERSION,
    CameraConnectMessage,
    HelloMessage,
    DeviceListRequest,
    ProtocolError,
    device_to_dto,
    parse_client_message,
)


class TestProtocolVersion:
    def test_is_version_one(self):
        assert PROTOCOL_VERSION == 1


class TestClientMessageParsing:
    def test_hello_roundtrip(self):
        message = parse_client_message({"type": "hello", "client": "pytest"})
        assert isinstance(message, HelloMessage)
        assert message.client == "pytest"

    def test_devices_list(self):
        message = parse_client_message({"type": "devices.list"})
        assert isinstance(message, DeviceListRequest)

    def test_connect_device_id_optional(self):
        empty = parse_client_message({"type": "camera.connect"})
        assert isinstance(empty, CameraConnectMessage)
        assert empty.deviceId is None

        explicit = parse_client_message({"type": "camera.connect", "deviceId": "opencv:1"})
        assert explicit.deviceId == "opencv:1"

    def test_unknown_type_rejected(self):
        with pytest.raises(ProtocolError):
            parse_client_message({"type": "time.travel"})

    def test_non_object_payload_rejected(self):
        with pytest.raises(ProtocolError):
            parse_client_message([1, 2, 3])

    def test_wrong_field_type_rejected(self):
        with pytest.raises(ProtocolError):
            parse_client_message({"type": "camera.connect", "deviceId": 123})


class TestDeviceDtoMapping:
    def test_full_mapping_uses_camel_case_wire_names(self):
        info = CaptureDeviceInfo(
            id="opencv:0",
            label="Camera 0 (Media Foundation)",
            kind=SourceKind.CAMERA,
            resolution=Resolution(1280, 720),
            max_fps=30.0,
            backend="Media Foundation",
        )
        dto = device_to_dto(info).model_dump()
        assert dto == {
            "id": "opencv:0",
            "label": "Camera 0 (Media Foundation)",
            "kind": "camera",
            "resolution": {"width": 1280, "height": 720},
            "maxFps": 30.0,
            "backend": "Media Foundation",
        }

    def test_optional_fields_map_to_none(self):
        info = CaptureDeviceInfo(id="x", label="X")
        dto = device_to_dto(info).model_dump()
        assert dto["resolution"] is None
        assert dto["maxFps"] is None
        assert dto["backend"] is None
