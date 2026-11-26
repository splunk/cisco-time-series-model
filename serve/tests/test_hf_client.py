from huggingface_hub.utils import get_session

from app.core import hf_client
from app.core.config import Settings


def test_hub_http_factory_preserves_native_client_type(monkeypatch):
    client_type = type(get_session())
    factories = []
    monkeypatch.setattr(hf_client, "_truststore", None)
    monkeypatch.setattr(hf_client.hf_constants, "HF_HUB_DOWNLOAD_TIMEOUT", 30)
    monkeypatch.setattr(hf_client, "set_client_factory", factories.append)

    hf_client.configure_huggingface_http(
        Settings(auth_token="test-key", hf_hub_download_timeout_seconds=90)
    )

    (factory,) = factories
    with factory() as client:
        assert type(client) is client_type
        assert client.follow_redirects
        assert client.event_hooks["request"] == [hf_client.hf_request_event_hook]
    assert hf_client.hf_constants.HF_HUB_DOWNLOAD_TIMEOUT == 90
