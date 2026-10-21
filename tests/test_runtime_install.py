import io

import pytest

from scripts import install_edgetpu


def test_unverified_runtime_is_rejected_before_installation(tmp_path, monkeypatch):
    monkeypatch.setattr(install_edgetpu.platform, "system", lambda: "Linux")
    monkeypatch.setattr(install_edgetpu.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        install_edgetpu.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b"corrupt")
    )
    destination = tmp_path / "lib"
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        install_edgetpu.install_runtime(destination, tmp_path / "license")
    assert not destination.exists()


def test_unsupported_runtime_platform_fails_before_download(tmp_path, monkeypatch):
    monkeypatch.setattr(install_edgetpu.platform, "system", lambda: "Linux")
    monkeypatch.setattr(install_edgetpu.platform, "machine", lambda: "unsupported")
    monkeypatch.setattr(
        install_edgetpu.urllib.request,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("must not download a runtime for the wrong machine"),
    )
    with pytest.raises(RuntimeError, match="supported Linux hardware"):
        install_edgetpu.install_runtime(tmp_path / "lib", tmp_path / "license")
