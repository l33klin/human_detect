"""Install the pinned official standard runtime without relying on the Coral APT service."""

import hashlib
import io
import platform
import urllib.request
import zipfile
from pathlib import Path

URL = (
    "https://github.com/google-coral/libedgetpu/releases/download/"
    "release-grouper/edgetpu_runtime_20221024.zip"
)
SHA256 = "7f17d85e6f21ec861af3f9168043a7ec351b80e08c9f168013ab68e99410aee2"
ARCHITECTURES = {"x86_64": "k8", "aarch64": "aarch64", "armv7l": "armv7a"}


def install_runtime(
    destination=Path("/usr/local/lib"), license_dir=Path("/usr/local/share/edgetpu")
):
    if platform.system() != "Linux" or platform.machine() not in ARCHITECTURES:
        raise RuntimeError("the packaged runtime requires supported Linux hardware")
    with urllib.request.urlopen(URL, timeout=60) as response:
        raw = response.read()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise RuntimeError("Coral runtime archive checksum mismatch")
    arch = ARCHITECTURES[platform.machine()]
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        library = archive.read(f"edgetpu_runtime/libedgetpu/throttled/{arch}/libedgetpu.so.1.0")
        license_text = archive.read("edgetpu_runtime/libedgetpu/LICENSE")
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "libedgetpu.so.1.0"
    target.write_bytes(library)
    target.chmod(0o755)
    (destination / "libedgetpu.so.1").symlink_to(target.name)
    license_dir.mkdir(parents=True, exist_ok=True)
    (license_dir / "LICENSE").write_bytes(license_text)
    print(f"Installed standard runtime ({arch}), SHA-256 {hashlib.sha256(library).hexdigest()}")


if __name__ == "__main__":
    install_runtime()
