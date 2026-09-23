"""Package a Linux onefile Node as the standard single-entrypoint release archive."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile
import tomllib


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, default=root / "dist/onefile/lerobot_infer")
    p.add_argument("--output-dir", type=Path, default=root / "dist/release")
    args = p.parse_args()
    binary = args.binary
    if binary.is_symlink() or not binary.is_file():
        p.error("--binary must be a regular onefile executable")
    with binary.open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            p.error("--binary must be a Linux ELF executable")
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    artifact_id = f"lerobot_infer-{version}-linux-x86_64"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / f"{artifact_id}.tar.gz"
    if archive.exists():
        p.error(f"archive already exists: {archive}")
    temporary = archive.with_suffix(archive.suffix + ".part")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=1) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as bundle:
                info = tarfile.TarInfo("lerobot_infer")
                info.size = binary.stat().st_size
                info.mode = 0o755
                info.mtime = 0
                with binary.open("rb") as stream:
                    bundle.addfile(info, stream)
    binary_hash = sha256(binary)
    with tarfile.open(temporary) as bundle:
        members = bundle.getmembers()
        if len(members) != 1 or members[0].name != "lerobot_infer" or not members[0].isfile():
            raise RuntimeError("release layout verification failed")
        h = hashlib.sha256()
        with bundle.extractfile(members[0]) as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                h.update(block)
        if h.hexdigest() != binary_hash:
            raise RuntimeError("archived executable checksum mismatch")
    temporary.rename(archive)
    metadata = dict(node_id="lerobot_infer", artifact_id=artifact_id, version=version,
                    platform="linux", arch="x86_64", artifact_type="executable_tar_gz",
                    entrypoint="lerobot_infer", sha256=sha256(archive),
                    size_bytes=archive.stat().st_size, binary_sha256=binary_hash)
    (args.output_dir / f"{artifact_id}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
