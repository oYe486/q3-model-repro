"""Restore exact Q3 weights from a verified transfer initialization checkpoint.

The submission carries only the ten tensors changed by final fusion training.
The large initialization checkpoint is supplied as a GitHub Release asset (once
published) or via --source. This script never edits the source checkpoint.
"""

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "weights/manifest.json").read_text(encoding="utf-8"))
INIT = ROOT / "checkpoints/transfer_init/model.pth"
FINAL = ROOT / "checkpoints/final/model.pth"
BERT = ROOT / "pretrained/bert-base-uncased/pytorch_model.bin"
DELTA = ROOT / "weights/final_delta.pth"


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def verify(path, entry):
    path = Path(path)
    if path.stat().st_size != entry["bytes"]:
        raise ValueError("Unexpected size: %s" % path)
    actual = digest(path)
    if actual != entry["sha256"]:
        raise ValueError("SHA256 mismatch: %s (got %s)" % (path, actual))
    print("Verified", path)


def tensor_digest(state):
    """Hash names, dtypes, shapes and raw tensor values, independent of .pth ZIP metadata."""
    sha = hashlib.sha256()
    for name, value in state.items():
        for field in (name, str(value.dtype), str(tuple(value.shape))):
            sha.update(field.encode("utf-8"))
            sha.update(b"\0")
        sha.update(value.detach().contiguous().numpy().tobytes())
    return sha.hexdigest()


def download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    request = urllib.request.Request(url, headers={"User-Agent": "q3-model-reproduction/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
        shutil.copyfileobj(response, stream, 8 * 1024 * 1024)
    try:
        verify(temporary, MANIFEST["transfer_init"])
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Existing original transfer_init/model.pth")
    parser.add_argument("--url", help="GitHub Release asset URL; overrides manifest URL")
    parser.add_argument("--skip-bert", action="store_true", help="Use an existing local BERT directory")
    args = parser.parse_args()

    if args.source:
        source = args.source.resolve()
    else:
        local = ROOT.parent / "DLF-main/checkpoints/transfer_init/model.pth"
        source = local if local.is_file() else None
    if source is not None:
        verify(source, MANIFEST["transfer_init"])
        INIT.parent.mkdir(parents=True, exist_ok=True)
        if source != INIT.resolve():
            shutil.copyfile(source, INIT)
    elif not INIT.is_file():
        url = args.url or MANIFEST["release_asset_url"]
        if not url:
            parser.error("No source checkpoint or published Release URL is available yet; pass --source or --url")
        download(url, INIT)
    verify(INIT, MANIFEST["transfer_init"])

    base = torch.load(INIT, map_location="cpu")
    delta = torch.load(DELTA, map_location="cpu")
    if len(delta) != 10 or not set(delta).issubset(base):
        raise ValueError("Unexpected final-parameter delta")
    for name, value in delta.items():
        if base[name].shape != value.shape or base[name].dtype != value.dtype:
            raise ValueError("Parameter mismatch: " + name)
        base[name] = value
    if tensor_digest(base) != MANIFEST["final"]["tensor_sha256"]:
        raise ValueError("Restored final parameter values do not match the original model")
    FINAL.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, FINAL)
    print("Verified exact final tensor values:", FINAL)

    if not args.skip_bert:
        bert = {name[len("text_model.model."):]: value
                for name, value in base.items() if name.startswith("text_model.model.")}
        if len(bert) != 199:
            raise ValueError("Unexpected BERT parameter count")
        BERT.parent.mkdir(parents=True, exist_ok=True)
        torch.save(bert, BERT)
        print("Prepared local BERT loader weights:", BERT)
    print("Exact final model ready:", FINAL)


if __name__ == "__main__":
    main()
