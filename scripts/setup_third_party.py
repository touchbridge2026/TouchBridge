#!/usr/bin/env python3
"""Fetch pinned upstream repositories and apply TouchBridge's ProPainter patch."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def git(*args, cwd=None, check=True):
    result = subprocess.run(["git", *map(str, args)], cwd=cwd,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result


def main():
    versions = json.loads((ROOT / "third_party/versions.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("components", nargs="*", metavar="COMPONENT", help="sam2, sam3, ProPainter; default: all")
    args = parser.parse_args()
    args.components = args.components or list(versions)
    if set(args.components) - set(versions):
        parser.error("Unknown component; choose sam2, sam3, or ProPainter")
    parent = ROOT / ".third_party"
    parent.mkdir(exist_ok=True)
    for name in args.components:
        spec = versions[name]
        path = parent / name
        if not path.exists():
            print(f"Fetching {name} @ {spec['commit']}", flush=True)
            git("init", path)
            git("remote", "add", "origin", spec["url"], cwd=path)
            git("fetch", "--depth", "1", "origin", spec["commit"], cwd=path)
            git("checkout", "--detach", "FETCH_HEAD", cwd=path)
        actual = git("rev-parse", "HEAD", cwd=path).stdout.strip()
        if actual != spec["commit"]:
            raise RuntimeError(f"{path}: expected {spec['commit']}, found {actual}; refusing to overwrite")
        if spec.get("patch"):
            patch = ROOT / spec["patch"]
            if git("apply", "--reverse", "--check", patch, cwd=path, check=False).returncode == 0:
                print(f"{name}: patch already present")
            else:
                git("apply", "--check", patch, cwd=path)
                git("apply", patch, cwd=path)
        print(f"Ready: {path}")


if __name__ == "__main__":
    main()
