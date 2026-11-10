#!/usr/bin/env python3
"""Block commits and pushes that contain an API key.

Usage: check_secrets.py [files...]   (no arguments: every file tracked by git)
Exit code 1 and a list of offending lines if anything that looks like a secret is found.
"""
import re
import subprocess
import sys
from pathlib import Path

PATTERNS = [
    ("Anthropic API key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("OpenAI-style key", re.compile(r"\bsk-[A-Za-z0-9]{32,}\b")),
    ("hard-coded api_key", re.compile(r"""api[_-]?key\s*[=:]\s*["'][A-Za-z0-9_\-]{20,}["']""", re.I)),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("private key", re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----")),
]
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".ico", ".woff", ".woff2"}


def files_to_scan(args):
    if args:
        return [Path(a) for a in args]
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout
    return [Path(p) for p in out.splitlines()]


def main(argv):
    hits = []
    for path in files_to_scan(argv):
        if path.suffix.lower() in SKIP_SUFFIXES or not path.is_file():
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for name, pat in PATTERNS:
                if pat.search(line):
                    hits.append(f"{path}:{lineno}: looks like a {name}")
    if hits:
        print("Possible secrets found. Remove them before committing:\n  " + "\n  ".join(hits))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
