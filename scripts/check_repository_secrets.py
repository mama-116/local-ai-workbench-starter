from __future__ import annotations

import argparse
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


_FORBIDDEN_SUFFIXES = (
    ".pem",
    ".key",
    ".pfx",
    ".p12",
    ".ppk",
    ".db",
    ".db-wal",
    ".db-shm",
    ".sqlite",
    ".sqlite-wal",
    ".sqlite-shm",
    ".sqlite3",
    ".sqlite3-wal",
    ".sqlite3-shm",
    ".log",
)
_LOG_DIRECTORIES = {"log", "logs", "personal-log", "personal-logs", "personal_logs"}
_PRIVATE_KEY_NAMES = {"id_rsa", "id_dsa", "id_ecdsa", "id_ed25519"}
_SAFE_ENV_TEMPLATES = {".env.example", ".env.template"}
_CONTENT_PATTERNS = (
    (
        "private key",
        re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    ),
    ("AWS access key", re.compile(rb"AKIA[0-9A-Z]{16}")),
    ("GitHub token", re.compile(rb"(?:ghp|github_pat)_[A-Za-z0-9_]{20,}")),
    ("OpenAI-style token", re.compile(rb"sk-[A-Za-z0-9_-]{20,}")),
)


@dataclass(frozen=True, slots=True)
class Finding:
    path: Path
    reason: str


def tracked_paths(root: Path) -> tuple[Path, ...]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return tuple(
        root / os.fsdecode(value)
        for value in result.stdout.split(b"\0")
        if value
    )


def scan_repository(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in tracked_paths(root):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        name = relative.name.lower()
        parts = {part.lower() for part in relative.parts[:-1]}
        filename_reason = _filename_reason(name, parts)
        if filename_reason is not None:
            findings.append(Finding(relative, filename_reason))
            continue
        payload = path.read_bytes()
        for reason, pattern in _CONTENT_PATTERNS:
            if pattern.search(payload):
                findings.append(Finding(relative, reason))
                break
    return findings


def _filename_reason(name: str, parent_parts: set[str]) -> str | None:
    if name == ".env" or (
        name.startswith(".env.") and name not in _SAFE_ENV_TEMPLATES
    ):
        return "environment file"
    if name.endswith(_FORBIDDEN_SUFFIXES):
        return "private key, database, or log filename"
    if name in _PRIVATE_KEY_NAMES:
        return "private key filename"
    if parent_parts & _LOG_DIRECTORIES:
        return "personal log directory"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail when tracked files contain secrets or private runtime data."
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        findings = scan_repository(root)
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"Repository secret scan could not run: {type(error).__name__}")
        return 2
    if findings:
        print("Repository secret scan failed. Remove these tracked files or values:")
        for finding in findings:
            print(f"- {finding.path.as_posix()}: {finding.reason}")
        return 1
    print("Repository secret scan passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
