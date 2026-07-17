from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml"
SECRET_CHECK = REPOSITORY_ROOT / "scripts" / "check_repository_secrets.py"


def test_ci_runs_locked_quality_gates_with_read_only_permissions() -> None:
    assert WORKFLOW.is_file(), "PRとmain push向けCIがまだありません。"
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "pull_request:" in workflow
    assert "push:" in workflow and "main" in workflow
    assert "permissions:\n  contents: read" in workflow
    assert 'python-version: "3.12"' in workflow
    assert "uv run --frozen pytest" in workflow
    assert "uv run --frozen mypy" in workflow
    assert "check_repository_secrets.py" in workflow
    assert "actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd" in workflow
    assert "astral-sh/setup-uv@08807647e7069bb48b6ef5acd8ec9567f424441b" in workflow


def test_secret_check_rejects_private_files_without_printing_values(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    secret_value = "ghp_" + "A" * 30
    fixtures = {
        ".env": f"TOKEN={secret_value}",
        "identity.pem": "placeholder",
        "id_ed25519": "placeholder",
        "putty.ppk": "placeholder",
        "chat.sqlite3": "placeholder",
        "chat.db-wal": "placeholder",
        "personal.log": "placeholder",
        "notes.txt": secret_value,
        "key-material.txt": "-----BEGIN " + "PRIVATE KEY-----",
    }
    for name, content in fixtures.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "--", *fixtures], cwd=tmp_path, check=True)

    result = subprocess.run(
        [sys.executable, str(SECRET_CHECK), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert result.returncode == 1
    assert all(name in result.stdout for name in fixtures)
    assert secret_value not in result.stdout
