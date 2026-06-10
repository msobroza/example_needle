"""needle-demo — one-command Continue.dev demo setup for example_needle.

``needle-demo init`` wires three things into the workspace:

1. ``.continue/pydocs-mcp.yaml`` — pydocs-mcp config overlay selecting the
   ``Qwen/Qwen3-Embedding-0.6B`` dense embedder (``sentence_transformers``
   provider, dim 1024).
2. coding-agent-playbook — ``cap init`` (first run) or ``cap sync`` (re-runs)
   generates Continue rules/prompts under ``.continue/`` plus ``AGENTS.md``.
3. ``.continue/mcpServers/{needle-docs,playbook-docs}.yaml`` — Continue MCP
   block files declaring two pydocs-mcp servers: a project-only index of
   example_needle and one of the coding-agent-playbook checkout
   (``--skip-deps --no-inspect``).

``needle-demo index`` pre-builds both indexes so the first Continue launch is
instant (the MCP servers re-index on startup, hitting the package-level cache).

Stdlib-only; the heavy work happens in the ``pydocs-mcp`` / ``cap``
executables installed into this venv by ``uv sync --group demo``.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# src/needle_demo/cli.py -> repo root. Valid for the editable install this
# demo uses (`uv sync` installs the project editable); --root overrides.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLAYBOOK_RELATIVE_DEFAULT = "../coding-agent-playbook/coding-agent-playbook"
_MANAGED_MARKER = "Managed by `needle-demo init`"
_SYNC_HINT = "run `uv sync --group demo` in the example_needle root first"

_OVERLAY_TEMPLATE = """\
# {marker} — pydocs-mcp config overlay for the Continue demo.
# Passed to every demo server / index invocation via `pydocs-mcp --config`.
# Full reference of tunables: pydocs-mcp's shipped default_config.yaml.
embedding:
  provider: sentence_transformers
  model_name: Qwen/Qwen3-Embedding-0.6B
  dim: 1024
  batch_size: 8
"""

# Workspace MCP block file (.continue/mcpServers/<name>.yaml) — the
# documented repo-level mechanism for MCP servers, sibling to the
# cap-generated .continue/rules/ and .continue/prompts/ blocks. Models stay
# in the user's global ~/.continue config.
_MCP_BLOCK_TEMPLATE = """\
# {marker} — re-running init overwrites this file.
name: {server_name}
version: 0.0.1
schema: v1

mcpServers:
  - name: {server_name}
    type: stdio
    command: {pydocs_bin}
    args:
      - --config
      - {overlay}
      - serve
      - {target_root}
      - --skip-deps
      - --no-inspect
"""


def render_pydocs_overlay() -> str:
    """The pydocs-mcp YAML overlay (Qwen3 embeddings) written by ``init``."""
    return _OVERLAY_TEMPLATE.format(marker=_MANAGED_MARKER)


def render_mcp_block(
    server_name: str,
    pydocs_bin: Path,
    overlay: Path,
    target_root: Path,
) -> str:
    """One ``.continue/mcpServers/<server_name>.yaml`` block (absolute paths)."""
    return _MCP_BLOCK_TEMPLATE.format(
        marker=_MANAGED_MARKER,
        server_name=server_name,
        pydocs_bin=pydocs_bin,
        overlay=overlay,
        target_root=target_root,
    )


def _venv_bin(name: str) -> Path | None:
    """Locate ``name`` next to the running interpreter, else on PATH."""
    candidate = Path(sys.executable).with_name(name)
    if candidate.exists():
        return candidate
    found = shutil.which(name)
    return Path(found) if found else None


def _require_bin(name: str) -> Path:
    binary = _venv_bin(name)
    if binary is None:
        raise SystemExit(f"error: `{name}` executable not found — {_SYNC_HINT}.")
    return binary


def _backup_if_unmanaged(path: Path) -> Path | None:
    """Back up an existing file we did not generate before overwriting it."""
    if not path.exists():
        return None
    if _MANAGED_MARKER in path.read_text(encoding="utf-8"):
        return None
    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)
    return backup


def _run(cmd: list[str]) -> int:
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=False).returncode


def _resolve_roots(args: argparse.Namespace) -> tuple[Path, Path]:
    root = Path(args.root).resolve()
    playbook = (
        Path(args.playbook_root).resolve()
        if args.playbook_root
        else (root / _PLAYBOOK_RELATIVE_DEFAULT).resolve()
    )
    return root, playbook


def cmd_init(args: argparse.Namespace) -> int:
    root, playbook_root = _resolve_roots(args)
    if not playbook_root.is_dir():
        print(
            f"error: coding-agent-playbook checkout not found at {playbook_root}\n"
            "       point at it with --playbook-root",
            file=sys.stderr,
        )
        return 2
    pydocs_bin = _require_bin("pydocs-mcp")
    cap_bin = _require_bin("cap")

    continue_dir = root / ".continue"
    continue_dir.mkdir(exist_ok=True)

    # 1. pydocs-mcp overlay (Qwen3 embeddings, shared by both MCP servers).
    overlay = continue_dir / "pydocs-mcp.yaml"
    overlay.write_text(render_pydocs_overlay(), encoding="utf-8")
    print(f"wrote {overlay}")

    # 2. Playbook artifacts. `cap init` bootstraps once (and auto-runs sync);
    #    re-runs go through `cap sync` so user edits to the cap config stick.
    if (root / ".coding-agent-playbook.toml").exists():
        rc = _run([str(cap_bin), "sync", "--repo-root", str(root)])
    else:
        rc = _run(
            [
                str(cap_bin),
                "init",
                "--repo-root",
                str(root),
                "--non-interactive",
                "--name",
                "example-needle",
                "--kind",
                "python-library",
                "--adapters",
                "continue",
                "--description",
                "Page-level multimodal document retrieval demo",
            ]
        )
    if rc != 0:
        print("error: coding-agent-playbook bootstrap failed", file=sys.stderr)
        return rc

    # 3. One Continue MCP block file per server.
    blocks_dir = continue_dir / "mcpServers"
    blocks_dir.mkdir(exist_ok=True)
    servers = [("needle-docs", root), ("playbook-docs", playbook_root)]
    for server_name, target_root in servers:
        block_path = blocks_dir / f"{server_name}.yaml"
        backup = _backup_if_unmanaged(block_path)
        if backup is not None:
            print(f"existing unmanaged {block_path.name} backed up to {backup.name}")
        block_path.write_text(
            render_mcp_block(server_name, pydocs_bin, overlay, target_root),
            encoding="utf-8",
        )
        print(f"wrote {block_path}")

    print(
        "\nDemo configured. Next steps:\n"
        "  1. uv run needle-demo index   # pre-build both indexes (first run\n"
        "     downloads Qwen/Qwen3-Embedding-0.6B from Hugging Face)\n"
        "  2. Open example_needle in VS Code and reload Continue — the\n"
        "     needle-docs and playbook-docs MCP servers expose search/lookup,\n"
        "     and the generated rules/prompts are picked up automatically."
    )
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    root, playbook_root = _resolve_roots(args)
    pydocs_bin = _require_bin("pydocs-mcp")
    overlay = root / ".continue" / "pydocs-mcp.yaml"
    if not overlay.exists():
        print(
            f"error: {overlay} missing — run `needle-demo init` first.",
            file=sys.stderr,
        )
        return 2

    targets = [("example_needle", root), ("coding-agent-playbook", playbook_root)]
    for label, target in targets:
        print(f"\n=== indexing {label} ({target}) ===")
        cmd = [
            str(pydocs_bin),
            "--config",
            str(overlay),
            "index",
            str(target),
            "--skip-deps",
            "--no-inspect",
        ]
        if args.force:
            cmd.append("--force")
        rc = _run(cmd)
        if rc != 0:
            print(f"error: indexing {label} failed (exit {rc})", file=sys.stderr)
            return rc
    print("\nBoth indexes built — Continue MCP startups will hit the cache.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="needle-demo",
        description="Continue.dev demo setup: pydocs-mcp MCP servers + "
        "coding-agent-playbook rules for example_needle.",
    )
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--root",
        default=str(_REPO_ROOT),
        help="example_needle repo root (default: auto-detected)",
    )
    shared.add_argument(
        "--playbook-root",
        default=None,
        help="coding-agent-playbook checkout "
        f"(default: <root>/{_PLAYBOOK_RELATIVE_DEFAULT})",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser(
        "init",
        parents=[shared],
        help="one-shot setup: Continue config + MCP overlay + playbook init",
    )
    p_init.set_defaults(func=cmd_init)

    p_index = sub.add_parser(
        "index",
        parents=[shared],
        help="pre-build both pydocs-mcp indexes (project-only, Qwen3)",
    )
    p_index.add_argument(
        "--force", action="store_true", help="clear caches and re-index"
    )
    p_index.set_defaults(func=cmd_index)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
