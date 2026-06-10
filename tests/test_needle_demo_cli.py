"""Tests for the needle-demo Continue setup CLI (pure parts, no subprocess)."""

from needle_demo.cli import (
    _MANAGED_MARKER,
    _backup_if_unmanaged,
    build_parser,
    render_mcp_block,
    render_pydocs_overlay,
)


def test_overlay_selects_qwen3_via_sentence_transformers():
    overlay = render_pydocs_overlay()
    assert "provider: sentence_transformers" in overlay
    assert "model_name: Qwen/Qwen3-Embedding-0.6B" in overlay
    assert "dim: 1024" in overlay
    assert _MANAGED_MARKER in overlay


def test_mcp_block_has_metadata_and_project_only_flags(tmp_path):
    pydocs_bin = tmp_path / ".venv" / "bin" / "pydocs-mcp"
    block = render_mcp_block(
        "needle-docs",
        pydocs_bin=pydocs_bin,
        overlay=tmp_path / ".continue" / "pydocs-mcp.yaml",
        target_root=tmp_path,
    )
    # Standalone .continue/mcpServers/ blocks require name/version/schema.
    assert "name: needle-docs" in block
    assert "version:" in block
    assert "schema: v1" in block
    assert "type: stdio" in block
    assert str(pydocs_bin) in block
    assert "--skip-deps" in block
    assert "--no-inspect" in block
    # --config is a top-level pydocs-mcp flag: it must precede `serve`.
    assert block.index("--config") < block.index("- serve")


def test_backup_preserves_unmanaged_user_file(tmp_path):
    target = tmp_path / "config.yaml"
    target.write_text("user content\n", encoding="utf-8")
    backup = _backup_if_unmanaged(target)
    assert backup is not None
    assert backup.read_text(encoding="utf-8") == "user content\n"


def test_backup_skips_managed_file(tmp_path):
    target = tmp_path / "config.yaml"
    target.write_text(f"# {_MANAGED_MARKER}\n", encoding="utf-8")
    assert _backup_if_unmanaged(target) is None


def test_parser_wires_subcommands():
    args = build_parser().parse_args(["index", "--force"])
    assert args.force is True
    args = build_parser().parse_args(["init", "--playbook-root", "/tmp/pb"])
    assert args.playbook_root == "/tmp/pb"
