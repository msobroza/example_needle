"""Continue.dev demo wiring for example_needle.

One-command setup (``needle-demo init``) for the Continue workspace config,
the pydocs-mcp MCP servers (project-only indexes of example_needle and the
coding-agent-playbook checkout, Qwen3 embeddings), and the coding-agent-playbook
rules/prompts. Stdlib-only by design — the heavy lifting is delegated to the
``pydocs-mcp`` and ``cap`` executables installed by the ``demo`` dependency
group (``uv sync --group demo``).
"""
