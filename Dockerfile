# syntax=docker/dockerfile:1
# Minimal image for running the `needle` CLI and the dependency-light core.
# The heavy model backends are optional; install with the [retrieval] extra
# (and [pdf] for rasterising PDFs) if you need them.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --upgrade pip && python -m pip install .

# Run as a non-root user.
RUN useradd --create-home --uid 10001 needle
USER needle

ENTRYPOINT ["needle"]
CMD ["--help"]
