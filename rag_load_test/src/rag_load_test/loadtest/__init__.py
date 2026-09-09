"""Locust-free helpers used by ``loadtest/locustfile.py``.

Auth headers, per-stage events and the question bank live here so they can be
unit-tested without importing locust (which gevent-patches the interpreter).
"""
