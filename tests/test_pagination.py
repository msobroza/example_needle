"""Tests for offset/limit pagination value objects."""

from __future__ import annotations

import pytest

from needle_core.domain.pagination import Page, PageRequest


def test_defaults() -> None:
    req = PageRequest()
    assert req.offset == 0
    assert req.limit == 10


def test_invalid_offset() -> None:
    with pytest.raises(ValueError):
        PageRequest(offset=-1)


def test_invalid_limit() -> None:
    with pytest.raises(ValueError):
        PageRequest(limit=0)
    with pytest.raises(ValueError):
        PageRequest(limit=-5)


def test_first_page_navigation() -> None:
    page = Page(items=list(range(10)), total=25, request=PageRequest(0, 10))
    assert page.has_prev is False
    assert page.has_next is True
    assert page.num_pages == 3
    assert page.page_number == 1


def test_middle_page_navigation() -> None:
    page = Page(items=list(range(10)), total=25, request=PageRequest(10, 10))
    assert page.has_prev is True
    assert page.has_next is True
    assert page.page_number == 2


def test_last_page_navigation() -> None:
    page = Page(items=list(range(5)), total=25, request=PageRequest(20, 10))
    assert page.has_prev is True
    assert page.has_next is False
    assert page.page_number == 3


def test_empty_result_set() -> None:
    page = Page(items=[], total=0, request=PageRequest(0, 10))
    assert page.num_pages == 0
    assert page.has_next is False
    assert page.has_prev is False


def test_exact_multiple_num_pages() -> None:
    page = Page(items=list(range(10)), total=20, request=PageRequest(0, 10))
    assert page.num_pages == 2
