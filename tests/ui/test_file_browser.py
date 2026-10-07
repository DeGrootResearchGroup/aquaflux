"""Tests for the file browser's directory listing."""

from __future__ import annotations

import pytest

pytest.importorskip("trame")

from aquaflux_ui.file_browser import list_directory


def test_folders_come_first_then_wanted_files_each_by_name_hidden_ones_left_out(tmp_path):
    for name in ("b", "A", ".hidden"):
        (tmp_path / name).mkdir()
    for name in ("z.yaml", "a.YML", "notes.txt", ".secret.yaml"):
        (tmp_path / name).write_text("")
    listing = list_directory(tmp_path, (".yaml", ".yml"))
    assert [(e["name"], e["is_dir"]) for e in listing.entries] == [
        ("A", True), ("b", True), ("a.YML", False), ("z.yaml", False),
    ]  # fmt: skip
    assert listing.error is None
    assert listing.crumbs[-1] == {"name": tmp_path.name, "path": str(tmp_path.resolve())}
    assert listing.crumbs[0]["path"] == tmp_path.resolve().anchor


def test_a_file_is_listed_as_the_directory_it_is_in(tmp_path):
    (tmp_path / "case.yaml").write_text("")
    assert list_directory(tmp_path / "case.yaml", (".yaml",)).directory == tmp_path.resolve()


def test_a_directory_that_cannot_be_read_says_so(tmp_path):
    listing = list_directory(tmp_path / "absent", (".yaml",))
    assert listing.entries == [] and "cannot be read" in listing.error
