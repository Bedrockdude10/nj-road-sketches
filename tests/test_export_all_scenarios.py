"""export_all_scenarios.py clears the exports it is about to rewrite.

A file left over from an earlier run is indistinguishable from this run's output, so a
scenario that failed (or was removed) this time kept its old JSON and diff_exports reported it
unchanged. Clearing the directory by hand is what the tool exists to spare you.
"""
from scripts.export_all_scenarios import clear_previous_exports


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")
    return path


def test_the_sites_being_exported_lose_their_old_exports(tmp_path):
    stale = _touch(tmp_path / "site_a" / "geometry_removed_scenario.json")
    clear_previous_exports(tmp_path, ["site_a"])
    assert not stale.exists()


def test_other_sites_and_other_files_are_left_alone(tmp_path):
    other_site = _touch(tmp_path / "site_b" / "geometry_existing.json")
    not_an_export = _touch(tmp_path / "site_a" / "phase3_before_after.png")
    clear_previous_exports(tmp_path, ["site_a"])
    assert other_site.exists() and not_an_export.exists()


def test_it_names_what_it_left_from_an_earlier_run(tmp_path):
    _touch(tmp_path / "site_b" / "geometry_existing.json")
    assert clear_previous_exports(tmp_path, ["site_a"]) == ["site_b"]


def test_an_absent_directory_is_not_an_error(tmp_path):
    assert clear_previous_exports(tmp_path / "missing", ["site_a"]) == []
