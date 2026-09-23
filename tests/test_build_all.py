"""A 2D build leaves no geometry_*.json from an earlier run beside the sheets it just drew.

The JSON is the numbers the PNGs were drawn from. Written only on --render-3d, a 2D-only build
left the last 3D run's JSON next to fresh sheets, and a scenario that was removed or refused
kept its old file indefinitely - both read as this build's answer.
"""
import scripts.build_all as build_all
from tests.conftest import needs_source_data

SITE = "broad_st_greenwood"


@needs_source_data
def test_a_2d_build_rewrites_the_json_and_drops_what_it_did_not_write(tmp_path, monkeypatch):
    monkeypatch.setattr(build_all, "site_output_dir", lambda site: tmp_path)
    stale = tmp_path / "geometry_removed_scenario.json"
    stale.write_text("{}")

    failures, blender_jobs = build_all.build_site(SITE, render_3d=False, dpi=40)

    assert failures == [] and blender_jobs == []
    assert not stale.exists()
    assert (tmp_path / "geometry_existing.json").stat().st_size > 1000
