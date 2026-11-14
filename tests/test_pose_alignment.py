"""Pose alignment logic that runs without COLMAP, GeoCalib or LightGlue."""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from pose_alignment.align_poses import _apply_alignment
from pose_alignment.geo_common import RoiPaths
from data_ingest.pano import crop_axes
from pose_alignment.reconstruct import weak_ends

# A camera looking north: columns are its right (east), down and forward (north) axes in ENU.
LOOK_NORTH = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]
LINE = [[-83.7400, 42.2800], [-83.7390, 42.2800]]


def test_crop_axes_turns_the_view_right():
    R = crop_axes(90.0)
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(
        R[:, 2], [1.0, 0.0, 0.0], atol=1e-12
    )  # forward = the 360 camera's right
    np.testing.assert_allclose(crop_axes(-90.0)[:, 2], [-1.0, 0.0, 0.0], atol=1e-12)


def _matches_db(path, n, links):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE images (image_id INTEGER, name TEXT)")
    con.execute("CREATE TABLE two_view_geometries (pair_id INTEGER, rows INTEGER)")
    con.executemany(
        "INSERT INTO images VALUES (?, ?)", [(i + 1, f"f{i:02d}.jpg") for i in range(n)]
    )
    for (a, b), rows in links.items():
        i, j = sorted((a + 1, b + 1))
        con.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?)", (i * 2147483647 + j, rows)
        )
    con.commit()
    con.close()


def test_weak_ends_drops_a_tail_held_by_few_matches(tmp_path):
    n = 30
    links = {(k, k + 1): 300 for k in range(n - 1)} | {
        (k, k + 2): 200 for k in range(n - 2)
    }
    links |= {
        (26, 27): 20,
        (25, 27): 10,
        (26, 28): 10,
    }  # frames 27-29 hang on by a few matches
    _matches_db(tmp_path / "db.db", n, links)
    groups = [[f"f{i:02d}.jpg"] for i in range(n)]
    keep, dropped, info = weak_ends(tmp_path / "db.db", groups)
    assert dropped == ["f27.jpg", "f28.jpg", "f29.jpg"]
    assert keep == [f"f{i:02d}.jpg" for i in range(27)]
    assert info["back_frames_dropped"] == 3 and info["front_frames_dropped"] == 0


def test_weak_ends_keeps_an_evenly_matched_leg(tmp_path):
    n = 30
    _matches_db(tmp_path / "db.db", n, {(k, k + 1): 300 for k in range(n - 1)})
    keep, dropped, _ = weak_ends(
        tmp_path / "db.db", [[f"f{i:02d}.jpg"] for i in range(n)]
    )
    assert len(keep) == n and dropped == []


def _camera(img_id, file, lon, **extra):
    return {
        "file": file,
        "id": img_id,
        "leg": "leg01",
        "lon": lon,
        "lat": 42.2800,
        "height_m": 2.1,
        "rotation_cam_to_enu": LOOK_NORTH,
        "intrinsics": {
            "model": "SIMPLE_RADIAL",
            "width": 1000,
            "height": 750,
            "fx": 800.0,
        },
        "leg_fit_m": 0.8,
        "reliable": True,
        **extra,
    }


def _roi(tmp_path, alignment):
    paths = RoiPaths(tmp_path / "mapillary_roi_1")
    paths.alignment.mkdir(parents=True)
    paths.images.mkdir(parents=True)
    paths.roi_json.write_text(json.dumps({"lines": [LINE], "corridor_m": 15}))
    paths.sequences_json.write_text(json.dumps([{"folder": f} for f in alignment]))
    for folder, cams in alignment.items():
        (paths.alignment / f"{folder}.json").write_text(
            json.dumps({"folder": folder, "cameras": cams})
        )
    (paths.alignment / "summary.json").write_text("{}")  # not a sequence: skipped
    return paths


def test_apply_alignment_overrides_and_adds_360_crops(tmp_path):
    crop = {
        "intrinsics": {
            "model": "SIMPLE_RADIAL",
            "width": 960,
            "height": 960,
            "fx": 480.0,
        }
    }
    paths = _roi(
        tmp_path,
        {
            "line1_cam0_aaaaaaaa": [_camera("111", "frame_0001.jpg", -83.7398)],
            # a 360 frame's right and left crops share the Mapillary id
            "line2_cam0_pppppppp": [
                _camera("222", "frame_0001.jpg", -83.7395, pano_yaw_deg=90.0, **crop)
            ],
            "line2_cam1_pppppppp": [
                _camera("222", "frame_0001.jpg", -83.7395, pano_yaw_deg=-90.0, **crop)
            ],
        },
    )
    mapillary = {
        "111": {
            "latitude": 42.2801,
            "longitude": -83.7398,
            "original_width": 4000,
            "original_height": 3000,
        },
        # the fetch stage's own views of the 360 photo (data_ingest.pano), Mapillary's pose
        **{
            f"222_{side}": {
                "latitude": 42.2801,
                "longitude": -83.7395,
                "original_width": 1440,
                "original_height": 1440,
                "pano_crop": {"id": "222", "side": side},
            }
            for side in ("right", "left")
        },
    }
    summary = _apply_alignment(paths, mapillary)

    assert summary["overridden"] == 1 and summary["added"] == 2
    assert mapillary["111"]["pose_source"] == "colmap"
    assert (
        mapillary["111"]["focal_length_pixels"] == 800.0 * 4000 / 1000
    )  # scaled to the original size
    # the aligned views replace the fetch stage's unaligned ones
    assert "222_right" not in mapillary and "222_left" not in mapillary
    for side in ("line2_cam0_pppppppp", "line2_cam1_pppppppp"):
        added = mapillary[f"222_{side}"]
        assert Path(added["image_url"]).parts[-2:] == (side, "frame_0001.jpg")
        assert added["original_width"] == 960 and added["focal_length_pixels"] == 480.0
    overrides = json.loads((paths.root / "pose_overrides.json").read_text())
    assert overrides["images"]["111"]["leg_fit_to_walls_m"] == 0.8


def test_apply_alignment_skips_unreliable_and_off_street_legs(tmp_path):
    far = _camera("333", "frame_0001.jpg", -83.7398)
    far["lat"] = 42.2810  # about 110 m north of the street
    paths = _roi(
        tmp_path,
        {
            "line1_cam0_aaaaaaaa": [
                _camera("111", "frame_0001.jpg", -83.7398, reliable=False)
            ],
            "line1_cam1_bbbbbbbb": [far],
        },
    )
    mapillary = {
        i: {
            "latitude": 42.28,
            "longitude": -83.7398,
            "original_width": 1000,
            "original_height": 750,
        }
        for i in ("111", "333")
    }
    summary = _apply_alignment(paths, mapillary)
    assert summary["overridden"] == 0 and summary["rejected_legs_off_street_m"]
    assert all("pose_source" not in m for m in mapillary.values())


def test_pano_views_turn_a_360_photo_into_right_and_left_views():
    from data_ingest.pano import crop_rotvec, pano_views

    # a 360 sequence driving east; the image centre looks along travel
    records = [
        {
            "id": str(k),
            "sequence": "s360",
            "camera_type": "spherical",
            "captured_at": k * 1000,
            "computed_compass_angle": 90.0,
            "width": 5760,
            "geometry": {"coordinates": [-83.74 + k * 1e-4, 42.28]},
        }
        for k in range(3)
    ] + [{"id": "9", "camera_type": "perspective", "geometry": {"coordinates": [0, 0]}}]
    rot = [0.3, -1.1, 0.4]
    metadata = {
        r["id"]: {
            "rotvec": rot,
            "original_width": r.get("width"),
            "camera_type": r["camera_type"],
        }
        for r in records
    }
    out = pano_views(metadata, records)
    assert (
        "1" not in out and "9" in out
    )  # the 360 photo is replaced, a perspective one kept
    right, left = out["1_right"], out["1_left"]
    assert right["pano_crop"]["yaw_deg"] == pytest.approx(90.0, abs=1)
    assert left["pano_crop"]["yaw_deg"] == pytest.approx(-90.0, abs=1)
    assert (
        right["original_width"] == right["original_height"] == 1440
    )  # 5760 * 90 / 360
    assert right["focal_length_pixels"] == pytest.approx(720.0)  # 90 deg across
    np.testing.assert_allclose(
        right["rotvec"], crop_rotvec(rot, right["pano_crop"]["yaw_deg"])
    )


def test_crop_view_looks_where_its_yaw_says():
    from data_ingest.pano import crop_view

    # an equirectangular image whose blue channel is a quarter of its column
    pano = np.zeros((400, 800, 3), np.uint8)
    pano[..., 0] = (np.arange(800) // 4)[None, :]
    centre = lambda yaw: int(crop_view(pano, yaw, 90.0, 100)[50, 50, 0])  # noqa: E731
    assert abs(centre(90.0) - 150) <= 1  # +90 deg: column 600, three quarters across
    assert abs(centre(-90.0) - 50) <= 1  # -90 deg: column 200, one quarter across


def _street(cross_every=(25.0, 40.0, 30.0, 45.0, 35.0)):
    """Walls of a straight street running north (x = -10 and x = +10, y 0..200) broken by cross
    streets at irregular spacing (cross walls at each gap); points on the walls every 0.5 m."""
    import numpy as np

    A, B, P = [], [], []
    y = 0.0
    for gap in cross_every:
        for x in (-10.0, 10.0):
            A.append([x, y]), B.append([x, y + gap - 8])
            P += [[x, v] for v in np.arange(y, y + gap - 8, 0.5)]
            side = -1 if x < 0 else 1
            A.append([x, y + gap - 8]), B.append([x + side * 15, y + gap - 8])
            P += [[x + side * u, y + gap - 8] for u in np.arange(0, 15, 0.5)]
        y += gap
    cams = np.c_[np.zeros(40), np.linspace(5, y - 5, 40)]
    return np.array(A, float), np.array(B, float), np.array(P, float), cams


def _checks(P, A, B, cams):

    from pose_alignment.align import OrientedWalls, facade_dirs, placement_checks
    from pose_alignment.osm import WallMap

    lo, hi = cams.min(0) - 120, cams.max(0) + 120
    return placement_checks(
        P,
        facade_dirs(P, 2.0),
        cams,
        A,
        B,
        WallMap(A, B, [], lo, hi),
        OrientedWalls(A, B, lo, hi),
    )


def test_a_placed_leg_on_its_walls_is_reliable():
    from pose_alignment.align import verdict

    A, B, P, cams = _street()
    chk = _checks(P, A, B, cams)
    assert chk["near_median_m"] < 0.3 and chk["near_explained_share"] > 0.9
    assert abs(chk["along_best_shift_m"]) <= 1 and chk["along_margin"] > 0.1
    assert verdict(chk, 0.0, [], 2.0) == []


def test_long_uniform_frontages_are_ambiguous_along_the_street():
    import numpy as np

    from pose_alignment.align import verdict

    # only the two long frontages, no cross walls: nothing holds the leg along the street
    A = np.array([[-10.0, 0.0], [10.0, 0.0]])
    B = np.array([[-10.0, 200.0], [10.0, 200.0]])
    P = np.array([[x, v] for x in (-10.0, 10.0) for v in np.arange(20, 180, 0.5)])
    cams = np.c_[np.zeros(40), np.linspace(5, 195, 40)]
    fails = verdict(_checks(P, A, B, cams), 0.0, [], 2.0)
    assert any("ambiguous along the street" in f for f in fails)


def test_verdict_names_each_failed_check():
    from pose_alignment.align import verdict

    chk = {
        "near_points": 10,
        "near_explained_share": 0.3,
        "near_median_m": 2.5,
        "along_best_shift_m": 12.0,
        "along_margin": 0.5,
        "along_second_shift_m": -20.0,
    }
    fails = verdict(chk, 0.3, ["frame_0002.jpg"], 2.0)
    assert len(fails) == 6  # points, unexplained, fit, better placement, inside, order


def test_facade_heading_finds_the_turn_between_model_and_map():
    import math

    from pose_alignment.align import facade_dirs, facade_heading, rotate_dirs

    A, B, P, cams = _street()
    # the model is the map turned by -2 deg: its facade directions must be turned +2 deg back
    D = rotate_dirs(facade_dirs(P, 2.0), math.radians(-2.0))
    ang, info = facade_heading(D, cams, 0.0, A, B)
    assert ang is not None and abs(math.degrees(ang) - 2.0) < 0.3, info


def test_geocalib_looks_unreliable_for_a_rig_with_scattered_focals():
    from pose_alignment.geo_common import calibration_unreliable

    class _Roi:
        def frames(self, folder):
            make = "Apple" if folder == "phone" else "Suzuki"
            return {"frame_0001.jpg": {"make": make, "model": "DL1000"}}

    def doc(f, not_converged=0):
        return {
            "focal_norm": f,
            "width": 1920,
            "height": 1080,
            "sample_frames": list(range(10)),
            "sample_not_converged": not_converged,
        }

    docs = {"rig_a": doc(0.57), "rig_b": doc(0.78), "phone": doc(0.80, not_converged=7)}
    reasons = calibration_unreliable(_Roi(), docs)
    assert set(reasons) == {"rig_a", "rig_b", "phone"}
    assert "rig cameras got focals 0.57-0.78" in reasons["rig_a"][0]
    assert "no convergence on 7/10" in reasons["phone"][0]
    assert (
        calibration_unreliable(_Roi(), {"rig_a": doc(0.60), "rig_b": doc(0.62)}) == {}
    )


def test_fold_keep_keeps_the_longest_stretch():
    """A camera behind the one before cuts the leg; the longest stretch is kept if long enough."""
    from pose_alignment.align import backward_steps, fold_keep

    y = np.r_[
        np.arange(21.0), [19.0], np.arange(20.0, 28.0)
    ]  # frame 22 placed behind 21
    Lc = np.c_[np.zeros(len(y)), y, np.zeros(len(y))]
    assert backward_steps(Lc).tolist() == [21]
    assert fold_keep(Lc) == (0, 21)
    assert fold_keep(Lc[:12]) is None  # its first 12 cameras hold no fold
    straight = np.c_[np.zeros(10), np.arange(10.0), np.zeros(10)]
    assert fold_keep(straight) is None


def test_gps_check_flags_broken_computed_tracks():
    """Measured GPS where the computed track's length is off, or rig cameras disagree."""
    from pose_alignment.align import gps_check

    def frames(n, step_c, step_m, t0=0):
        return {
            f"frame_{i:04d}.jpg": {
                "lon": -83.74 + i * step_c / 82000,
                "lat": 42.28,
                "gps": [-83.74 + i * step_m / 82000, 42.28],
                "captured_at_ms": t0 + i * 1000,
            }
            for i in range(n)
        }

    class _P:
        def __init__(self, data):
            self.data = data

        def frames(self, folder):
            return self.data[folder]

    good = {"a": frames(20, 5.0, 5.1)}
    leg = {"cameras": [{"folder": "a", "frame_files": list(good["a"])}]}
    assert gps_check(_P(good), leg) == ("computed", [])
    short = {"a": frames(20, 5.0, 7.0)}  # measured 40% longer
    src, why = gps_check(_P(short), leg)
    assert src == "measured" and "measured GPS" in why[0]
    rig = {
        "a": frames(20, 5.0, 5.0),
        "b": frames(20, 3.5, 3.5),
    }  # same time, 30% shorter
    leg2 = {"cameras": [{"folder": f, "frame_files": list(rig[f])} for f in "ab"]}
    src, why = gps_check(_P(rig), leg2)
    assert src == "measured" and "rig cameras" in why[0]


def test_alignment_args_fixed_method():
    """The method is fixed (run_pipeline.py's defaults); options from older jobs are ignored."""
    from pose_alignment.align_poses import alignment_args

    assert alignment_args({}) == []
    assert alignment_args(None) == []
    assert (
        alignment_args(
            {"features": "sift", "alignment": "previous", "focal_rule": False}
        )
        == []
    )


def test_steps_against_track_tells_u_turns_from_folds():
    """A capture that turns back is no fold; a stretch slid back while the capture moved on is."""
    from pose_alignment.geo_common import plan_onto, steps_against_track

    out_and_back = np.r_[
        np.c_[np.arange(10.0), np.zeros(10)], np.c_[np.arange(9.0, -1, -1), np.ones(10)]
    ]
    gps = out_and_back * 3.0  # the same walk in metres
    model = np.c_[out_and_back, np.zeros(len(out_and_back))]  # 3D, another scale
    assert steps_against_track(plan_onto(model, gps), gps) == []

    straight = np.c_[np.arange(20.0) * 3, np.zeros(20)]
    folded = straight.copy()
    folded[12:, 0] -= 10.0  # frames 12.. slid back while the capture kept going
    assert steps_against_track(plan_onto(folded, straight), straight) == [12]


def test_jump_steps_and_fold_keep_cut_a_sideways_jump():
    """A camera placed far to the side while the capture moved on a little is cut off like a fold;
    a steady walk, and a faster stretch the GPS also shows, are not."""
    from pose_alignment.align import fold_keep, jump_steps

    gps = np.c_[np.arange(20.0) * 1.5, np.zeros(20)]  # 1.5 m steps
    model = np.c_[gps / 3.0, np.zeros(20)]  # the same walk at another scale
    assert jump_steps(model, gps).tolist() == []
    assert fold_keep(model, gps) is None

    jumped = model.copy()
    jumped[:3, 1] += 4.0  # the first three cameras 12 m to the side
    assert jump_steps(jumped, gps).tolist() == [3]
    assert fold_keep(jumped, gps) == (3, 20)

    fast = gps.copy()
    fast[10:, 0] += np.arange(10) * 4.5  # the walker ran: 6 m steps, in the GPS too
    assert jump_steps(np.c_[fast / 3.0, np.zeros(20)], fast).tolist() == []
