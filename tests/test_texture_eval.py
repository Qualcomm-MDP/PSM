"""The UV texture quality evaluation (src/evaluation/texture_uv.py): obstruction,
distortion, FID and the confidence, on synthetic inputs with known answers. No network:
Inception-v3 is replaced by a stand-in feature extractor."""

import json
import math
import sys

import numpy as np
import pytest
import trimesh
from PIL import Image
from trimesh import visual

from common.pipeline_chain import PipelineState
from evaluation import (
    evaluate_texture_model,
    evaluate_texture_stage,
    texture_uv as tu,
)

# ---- the score ----


def test_weights_must_be_non_negative_and_sum_to_one():
    tu.check_weights(0.35, 0.25, 0.40)
    with pytest.raises(ValueError):
        tu.check_weights(0.5, 0.5, 0.5)
    with pytest.raises(ValueError):
        tu.check_weights(-0.1, 0.6, 0.5)


def test_confidence_and_its_renormalisation_when_f_is_missing():
    w = {"O": 0.35, "D": 0.25, "F": 0.40}
    conf, partial, used = tu.confidence({"O": 0.2, "D": 0.1, "F": 0.5}, w)
    assert conf == pytest.approx(1 - (0.35 * 0.2 + 0.25 * 0.1 + 0.40 * 0.5))
    assert not partial and used == w

    conf, partial, used = tu.confidence({"O": 0.2, "D": 0.1, "F": None}, w)
    assert partial and set(used) == {"O", "D"}
    assert used["O"] == pytest.approx(0.35 / 0.60)
    assert used["D"] == pytest.approx(0.25 / 0.60)
    assert conf == pytest.approx(1 - (used["O"] * 0.2 + used["D"] * 0.1))


def test_f_maps_fid_to_zero_one():
    assert tu.f_term(0.0, 100.0) == 0.0
    assert tu.f_term(100.0, 100.0) == pytest.approx(1 - math.exp(-1))
    assert tu.f_term(1e9, 100.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        tu.f_term(10.0, 0.0)


def test_fid_matches_the_closed_form():
    rng = np.random.default_rng(1)
    f1 = rng.normal(size=(500, 6))
    assert tu.frechet_distance(f1, f1) == pytest.approx(0.0, abs=1e-6)
    # shifted: same covariance, so FID = |mu|^2
    mu = np.array([1.0, -2.0, 0.5, 0, 0, 3.0])
    assert tu.frechet_distance(f1, f1 + mu) == pytest.approx(mu @ mu, rel=1e-6)
    # scaled by 2: S2 = 4 S1, sqrt(S1 S2) = 2 S1, so FID = |mu|^2 + tr(S1)
    c = f1 - f1.mean(0)
    s1 = np.cov(f1, rowvar=False)
    assert tu.frechet_distance(f1, 2 * c + f1.mean(0)) == pytest.approx(
        np.trace(s1), rel=1e-5
    )


# ---- distortion ----

SQUARE = np.array([[0, 0, 0], [10, 0, 0], [10, 0, 10], [0, 0, 10.0]])  # a 10 m wall
TWO = np.array([[0, 1, 2], [0, 2, 3]])


def test_an_isotropic_map_has_no_stretch_and_a_stretched_one_does():
    tex = SQUARE[:, [0, 2]] * 7.0  # 7 texels a metre, both ways
    s = tu.stretch(SQUARE, tex, TWO)
    assert s == pytest.approx([1.0, 1.0])

    tex2 = tex.copy()
    verts = np.vstack([SQUARE, SQUARE + [20, 0, 0]])
    tex2 = np.vstack([tex, tex * [2.0, 1.0] + [300, 0]])  # second wall: 2x along it
    faces = np.vstack([TWO, TWO + 4])
    s = tu.stretch(verts, tex2, faces)
    assert s == pytest.approx([1, 1, 2, 2])
    area = tu.face_areas(verts, faces)
    d, bad, use = tu.distortion(area, s, np.ones(4, bool), 1.5)
    assert d == pytest.approx(0.5) and bad.tolist() == [False, False, True, True]


def test_degenerate_triangles():
    verts = np.vstack([SQUARE, [[0, 0, 0], [5, 0, 0], [10, 0, 0]]])  # last: no 3D area
    tex = np.vstack([SQUARE[:, [0, 2]] * 7.0, [[0, 0], [1, 1], [2, 2]]])
    # a chart triangle of zero area (corners in a line), and a 3D-degenerate one
    tex_line = tex.copy()
    tex_line[1] = [35, 35]  # face 0's corners 0, 1, 2 now on a line
    faces = np.vstack([TWO, [[4, 5, 6]]])
    s = tu.stretch(verts, tex_line, faces)
    assert np.isinf(s[0]) and s[1] == pytest.approx(1.0) and np.isnan(s[2])
    area = tu.face_areas(verts, faces)
    d, bad, use = tu.distortion(area, s, np.ones(3, bool), 1.5)
    assert bad.tolist() == [True, False, False] and not use[2]
    assert d == pytest.approx(0.5)

    # all three UVs at one point: no chart (texturing's untextured faces)
    collapsed = tex.copy()
    collapsed[[0, 1, 2]] = [3.0, 3.0]
    assert tu.no_chart(collapsed, faces).tolist() == [True, False, False]


# ---- obstruction ----


def test_coverage_counts_texels_per_triangle_and_no_chart_as_missing():
    tex = SQUARE[:, [0, 2]] * 10.0  # 100 x 100 texels
    ids = tu.chart_raster(tex, TWO, (100, 100))
    assert (ids >= 0).sum() == pytest.approx(100 * 100, rel=0.03)
    valid = np.zeros((100, 100), bool)
    valid[:, :50] = True  # the left half holds photo texture
    share = tu.coverage(ids, valid, tex, TWO, np.ones(2, bool))
    area = tu.face_areas(SQUARE, TWO)
    o = 1 - (area * share).sum() / area.sum()
    assert o == pytest.approx(0.5, abs=0.02)
    # a triangle without a chart has none of its texture
    share = tu.coverage(ids, valid, tex, TWO, np.array([True, False]))
    assert share[1] == 0.0


def test_valid_texels_default_to_non_background():
    img = np.full((10, 10, 3), 225, np.uint8)
    img[2:5, 2:5] = (10, 20, 30)
    m = tu.valid_texels(img)
    assert m.sum() == 9
    alpha = np.zeros((10, 10), np.uint8)
    alpha[0] = 255
    assert tu.valid_texels(img, alpha).sum() == 10


def test_patches_stay_inside_the_mask_and_are_reproducible():
    img = np.zeros((40, 40, 3), np.uint8)
    img[:, 20:] = 200
    mask = np.zeros((40, 40), bool)
    mask[:, 20:] = True
    a = tu.sample_patches([(img, mask)], 50, 8, np.random.default_rng(3))
    b = tu.sample_patches([(img, mask)], 50, 8, np.random.default_rng(3))
    assert a.shape == (50, 8, 8, 3) and (a == 200).all()
    assert np.array_equal(a, b)
    none = tu.sample_patches(
        [(img, np.zeros((40, 40), bool))], 5, 8, np.random.default_rng()
    )
    assert len(none) == 0


# ---- a whole model ----

ATLAS_W, ATLAS_H = 256, 128


def textured_glb(tmp_path, *, with_mask=True, stretched=False, photos=2):
    """A job folder: model.glb (two 10 m walls on one 256x128 atlas, the second wall
    left untextured like texturing does), its photo mask, and street photos."""
    verts, faces, uvs = [], [], []
    walls = [SQUARE, SQUARE + [15, 0, 0]]
    for w, wall in enumerate(walls):
        for f in TWO:
            base = len(verts)
            for vi in f:
                verts.append(wall[vi])
                if w == 0:
                    x, z = wall[vi][0] * 10.0, wall[vi][2] * 10.0
                    if stretched:
                        x *= 2.0
                    uvs.append([x / ATLAS_W, 1.0 - (100.0 - z) / ATLAS_H])
                else:  # untextured: all corners at one grey texel
                    uvs.append([0.5, 1.0 - 120.0 / ATLAS_H])
            faces.append([base, base + 1, base + 2])
    atlas = np.full((ATLAS_H, ATLAS_W, 3), tu.UNTEXTURED_GREY, np.uint8)
    rng = np.random.default_rng(0)
    atlas[:100, : (200 if stretched else 100)] = rng.integers(
        0, 255, (100, 200 if stretched else 100, 3)
    )
    image = Image.fromarray(atlas)
    mesh = trimesh.Trimesh(np.array(verts, float), np.array(faces), process=False)
    mesh.visual = visual.TextureVisuals(
        uv=np.array(uvs),
        material=visual.material.PBRMaterial(baseColorTexture=image),
        image=image,
    )
    job = tmp_path / "job"
    job.mkdir()
    glb = job / "model.glb"
    trimesh.Scene({"buildings": mesh}).export(glb)
    if with_mask:
        mask = np.zeros((ATLAS_H, ATLAS_W), np.uint8)
        mask[:100, :50] = 255  # half the first wall came from a photo
        Image.fromarray(mask).save(job / ("model" + tu.MASK_SUFFIX))
    (job / "images").mkdir()
    for i in range(photos):
        Image.fromarray(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)).save(
            job / "images" / f"{i}_original.jpg"
        )
    return glb


def fake_extractor(device=None):
    def features(patches):
        x = patches.reshape(len(patches), -1, 3).astype(np.float64)
        return np.c_[x.mean(1), x.std(1)]  # 6 numbers a patch

    features.device = "cpu"
    return features


def test_a_model_is_scored_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(tu, "feature_extractor", fake_extractor)
    glb = textured_glb(tmp_path)
    out = tmp_path / "out"
    web = tu.run(tu.settings(k=64, patch=16), glb, out)

    assert web["status"] == "done" and web["mask_source"] == "texturing"
    # wall 1 half photo, wall 2 untextured: a quarter of the surface has photo texture
    assert web["terms"]["O"] == pytest.approx(0.75, abs=0.03)
    assert web["counts"]["no_chart"] == 2
    assert web["terms"]["D"] == pytest.approx(0.0)  # isotropic; untextured left out
    assert web["fid"] is not None and 0 <= web["terms"]["F"] <= 1
    assert not web["partial"]
    conf = 1 - (
        0.35 * web["terms"]["O"] + 0.25 * web["terms"]["D"] + 0.40 * web["terms"]["F"]
    )
    assert web["confidence"] == pytest.approx(conf, abs=1e-3)
    for name in tu.OUTPUT_FILES:
        assert (out / name).exists(), name
    doc = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert doc["tau"] == tu.TAU and doc["web"]["confidence"] == web["confidence"]
    assert "UV texture quality" in (out / "summary.md").read_text(encoding="utf-8")


def test_stretch_and_the_colour_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(tu, "feature_extractor", fake_extractor)
    glb = textured_glb(tmp_path, with_mask=False, stretched=True)
    web = tu.run(tu.settings(k=32, patch=16, no_fid=True), glb, tmp_path / "out")
    assert web["mask_source"] == "colour"
    assert web["terms"]["D"] == pytest.approx(1.0)  # every charted triangle 2x
    # no mask file: every non-grey texel counts, so wall 1 is fully textured
    assert web["terms"]["O"] == pytest.approx(0.5, abs=0.03)
    assert web["partial"] and web["terms"]["F"] is None


def test_without_inception_the_score_is_partial(tmp_path, monkeypatch):
    def no_weights(device=None):
        raise RuntimeError("couldn't download the weights")

    monkeypatch.setattr(tu, "feature_extractor", no_weights)
    web = tu.run(tu.settings(k=32, patch=16), textured_glb(tmp_path), tmp_path / "out")
    assert web["partial"] and web["terms"]["F"] is None
    assert "Inception-v3 not available" in web["fid_reason"]
    assert set(web["weights_used"]) == {"O", "D"}


def test_the_paper_setting_with_a_ground_truth_texture(tmp_path, monkeypatch):
    monkeypatch.setattr(tu, "feature_extractor", fake_extractor)
    glb = textured_glb(tmp_path)
    gt = np.full((ATLAS_H, ATLAS_W, 4), 0, np.uint8)
    gt[:100, :100, :3] = 90
    gt[:100, :100, 3] = 255  # the ground truth's valid texels: its alpha
    Image.fromarray(gt).save(tmp_path / "gt.png")
    web = tu.run(
        tu.settings(k=32, patch=16, gt_texture=str(tmp_path / "gt.png")),
        glb,
        tmp_path / "out",
    )
    # O = 1 - |M_res ∩ M_gt| / |M_gt|: the mask covers half of the gt's valid texels
    assert web["terms"]["O"] == pytest.approx(0.5, abs=0.01)


def test_a_model_without_texture_is_unavailable_not_a_crash(tmp_path):
    glb = tmp_path / "plain.glb"
    trimesh.Scene({"b": trimesh.creation.box()}).export(glb)
    result = evaluate_texture_model(glb, out=tmp_path / "out")
    assert result["status"] == "unavailable"
    assert "no textured building mesh" in result["reason"]


def test_bad_weights_are_a_failed_evaluation(tmp_path, monkeypatch):
    monkeypatch.setattr(tu, "feature_extractor", fake_extractor)
    result = evaluate_texture_model(
        textured_glb(tmp_path), out=tmp_path / "out", alpha=0.9
    )
    assert result["status"] == "failed" and "sum to 1" in result["reason"]


def test_the_stage_is_skipped_when_off_and_records_its_result_when_on(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(tu, "feature_extractor", fake_extractor)
    state = PipelineState(None, None, metadata={})
    assert evaluate_texture_stage("mesh", state) == "mesh"
    assert state.get_metadata("texture_evaluation", None) is None

    glb = textured_glb(tmp_path)
    lines = []

    class Log:
        def info(self, msg):
            lines.append(msg)

        warning = error = info

    state = PipelineState(
        None,
        None,
        metadata={
            "evaluate_texture": True,
            "output_path": str(glb),
            "texture_evaluation_dir": str(tmp_path / "tex_eval"),
            "reporter": Log(),
        },
    )
    assert evaluate_texture_stage("mesh", state) == "mesh"
    result = state.get_metadata("texture_evaluation", None)
    assert (
        result["status"] == "done" and (tmp_path / "tex_eval" / "summary.md").exists()
    )
    assert any(line.startswith("[texture] Confidence") for line in lines)

    state = PipelineState(
        None, None, metadata={"evaluate_texture": True, "reporter": Log()}
    )
    evaluate_texture_stage("mesh", state)
    assert state.get_metadata("texture_evaluation", None)["status"] == "unavailable"


def test_the_command_line_on_a_job_folder(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tu, "feature_extractor", fake_extractor)
    glb = textured_glb(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_texture_uv.py", str(glb.parent), "--k", "32", "--patch", "16"],
    )
    tu.main()
    printed = capsys.readouterr().out
    assert "[texture] Confidence" in printed
    assert (glb.parent / "texture_eval" / "01_scorecard.png").exists()
