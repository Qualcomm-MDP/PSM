"""Obstruction classes by name and the class-map cache, without loading a model."""

import numpy as np

from segmentation.model import LabelCache
from segmentation.obstruction import obstruction_mask

VISTAS = {17: "Building", 19: "Person", 55: "Car", 64: "Ego Vehicle", 27: "Sky"}
CITYSCAPES = {2: "building", 11: "person", 13: "car", 10: "sky"}


def _labels(ids):
    """A 40 x 40 class map with one 10 px wide band per class id."""
    seg = np.zeros((40, 40), np.uint8)
    for k, i in enumerate(ids):
        seg[:, 10 * k : 10 * k + 10] = i
    return seg


def test_vistas_obstructions_include_the_capture_vehicle():
    mask = obstruction_mask(_labels([17, 19, 55, 64]), VISTAS)
    assert mask[20, 2] == 0  # building, more than 7 px from the person band
    assert (mask[:, 12:40] == 255).all()  # person, car and ego vehicle


def test_cityscapes_names_work_too():
    mask = obstruction_mask(_labels([2, 10, 13, 2]), CITYSCAPES)
    assert mask[20, 2] == 0 and mask[20, 12] == 0  # building, sky
    assert (mask[:, 20:30] == 255).all()  # car
    assert mask[20, 38] == 0


def test_label_cache_scales_maps_and_drops_another_models(tmp_path):
    cache = LabelCache(tmp_path / "labels")
    cache.start("vistas", VISTAS)
    cache.save("123", _labels([17, 55]))
    assert cache.model() == ("vistas", VISTAS)
    big = cache.get("123", (80, 80))
    assert big.shape == (80, 80) and big[0, 0] == 17 and big[0, 30] == 55
    assert cache.get("missing") is None

    cache.start("cityscapes", CITYSCAPES)  # another model's maps would mislabel classes
    assert cache.get("123") is None and cache.model()[0] == "cityscapes"


def test_batch_masks_match_one_at_a_time():
    from segmentation.obstruction import build_obstruction_mask, build_obstruction_masks

    img = np.zeros((40, 40, 3), np.uint8)
    labels = (_labels([17, 19, 55, 64]), VISTAS)
    one = [build_obstruction_mask(img, labels=labels) for _ in range(3)]
    batch = build_obstruction_masks(
        [img, img, img], labels_list=[labels, labels, labels], batch_size=3
    )
    assert len(batch) == 3
    assert all(np.array_equal(a, b) for a, b in zip(one, batch))


def test_segment_images_one_gpu_call_per_batch(monkeypatch):
    from segmentation import obstruction

    calls: list[int] = []

    def fake_segment(images, *a, **k):
        calls.append(len(images))
        return [np.zeros(im.shape[:2], np.uint8) for im in images]

    monkeypatch.setattr(obstruction, "segment", fake_segment)
    monkeypatch.setattr(obstruction, "load_model", lambda *a, **k: (None, object()))
    monkeypatch.setattr(obstruction, "id2label", lambda m: VISTAS)
    imgs = [np.zeros((8, 8, 3), np.uint8) for _ in range(5)]
    masks = obstruction.build_obstruction_masks(imgs, batch_size=5)
    assert calls == [5]
    assert len(masks) == 5


def test_segment_batch_falls_back_on_oom(monkeypatch):
    from segmentation import obstruction

    calls: list[int] = []

    def fake_segment(images, *a, **k):
        calls.append(len(images))
        if len(images) > 1:
            raise RuntimeError("CUDA out of memory")
        return [np.zeros(im.shape[:2], np.uint8) for im in images]

    monkeypatch.setattr(obstruction, "segment", fake_segment)
    monkeypatch.setattr(obstruction, "load_model", lambda *a, **k: (None, object()))
    monkeypatch.setattr(obstruction, "id2label", lambda m: VISTAS)
    imgs = [np.zeros((4, 4, 3), np.uint8) for _ in range(3)]
    masks = obstruction.build_obstruction_masks(imgs, batch_size=3)
    assert calls[0] == 3 and all(c == 1 for c in calls[1:])
    assert len(masks) == 3


def test_default_seg_batch_is_8_on_cuda(monkeypatch):
    import torch

    from segmentation.model import select_device
    from segmentation.obstruction import default_seg_batch

    monkeypatch.delenv("SEG_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert select_device("auto").type == "cuda"
    assert default_seg_batch("auto") == 8
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert default_seg_batch("cpu") == 1


def test_auto_and_gpu_pick_the_best_device(monkeypatch):
    import torch

    from segmentation.model import select_device

    monkeypatch.delenv("SEG_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert select_device("auto").type == "mps" and select_device("gpu").type == "mps"
    assert select_device(None).type == "mps" and select_device("cpu").type == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert select_device("gpu").type == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert select_device("gpu").type == "cpu"
