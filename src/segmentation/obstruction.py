from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from simple_lama_inpainting import SimpleLama

from common.reporter import Reporter, get_reporter
from segmentation.model import (
    class_lut,
    id2label,
    load_model,
    segment,
    select_device,
)

# Classes painted over before texturing (Mapillary Vistas and Cityscapes names, matched in
# any case): things in front of the facades. Vistas also marks the capture vehicle itself
# (Ego Vehicle, Car Mount), which Cityscapes' 19 classes cannot.
OBSTRUCTION_NAMES = {
    # Cityscapes
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    # Mapillary Vistas (Pole, Traffic Light, Vegetation, Person, Car, ... as above)
    "utility pole",
    "street light",
    "traffic sign (front)",
    "traffic sign (back)",
    "traffic sign frame",
    "bicyclist",
    "motorcyclist",
    "other rider",
    "boat",
    "caravan",
    "on rails",
    "other vehicle",
    "trailer",
    "wheeled slow",
    "car mount",
    "ego vehicle",
    "bird",
    "ground animal",
}

LAMA_MAX_DIM = 1024

_lama: SimpleLama | None = None
_lama_cpu_only = False  # set once LaMa failed on the GPU: the CPU from then on


def _load_lama(
    device: str | None = None, reporter: Reporter | None = None
) -> SimpleLama:
    """LaMa on the best device (select_device; on Apple's GPU about 14x faster than the
    CPU, the same result). The TorchScript file is loaded onto the CPU first (it was saved
    on CUDA), then moved."""
    global _lama, _lama_cpu_only
    target = torch.device("cpu") if _lama_cpu_only else select_device(device)
    if _lama is not None and _lama.device == target:
        return _lama
    (reporter or get_reporter()).info(f"Loading LaMa inpainting model on {target} …")
    _orig = torch.jit.load
    torch.jit.load = lambda *a, **kw: _orig(*a, **{**kw, "map_location": "cpu"})
    try:
        lama = SimpleLama(device=torch.device("cpu"))
    finally:
        torch.jit.load = _orig
    if target.type != "cpu":
        try:
            lama.model.to(target)
            lama.device = target
        except RuntimeError as exc:
            (reporter or get_reporter()).warning(
                f"LaMa on {target} failed ({exc}); using the CPU"
            )
            _lama_cpu_only = True
    _lama = lama
    return _lama


# CUDA can hold several Mask2Former 384px images; the CPU/MPS path stays at 1.
CUDA_SEG_BATCH = 8


def default_seg_batch(device: str | None = None) -> int:
    """How many photos Mask2Former should see at once: 8 on NVIDIA, else 1."""
    return CUDA_SEG_BATCH if select_device(device).type == "cuda" else 1


def _is_oom(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name in ("OutOfMemoryError", "CUDAOutOfMemoryError"):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg


def segment_images(
    images_bgr: list[np.ndarray],
    model_id: str | None = None,
    device: str | None = None,
    reporter: Reporter | None = None,
    batch_size: int = 1,
) -> list[np.ndarray]:
    """Class maps for every image, `batch_size` at a time. On CUDA OOM, drop to 1
    and keep going (never crash the pipeline)."""
    if not images_bgr:
        return []
    size = max(1, batch_size)
    out: list[np.ndarray] = []
    i = 0
    while i < len(images_bgr):
        chunk = images_bgr[i : i + size]
        try:
            out.extend(segment(chunk, model_id, device, reporter))
            i += len(chunk)
        except RuntimeError as exc:
            if size == 1 or not _is_oom(exc):
                raise
            (reporter or get_reporter()).warning(
                f"Segmentation batch of {size} ran out of GPU memory; retrying one at a time"
            )
            size = 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return out


def obstruction_mask(labels: np.ndarray, names: dict[int, str]) -> np.ndarray:
    """255 where the class map (with its class names) shows an obstruction, grown by 7 px."""
    mask = class_lut(names, OBSTRUCTION_NAMES, value=255)[labels]
    return cv2.dilate(mask, np.ones((15, 15), np.uint8))


def build_obstruction_masks(
    images_bgr: list[np.ndarray],
    model_id: str | None = None,
    device: str | None = None,
    reporter: Reporter | None = None,
    labels_list: list[tuple[np.ndarray, dict[int, str]] | None] | None = None,
    batch_size: int | None = None,
) -> list[np.ndarray]:
    """Obstruction masks for a batch of BGR images. labels_list[i] is the class map
    (and names) when image i was already segmented; None means run the model.
    batch_size None: 8 on CUDA, 1 otherwise."""
    if labels_list is None:
        labels_list = [None] * len(images_bgr)
    if len(labels_list) != len(images_bgr):
        raise ValueError("labels_list must match images_bgr")
    need = [i for i, lab in enumerate(labels_list) if lab is None]
    resolved: list[tuple[np.ndarray, dict[int, str]] | None] = list(labels_list)
    if need:
        size = default_seg_batch(device) if batch_size is None else max(1, batch_size)
        segs = segment_images(
            [images_bgr[i] for i in need],
            model_id,
            device,
            reporter,
            batch_size=size,
        )
        _, model = load_model(model_id, device, reporter)
        names = id2label(model)
        for i, seg in zip(need, segs):
            resolved[i] = (seg, names)
    return [obstruction_mask(*lab) for lab in resolved]  # type: ignore[arg-type]


def build_obstruction_mask(
    img_bgr: np.ndarray,
    model_id: str | None = None,
    device: str | None = None,
    reporter: Reporter | None = None,
    labels: tuple[np.ndarray, dict[int, str]] | None = None,
) -> np.ndarray:
    """labels: the image's class map and class names when it was segmented already
    (segmentation.model.LabelCache); otherwise the model runs on the image."""
    return build_obstruction_masks(
        [img_bgr], model_id, device, reporter, [labels], batch_size=1
    )[0]


def synthesize_texture(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    reporter: Reporter | None = None,
    device: str | None = None,
) -> np.ndarray:
    lama = _load_lama(device, reporter)
    h, w = img_bgr.shape[:2]

    scale = min(1.0, LAMA_MAX_DIM / max(h, w))
    ih, iw = int(h * scale), int(w * scale)
    img_s = cv2.resize(img_bgr, (iw, ih), interpolation=cv2.INTER_AREA)
    msk_s = cv2.resize(mask, (iw, ih), interpolation=cv2.INTER_NEAREST)

    pil_img = Image.fromarray(cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB))
    pil_mask = Image.fromarray(msk_s)
    try:
        result = lama(pil_img, pil_mask)
    except RuntimeError as exc:
        if lama.device.type == "cpu":
            raise
        (reporter or get_reporter()).warning(
            f"LaMa on {lama.device} failed ({exc}); using the CPU"
        )
        global _lama_cpu_only
        _lama_cpu_only = True
        lama.model.to(torch.device("cpu"))
        lama.device = torch.device("cpu")
        result = lama(pil_img, pil_mask)
    result_s = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

    inpainted = cv2.resize(result_s, (w, h), interpolation=cv2.INTER_LINEAR)
    out = img_bgr.copy()
    out[mask > 0] = inpainted[mask > 0]
    return out


def remove_obstructions(
    img_bgr: np.ndarray,
    model_id: str | None = None,
    device: str | None = None,
    reporter: Reporter | None = None,
    labels: tuple[np.ndarray, dict[int, str]] | None = None,
) -> np.ndarray:
    mask = build_obstruction_mask(img_bgr, model_id, device, reporter, labels)
    return synthesize_texture(img_bgr, mask, reporter, device)
