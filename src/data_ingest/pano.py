"""360 photos as pinhole views looking right and left of travel.

A 360 (equirectangular) photo cannot be projected like a perspective photo, so it is cut
into two pinhole views, ±90° from the direction of travel and fov_deg across (default 90°):
where the facades on either side of the street are. The pose alignment downloads 360 runs
as square views level with the horizon (data_ingest.mapillary_sequences.download_pano).
Texturing uses taller views of the same 360 photos (pano_views, and align_poses for the
aligned ones): TEXTURE_VIEW, from 45° below the horizon (the road and sidewalk, as the
square view) to 70° above it (the square view stops at 45°, below the upper floors of
taller buildings), as a view tilted up by half the difference, so its principal point stays
at the image centre as texturing expects.

The crop directions are fixed in the 360 image: the side's angle minus the sequence's
median view angle (where the image centre looks relative to travel; 0 when unknown).
"""

from __future__ import annotations

import math
import threading

import cv2
import numpy as np

PANO_TYPES = ("spherical", "equirectangular")
# crop directions relative to travel; the first is the primary
PANO_SIDES = {"right": 90.0, "left": -90.0}
PANO_FOV_DEG = 90.0
PANO_MAX_WIDTH = 2048  # crops no larger than a perspective frame at 2048 px
# texturing's views: how far below and above the horizon they reach (degrees), 90 deg across
TEXTURE_VIEW = {"down_deg": 45.0, "up_deg": 70.0}

_maps: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
_maps_lock = threading.Lock()  # download_pano crops in threads: each map is built once


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def is_pano(img: dict) -> bool:
    return bool(img.get("is_pano")) or img.get("camera_type") in PANO_TYPES


def crop_axes(yaw_deg: float, pitch_deg: float = 0.0) -> np.ndarray:
    """Columns: a crop's right, down and forward axes in the 360 camera's frame (x right, y down,
    z the image centre; OpenSfM's spherical camera), for a crop turned yaw_deg to the right and
    then tilted pitch_deg up."""
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    turn = np.array(
        [
            [math.cos(y), 0.0, math.sin(y)],
            [0.0, 1.0, 0.0],
            [-math.sin(y), 0.0, math.cos(y)],
        ]
    )
    tilt = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(p), -math.sin(p)],
            [0.0, math.sin(p), math.cos(p)],
        ]
    )
    return turn @ tilt


def texture_view(width: int, fov_deg: float = PANO_FOV_DEG) -> tuple[float, int]:
    """(pitch up in degrees, height in px, even) of texturing's view (TEXTURE_VIEW) that is
    width px and fov_deg across."""
    down, up = TEXTURE_VIEW["down_deg"], TEXTURE_VIEW["up_deg"]
    half_tall = math.radians((down + up) / 2)
    f = width / 2 / math.tan(math.radians(fov_deg) / 2)
    return (up - down) / 2, int(round(f * math.tan(half_tall))) * 2


def crop_size(pano_width: float | None, fov_deg: float = PANO_FOV_DEG) -> int:
    """Side (px, even) of a crop with the 360 image's own resolution, at most PANO_MAX_WIDTH."""
    return min(PANO_MAX_WIDTH, int(round((pano_width or 2048) * fov_deg / 360 / 2)) * 2)


def crop_rotvec(rotvec, yaw_deg: float, pitch_deg: float = 0.0) -> list[float]:
    """A crop's world-to-camera rotation (Rodrigues) from its 360 image's."""
    R = cv2.Rodrigues(np.asarray(rotvec, float))[0]
    return cv2.Rodrigues(crop_axes(yaw_deg, pitch_deg).T @ R)[0].ravel().tolist()


def pano_crop_maps(W, H, yaw_deg, fov_deg, size, pitch_deg=0.0, height=None):
    """cv2.remap maps from a size px wide (height px tall, default square) pinhole view, fov_deg
    across, looking yaw_deg right of the image centre and tilted pitch_deg up from the 360
    image's horizon, into a W x H equirectangular image."""
    height = height or size
    f = size / 2 / math.tan(math.radians(fov_deg) / 2)
    u, v = np.meshgrid(
        np.arange(size) - (size - 1) / 2, np.arange(height) - (height - 1) / 2
    )
    rays = np.stack([u / f, v / f, np.ones_like(u)], axis=-1)
    # as one 2-D product: NumPy's stacked (3-D) matmul crashed now and then on macOS
    # (segmentation fault in Accelerate, about 1 call in 10)
    d = (rays.reshape(-1, 3) @ crop_axes(yaw_deg, pitch_deg).T).reshape(rays.shape)
    lon = np.arctan2(d[..., 0], d[..., 2])
    lat = np.arctan2(-d[..., 1], np.hypot(d[..., 0], d[..., 2]))
    return (
        ((lon / (2 * math.pi) + 0.5) * W - 0.5).astype(np.float32),
        ((0.5 - lat / math.pi) * H - 0.5).astype(np.float32),
    )


def crop_view(
    pano: np.ndarray,
    yaw_deg: float,
    fov_deg: float,
    size: int,
    pitch_deg: float = 0.0,
    height: int | None = None,
) -> np.ndarray:
    """The pinhole view (size px wide, height px tall, default square) of a 360 image (BGR)
    looking yaw_deg right of its centre and pitch_deg up."""
    # The crop's centre has size/2 px per radian, the 360 image W/2pi: shrink a much sharper 360
    # image first (area averaging), so the remap does not alias
    if pano.shape[1] > 1.5 * math.pi * size:
        pano = cv2.resize(
            pano,
            (int(math.pi * size), int(math.pi * size) // 2),
            interpolation=cv2.INTER_AREA,
        )
    H, W = pano.shape[:2]
    key = (W, H, round(yaw_deg, 6), fov_deg, size, round(pitch_deg, 6), height)
    with _maps_lock:
        if key not in _maps:
            _maps[key] = pano_crop_maps(W, H, yaw_deg, fov_deg, size, pitch_deg, height)
        mx, my = _maps[key]
    return cv2.remap(pano, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def sequence_view_angles(records: list[dict]) -> dict[str, float | None]:
    """{sequence id: median angle of the 360 image centre relative to travel (deg; None
    when unknown)} over the 360 records, from their compass against their track."""
    from data_ingest.mapillary_sequences import view_angle
    from pose_alignment.geo_common import LocalFrame

    by_seq: dict[str, list[dict]] = {}
    for img in records:
        if is_pano(img) and img.get("sequence"):
            by_seq.setdefault(img["sequence"], []).append(img)
    out = {}
    for seq, imgs in by_seq.items():
        imgs.sort(key=lambda i: i.get("captured_at") or 0)
        lonlat = [i["geometry"]["coordinates"] for i in imgs]
        frame = LocalFrame(*lonlat[0])
        out[seq] = view_angle(imgs, [frame.xy(*p) for p in lonlat])
    return out


def pano_views(
    metadata: dict[str, dict], records: list[dict], fov_deg: float = PANO_FOV_DEG
) -> dict[str, dict]:
    """The fetch stage's metadata with every 360 photo replaced by its right and left views
    ("<id>_right", "<id>_left"): texturing's views (TEXTURE_VIEW), pinhole cameras with an
    exact focal, the 360 photo's position and its rotation turned to the view. Each keeps
    the 360 photo's URL and a "pano_crop" entry ({id, side, yaw_deg, pitch_deg, fov_deg,
    size, height}) telling texturing what to cut."""
    from scipy.spatial.transform import Rotation

    by_id = {str(r["id"]): r for r in records}
    views = sequence_view_angles(records)
    out = {}
    for img_id, meta in metadata.items():
        rec = by_id.get(str(img_id), {})
        if not (is_pano(rec) or meta.get("camera_type") in PANO_TYPES):
            out[img_id] = meta
            continue
        if meta.get("rotvec") is None:
            continue  # no orientation for the views
        view_deg = views.get(str(rec.get("sequence"))) or 0.0
        size = crop_size(meta.get("original_width") or rec.get("width"), fov_deg)
        pitch, height = texture_view(size, fov_deg)
        for side, angle in PANO_SIDES.items():
            yaw = wrap_deg(angle - view_deg)
            rotvec = crop_rotvec(meta["rotvec"], yaw, pitch)
            z, y, x = Rotation.from_rotvec(rotvec).as_euler("ZYX", degrees=True)
            out[f"{img_id}_{side}"] = {
                **meta,
                "rotvec": rotvec,
                "yaw": round(float(z), 2),
                "pitch": round(float(y), 2),
                "roll": round(float(x), 2),
                "focal_length_pixels": size / 2 / math.tan(math.radians(fov_deg) / 2),
                "original_width": size,
                "original_height": height,
                "camera_type": "perspective",
                "pano_crop": {
                    "id": str(img_id),
                    "side": side,
                    "yaw_deg": round(yaw, 3),
                    "pitch_deg": pitch,
                    "fov_deg": fov_deg,
                    "size": size,
                    "height": height,
                },
            }
    return out
