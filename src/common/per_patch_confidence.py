"""Per-texel confidence of the photo texture, stored as its alpha (texturing.blend, and
texturing.tex_projection's "best" mode): a texel seen square-on, close and unobstructed is
opaque; a poorly seen one lets the plain base mesh show through."""

import numpy as np

OPAQUE, SEE_THROUGH = 255, 200


def clip_confidence_score(confidence_raw, threshold=0.4, sharpen=1):
    """Alpha (uint8) from a confidence in [0, 1]: OPAQUE above threshold, else SEE_THROUGH.
    confidence_raw may already be raised to the power sharpen (blend's view weights are), so it
    is compared with threshold ** sharpen."""
    confidence = np.asarray(confidence_raw, dtype=np.float64)
    return np.where(confidence > threshold**sharpen, OPAQUE, SEE_THROUGH).astype(
        np.uint8
    )


def confidence_score(distance, cos, clear, average=True):
    """Alpha of a photo patch: cos(view angle) x unobstructed share / (1 + distance / 15 m), the
    same terms as blend's view weight, unsharpened. average: use the patch's mean distance."""
    if average:
        distance = np.mean(distance)
    distance_confidence = 1.0 / (1.0 + distance / 15.0)
    confidence = cos * clear * distance_confidence
    return clip_confidence_score(confidence)
