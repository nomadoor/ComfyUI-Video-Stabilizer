"""
Motion Apply node for applying v2 motion metadata to frames.
"""

from __future__ import annotations

from typing import Any

from comfy_api.latest import ComfyExtension, io
from comfy.utils import ProgressBar

from .motion_apply import apply_motion
from .stabilizer_utils import (
    _convert_masks_for_output,
    _normalize_video_input,
    _parse_padding_color,
    _reconstruct_video,
)

JSONType = io.Custom("JSON")
BLUR_QUALITY_SAMPLES = {
    "Draft": 5,
    "Standard": 9,
    "High": 17,
    "Ultra": 33,
}


class VideoStabilizerMotionApply(io.ComfyNode):
    """Apply motion_meta matrices to a video sequence."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        schema = io.Schema(
            node_id="video_stabilizer_motion_apply",
            display_name="Video Stabilizer Motion Apply",
            category="Video/Stabilization",
            description="Applies motion metadata to frames and emits a padding mask.",
        )
        schema.inputs = [
            io.Image.Input("frames", display_name="Frames"),
            JSONType.Input("motion_meta", display_name="Motion Meta"),
            io.Combo.Input(
                "framing_mode",
                options=["crop_and_pad", "crop", "expand"],
                default="crop_and_pad",
                display_name="Framing Mode",
            ),
            io.Combo.Input(
                "interpolation",
                options=["bilinear", "bicubic"],
                default="bilinear",
                display_name="Interpolation",
            ),
            io.Color.Input(
                "padding_color",
                default="#7F7F7F",
                display_name="Padding Color",
                tooltip="HEX padding color used where warping exposes empty pixels.",
            ),
            io.Float.Input(
                "motion_blur",
                default=0.0,
                min=0.0,
                max=1.0,
                step=0.05,
                display_name="Motion Blur",
                tooltip="Shutter fraction for matrix-sampled motion blur. 0 disables blur.",
                display_mode=io.NumberDisplay.slider,
            ),
            io.Combo.Input(
                "motion_blur_quality",
                options=list(BLUR_QUALITY_SAMPLES.keys()),
                default="Standard",
                display_name="Blur Quality",
                tooltip="Draft is faster. High and Ultra average more shutter samples for smoother blur.",
            ),
        ]
        schema.outputs = [
            io.Image.Output("frames", display_name="Frames"),
            io.Mask.Output("padding_mask", display_name="Padding Mask"),
            JSONType.Output("meta", display_name="Meta"),
        ]
        return schema

    @classmethod
    def execute(
        cls,
        frames: Any,
        motion_meta: dict[str, Any],
        framing_mode: str,
        interpolation: str,
        padding_color: str,
        motion_blur: float,
        motion_blur_quality: str,
    ) -> io.NodeOutput:
        context = _normalize_video_input(frames)
        padding_rgb = _parse_padding_color(padding_color)
        resolved_quality = motion_blur_quality if motion_blur_quality in BLUR_QUALITY_SAMPLES else "Standard"
        motion_blur_samples = BLUR_QUALITY_SAMPLES[resolved_quality]
        frame_count = len(context.frames)
        sample_count = int(max(3, min(33, motion_blur_samples))) if motion_blur > 0.0 else 1
        progress_total = frame_count * sample_count
        if framing_mode == "crop":
            progress_total += frame_count
        progress_total = max(progress_total, 1)
        pbar = ProgressBar(max(progress_total, 1))
        progress_done = 0

        def update_progress() -> None:
            nonlocal progress_done
            progress_done += 1
            pbar.update_absolute(min(progress_done, progress_total), progress_total)

        result = apply_motion(
            context,
            motion_meta,
            padding_rgb,
            framing_mode=framing_mode,  # type: ignore[arg-type]
            interpolation=interpolation,  # type: ignore[arg-type]
            motion_blur=motion_blur,
            motion_blur_samples=motion_blur_samples,
            progress_callback=update_progress,
        )
        result.meta.setdefault("motion_apply", {})["motion_blur_quality"] = resolved_quality
        pbar.update_absolute(progress_total, progress_total)
        video_payload = _reconstruct_video(result.frames, context)
        mask_payload = _convert_masks_for_output(result.masks)
        return io.NodeOutput(video_payload, mask_payload, result.meta)


class VideoStabilizerMotionApplyExtension(ComfyExtension):
    """Extension entrypoint used by ComfyUI to discover the node."""

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [VideoStabilizerMotionApply]
