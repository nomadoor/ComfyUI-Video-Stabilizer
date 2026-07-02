"""
Motion Apply node for applying v2 motion metadata to frames.
"""

from __future__ import annotations

from typing import Any

from comfy_api.latest import ComfyExtension, io

from .motion_apply import apply_motion
from .stabilizer_utils import (
    _convert_masks_for_output,
    _normalize_video_input,
    _parse_padding_color,
    _reconstruct_video,
)

JSONType = io.Custom("JSON")


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
                options=["pad", "crop"],
                default="pad",
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
    ) -> io.NodeOutput:
        context = _normalize_video_input(frames)
        padding_rgb = _parse_padding_color(padding_color)
        result = apply_motion(
            context,
            motion_meta,
            padding_rgb,
            framing_mode=framing_mode,  # type: ignore[arg-type]
            interpolation=interpolation,  # type: ignore[arg-type]
        )
        video_payload = _reconstruct_video(result.frames, context)
        mask_payload = _convert_masks_for_output(result.masks)
        return io.NodeOutput(video_payload, mask_payload, result.meta)


class VideoStabilizerMotionApplyExtension(ComfyExtension):
    """Extension entrypoint used by ComfyUI to discover the node."""

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [VideoStabilizerMotionApply]
