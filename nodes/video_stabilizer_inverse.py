"""
Inverse stabilization node for restoring the original motion/canvas.

This node consumes frames edited after stabilization plus the stabilizer metadata,
then applies the inverse of the exact warp matrices recorded during stabilization.
"""

from __future__ import annotations

from typing import Any

from comfy_api.latest import ComfyExtension, io

from .motion_apply import apply_motion
from .motion_meta import resolve_motion_meta
from .stabilizer_utils import (
    _convert_masks_for_output,
    _normalize_video_input,
    _parse_padding_color,
    _reconstruct_video,
)

JSONType = io.Custom("JSON")


class VideoStabilizerInverse(io.ComfyNode):
    """Apply inverse stabilization matrices from metadata."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        schema = io.Schema(
            node_id="video_stabilizer_inverse",
            display_name="Video Stabilizer Inverse",
            category="Video/Stabilization",
            description=(
                "Deprecated: use Video Stabilizer Motion Apply. Restores stabilized frames to the "
                "original canvas using stabilization metadata, and emits a padding mask for areas "
                "without source pixels."
            ),
            is_deprecated=True,
        )
        schema.inputs = [
            io.Image.Input("frames", display_name="Frames"),
            JSONType.Input("meta", display_name="Meta"),
            io.Color.Input(
                "padding_color",
                default="#7F7F7F",
                display_name="Padding Color",
                tooltip="HEX padding color used where inverse warping exposes empty pixels.",
            ),
        ]
        schema.outputs = [
            io.Image.Output("frames_restored", display_name="Restored Frames"),
            io.Mask.Output("padding_mask", display_name="Padding Mask"),
            JSONType.Output("meta", display_name="Meta"),
        ]
        return schema

    @classmethod
    def execute(
        cls,
        frames: Any,
        meta: dict[str, Any],
        padding_color: str,
    ) -> io.NodeOutput:
        context = _normalize_video_input(frames)
        padding_rgb = _parse_padding_color(padding_color)
        inverse_meta = dict(meta)
        inverse_meta.pop("motion_meta", None)
        motion = resolve_motion_meta(inverse_meta)
        result = apply_motion(
            context,
            inverse_meta,
            padding_rgb,
            framing_mode="pad",
            interpolation="bilinear",
        )
        if isinstance(meta, dict) and isinstance(meta.get("motion_meta"), dict):
            result.meta["motion_meta"] = meta["motion_meta"]
        result.meta.pop("motion_apply", None)
        result.meta["inverse_stabilization"] = {
            "source_size": [int(motion.output_size[0]), int(motion.output_size[1])],
            "input_size": [int(motion.input_size[0]), int(motion.input_size[1])],
            "output_size": [int(motion.output_size[0]), int(motion.output_size[1])],
            "matrix_convention": "stabilized_to_source",
            "source_matrix_convention": "source_to_stabilized",
            "framing_mode": meta.get("stabilization_warp", {}).get("framing_mode") if isinstance(meta, dict) else None,
            "note": "Restores original motion/canvas; pixels discarded by crop framing cannot be recovered.",
        }

        video_payload = _reconstruct_video(result.frames, context)
        mask_payload = _convert_masks_for_output(result.masks)
        return io.NodeOutput(video_payload, mask_payload, result.meta)


class VideoStabilizerInverseExtension(ComfyExtension):
    """Extension entrypoint used by ComfyUI to discover the node."""

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [VideoStabilizerInverse]
