"""
Shake Generator node for producing deterministic motion metadata.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from comfy_api.latest import ComfyExtension, io

from .shake_noise import STYLES, generate_shake_motion_meta
from .stabilizer_utils import _normalize_video_input

JSONType = io.Custom("JSON")


class VideoStabilizerShakeGenerator(io.ComfyNode):
    """Generate artificial handheld motion metadata without changing pixels."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        schema = io.Schema(
            node_id="video_stabilizer_shake_generator",
            display_name="Video Stabilizer Shake Generator",
            category="Video/Stabilization",
            description="Generates deterministic shake motion metadata; it does not alter input frames.",
        )
        schema.inputs = [
            io.Image.Input(
                "frames_context",
                display_name="Frames Context",
                tooltip=(
                    "The input frames are used only to read frame count and resolution. This node outputs "
                    "motion metadata only; connect it to Video Stabilizer Motion Apply to move pixels."
                ),
            ),
            io.Float.Input(
                "frame_rate",
                default=16.0,
                min=1.0,
                step=0.1,
                display_name="Input FPS",
                tooltip="Fallback frame rate when the input does not carry fps metadata.",
            ),
            io.Combo.Input(
                "style",
                options=list(STYLES.keys()),
                default="handheld",
                display_name="Style",
            ),
            io.Float.Input(
                "amount",
                default=1.0,
                min=0.0,
                max=3.0,
                step=0.05,
                display_name="Amount",
                display_mode=io.NumberDisplay.slider,
            ),
            io.Float.Input(
                "speed",
                default=1.0,
                min=0.1,
                max=3.0,
                step=0.05,
                display_name="Speed",
                display_mode=io.NumberDisplay.slider,
            ),
            io.Int.Input(
                "seed",
                default=0,
                min=0,
                max=0xFFFFFFFFFFFFFFFF,
                display_name="Seed",
                control_after_generate=io.ControlAfterGenerate.fixed,
            ),
        ]
        schema.outputs = [
            JSONType.Output("motion_meta", display_name="Motion Meta"),
        ]
        return schema

    @classmethod
    def execute(
        cls,
        frames_context: Any,
        frame_rate: float,
        style: str,
        amount: float,
        speed: float,
        seed: int,
    ) -> io.NodeOutput:
        context = _normalize_video_input(frames_context)
        fps_candidate = context.fps
        if not isinstance(fps_candidate, (int, float)) or not np.isfinite(fps_candidate) or fps_candidate <= 0.0:
            fps_candidate = frame_rate
        if not isinstance(fps_candidate, (int, float)) or not np.isfinite(fps_candidate) or fps_candidate <= 0.0:
            fps_candidate = 16.0
        motion_meta = generate_shake_motion_meta(
            frame_count=len(context.frames),
            width=context.width,
            height=context.height,
            fps=float(max(1.0, fps_candidate)),
            recipe=STYLES[style],
            amount=amount,
            speed=speed,
            seed=seed,
            node="shake_generator",
            style=style,
        )
        return io.NodeOutput({"motion_meta": motion_meta})


class VideoStabilizerShakeGeneratorExtension(ComfyExtension):
    """Extension entrypoint used by ComfyUI to discover the node."""

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [VideoStabilizerShakeGenerator]
