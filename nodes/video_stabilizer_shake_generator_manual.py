"""
Manual Shake Generator node for producing deterministic motion metadata.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from comfy_api.latest import ComfyExtension, io

from .shake_noise import STYLES, ShakeRecipe, generate_shake_motion_meta
from .stabilizer_utils import _normalize_video_input

JSONType = io.Custom("JSON")
HANDHELD_DEFAULT = STYLES["handheld"]


class VideoStabilizerShakeGeneratorManual(io.ComfyNode):
    """Generate artificial camera motion metadata from explicit recipe values."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        schema = io.Schema(
            node_id="video_stabilizer_shake_generator_manual",
            display_name="Video Stabilizer Shake Generator Manual",
            category="Video/Stabilization",
            description="Generates deterministic shake motion metadata from manual absolute values.",
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
            io.Float.Input("pan", default=HANDHELD_DEFAULT.pan, min=0.0, max=5.0, step=0.01, display_name="Pan"),
            io.Float.Input("tilt", default=HANDHELD_DEFAULT.tilt, min=0.0, max=5.0, step=0.01, display_name="Tilt"),
            io.Float.Input("roll", default=HANDHELD_DEFAULT.roll, min=0.0, max=5.0, step=0.01, display_name="Roll"),
            io.Float.Input("zoom", default=HANDHELD_DEFAULT.zoom, min=0.0, max=0.05, step=0.001, display_name="Zoom"),
            io.Float.Input(
                "drift_freq",
                default=HANDHELD_DEFAULT.drift_freq,
                min=0.05,
                max=2.0,
                step=0.05,
                display_name="Drift Frequency",
            ),
            io.Float.Input(
                "tremor",
                default=HANDHELD_DEFAULT.tremor,
                min=0.0,
                max=2.0,
                step=0.05,
                display_name="Tremor",
            ),
            io.Float.Input(
                "tremor_freq",
                default=HANDHELD_DEFAULT.tremor_freq,
                min=1.0,
                max=15.0,
                step=0.5,
                display_name="Tremor Frequency",
            ),
            io.Float.Input(
                "jitter_rate",
                default=HANDHELD_DEFAULT.jitter_rate,
                min=0.0,
                max=3.0,
                step=0.1,
                display_name="Jitter Rate",
            ),
            io.Float.Input(
                "step",
                default=HANDHELD_DEFAULT.step,
                min=0.0,
                max=2.0,
                step=0.05,
                display_name="Step",
            ),
            io.Float.Input(
                "randomness",
                default=HANDHELD_DEFAULT.randomness,
                min=0.0,
                max=1.0,
                step=0.05,
                display_name="Randomness",
                display_mode=io.NumberDisplay.slider,
            ),
            io.Float.Input(
                "virtual_fov",
                default=HANDHELD_DEFAULT.virtual_fov,
                min=10.0,
                max=120.0,
                step=1.0,
                display_name="Virtual FOV",
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
        pan: float,
        tilt: float,
        roll: float,
        zoom: float,
        drift_freq: float,
        tremor: float,
        tremor_freq: float,
        jitter_rate: float,
        step: float,
        randomness: float,
        virtual_fov: float,
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
        recipe = ShakeRecipe(
            pan=pan,
            tilt=tilt,
            roll=roll,
            zoom=zoom,
            drift_freq=drift_freq,
            tremor=tremor,
            tremor_freq=tremor_freq,
            jitter_rate=jitter_rate,
            step=step,
            randomness=randomness,
            virtual_fov=virtual_fov,
        )
        motion_meta = generate_shake_motion_meta(
            recipe=recipe,
            frame_count=len(context.frames),
            width=context.width,
            height=context.height,
            fps=float(max(1.0, fps_candidate)),
            amount=amount,
            speed=speed,
            seed=seed,
            node="shake_generator_manual",
            style="manual",
        )
        return io.NodeOutput({"motion_meta": motion_meta})


class VideoStabilizerShakeGeneratorManualExtension(ComfyExtension):
    """Extension entrypoint used by ComfyUI to discover the node."""

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [VideoStabilizerShakeGeneratorManual]
