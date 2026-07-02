"""
ComfyUI entrypoint for the Video Stabilizer custom node package.

Registers the stabilizer nodes and motion metadata producer/consumer helpers.
"""

from __future__ import annotations

from comfy_api.latest import ComfyExtension

from .nodes.node_replacements import register_node_replacements
from .nodes.video_stabilizer_inverse import VideoStabilizerInverse
from .nodes.video_stabilizer_classic import VideoStabilizerClassic
from .nodes.video_stabilizer_flow import VideoStabilizerFlow
from .nodes.video_stabilizer_motion_apply import VideoStabilizerMotionApply
from .nodes.video_stabilizer_shake_generator import VideoStabilizerShakeGenerator

__all__ = ["comfy_entrypoint"]


class VideoStabilizerSuiteExtension(ComfyExtension):
    async def get_node_list(self) -> list[type]:
        return [
            VideoStabilizerClassic,
            VideoStabilizerFlow,
            VideoStabilizerMotionApply,
            VideoStabilizerShakeGenerator,
            VideoStabilizerInverse,
        ]

    async def on_load(self) -> None:
        await register_node_replacements()


async def comfy_entrypoint() -> VideoStabilizerSuiteExtension:
    """Return the extension instance ComfyUI will use to discover nodes."""
    return VideoStabilizerSuiteExtension()
