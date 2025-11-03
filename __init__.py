"""
ComfyUI entrypoint for the Video Stabilizer custom node package.

Registers both the classic feature-tracking stabilizer and the dense-flow variant.
"""

from __future__ import annotations

from comfy_api.latest import ComfyExtension

from .nodes.video_stabilizer_classic import VideoStabilizerClassic
from .nodes.video_stabilizer_flow import VideoStabilizerFlow

__all__ = ["comfy_entrypoint"]


class VideoStabilizerSuiteExtension(ComfyExtension):
    async def get_node_list(self) -> list[type]:
        return [VideoStabilizerClassic, VideoStabilizerFlow]


async def comfy_entrypoint() -> VideoStabilizerSuiteExtension:
    """Return the extension instance ComfyUI will use to discover nodes."""
    return VideoStabilizerSuiteExtension()
