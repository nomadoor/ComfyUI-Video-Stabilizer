"""
ComfyUI entrypoint for the Video Stabilizer custom node package.

Exposes a V3-compatible extension that registers the Classic stabilizer node.
"""

from __future__ import annotations

from .nodes.video_stabilizer_classic import VideoStabilizerClassicExtension

__all__ = ["comfy_entrypoint"]


async def comfy_entrypoint() -> VideoStabilizerClassicExtension:
    """Return the extension instance ComfyUI will use to discover nodes."""
    return VideoStabilizerClassicExtension()

