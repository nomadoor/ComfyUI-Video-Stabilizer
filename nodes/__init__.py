"""ComfyUI node registration for the Video Stabilizer extension."""

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .video_stabilizer import VideoStabilizerNode
from .video_stabilizer_flow import VideoStabilizerFlowNode


class VideoStabilizerExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            VideoStabilizerNode,
            VideoStabilizerFlowNode,
        ]


async def comfy_entrypoint() -> VideoStabilizerExtension:
    return VideoStabilizerExtension()


NODE_CLASS_MAPPINGS = {
    "VideoStabilizerNode": VideoStabilizerNode,
    "VideoStabilizerFlowNode": VideoStabilizerFlowNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoStabilizerNode": "Video Stabilizer (Classic)",
    "VideoStabilizerFlowNode": "Video Stabilizer (Flow)",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "comfy_entrypoint",
    "VideoStabilizerExtension",
    "VideoStabilizerNode",
    "VideoStabilizerFlowNode",
]
