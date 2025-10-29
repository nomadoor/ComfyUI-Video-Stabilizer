"""ComfyUI node registration for the Video Stabilizer extension."""

from .video_stabilizer import (
    VideoStabilizerExtension,
    VideoStabilizerNode,
    comfy_entrypoint,
)

NODE_CLASS_MAPPINGS = {
    "VideoStabilizerNode": VideoStabilizerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoStabilizerNode": "Video Stabilizer (Classic)",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "comfy_entrypoint",
    "VideoStabilizerExtension",
]
