from __future__ import annotations

from comfy_api.latest import ComfyAPI, io

api = ComfyAPI()


async def register_node_replacements() -> None:
    await api.node_replacement.register(
        io.NodeReplace(
            new_node_id="video_stabilizer_motion_apply",
            old_node_id="video_stabilizer_inverse",
            old_widget_ids=["padding_color"],
            input_mapping=[
                {"new_id": "frames", "old_id": "frames"},
                {"new_id": "motion_meta", "old_id": "meta"},
                {"new_id": "padding_color", "old_id": "padding_color"},
                {"new_id": "framing_mode", "set_value": "pad"},
                {"new_id": "interpolation", "set_value": "bilinear"},
            ],
            output_mapping=[
                {"new_idx": 0, "old_idx": 0},
                {"new_idx": 1, "old_idx": 1},
                {"new_idx": 2, "old_idx": 2},
            ],
        )
    )
