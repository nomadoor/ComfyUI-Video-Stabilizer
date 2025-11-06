# 2025-11-05 – Crop Mode Rework Plan

## Status
Implemented

## Context
- Current crop mode shrinks the crop window after motion estimation using ad-hoc loops.
- When `keep_fov` is high and motion is large, the algorithm still produces padding or silently reduces zoom.
- Users expect crop mode to guarantee “no padding”, even if that means backing off stabilization strength.
- We need to reason from `keep_fov` first, support extreme cases (keep_fov = 0/1, non-square aspect, empty input) and provide actionable metadata.

## Decision
- Derive the required content bounding box per frame and compare it with the requested `keep_fov` allowance before deciding any crop.
- If the required crop would violate `keep_fov`, scale down stabilization deltas (strength/smooth effect) until the requested FOV can be respected, or report the effective FOV when impossible.
- Compute a single crop transform from that final box, apply a small configurable safety margin that never pushes the crop below the requested FOV, and reuse it for every frame for both classic and flow stabilizers.
- Refine the crop via an AND-mask ➔ optional erosion ➔ largest-rectangle-of-ones stage so that crop mode cannot leak padding even with numerical noise.
- Generate padding masks directly from the final warp and expose new metadata (`keep_fov_status`, `keep_fov_effective`, stabilization scale, effective padding ratios, etc.).

## Consequences
- Crop mode semantics become predictable: either `keep_fov` is met, or we clearly report why not.
- Stabilization strength may be softened automatically in extreme cases to prevent padding.
- Implementation touched both classic and flow nodes; documentation and regression coverage should stay in sync with the shared crop handling contract.
- Crop success is now validated using real post-warp coverage masks, ensuring metadata and padding alerts stay in sync with what the user sees.
- Edge handling (empty input, single frame, extreme keep_fov, non-square frames) is explicitly guarded so the nodes fail safe instead of emitting partial state.
