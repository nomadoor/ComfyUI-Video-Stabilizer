# Shake Generator Requirements

Last updated: 2026-07-02 (UTC+9)

## Purpose

Video Stabilizer Shake Generator creates deterministic artificial camera motion metadata. It does not alter or pass through frames; input frames are used only for frame count, resolution, and optional fps metadata.

## Controls

Basic controls:

- `frames_context`: frame count and resolution source only
- `frame_rate`: fallback fps, default `16.0`
- `style`: `tripod | handheld | walking | action | vibration`, default `handheld`
- `amount`: global motion amplitude multiplier, `0.0..3.0`
- `pace`: global frequency multiplier, `0.1..3.0`
- `seed`: ComfyUI fixed seed control

Advanced controls are style-relative multipliers unless noted:

- `pan_amount`, `tilt_amount`, `roll_amount`, `zoom_amount`: channel multipliers, `0.0..3.0`
- `drift_amount`, `tremor_amount`, `jitter_amount`: component multipliers, `0.0..3.0`
- `randomness`: slow amplitude variation depth, `0.0..1.0`
- `virtual_fov`: projection FOV in degrees, `10.0..120.0`

## Motion Model

Motion is generated in camera-angle space:

- `pan_deg`
- `tilt_deg`
- `roll_deg`
- `zoom_log`

The angle signal is projected to an absolute pixel-coordinate similarity matrix:

```text
f  = 0.5 * min(width, height) / tan(radians(virtual_fov) / 2)
tx = f * tan(radians(pan_deg))
ty = f * tan(radians(tilt_deg))
M  = T(cx + tx, cy + ty) @ R(roll_deg) @ S(exp(zoom_log)) @ T(-cx, -cy)
```

Each channel combines:

- `drift`: low-frequency holding sway
- `tremor`: physiological high-frequency shake
- `jitter`: occasional impulse events
- `step`: walking rhythm for the `walking` style only

Every generated channel subtracts its first-frame value, so `per_frame[0].matrix` is identity.

## Styles

Values below are the amount `1.0` baseline. Pan, tilt, and roll are degrees; zoom is log scale.

| style | pan/tilt amp | roll amp | zoom amp | drift Hz | tremor ratio | tremor Hz | jitter /s | step |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| tripod | 0.03 / 0.03 | 0.02 | 0.0002 | 0.20 | 0.15 | 4.0 | 0.0 | no |
| handheld | 0.40 / 0.33 | 0.50 | 0.0030 | 0.35 | 0.35 | 5.0 | 0.0 | no |
| walking | 0.46 / 0.60 | 0.70 | 0.0040 | 0.30 | 0.30 | 5.0 | 0.0 | yes |
| action | 0.80 / 0.66 | 1.00 | 0.0060 | 0.50 | 0.80 | 6.0 | 0.5 | no |
| vibration | 0.15 / 0.15 | 0.10 | 0.0010 | off | 1.00 | 8.0 | 0.0 | no |

`style` controls the quality of motion. `amount` controls how much of that motion is applied.

## Determinism

The same `(frame_count, width, height, fps, style, amount, pace, seed, advanced controls)` must produce identical JSON. The implementation uses only a local NumPy generator created from `seed`; global random APIs, time, and hidden state are not allowed.

Frequencies are multiplied by `pace` and clamped to `fps / 2` to avoid aliasing.
