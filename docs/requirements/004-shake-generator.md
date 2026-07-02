# Shake Generator Requirements

Last updated: 2026-07-02 (UTC+9)

## Purpose

Shake Generator nodes create deterministic artificial camera motion metadata. They do not alter or pass through frames; input frames are used only for frame count, resolution, and optional fps metadata.

## Nodes

### Video Stabilizer Shake Generator

Simple style-based producer:

- `frames_context`: frame count and resolution source only
- `frame_rate`: fallback fps, default `16.0`
- `style`: `tripod | handheld | walking | action | vibration`, default `handheld`
- `amount`: global motion amplitude multiplier, `0.0..3.0`
- `pace`: global frequency multiplier, `0.1..3.0`
- `seed`: ComfyUI fixed seed control

### Video Stabilizer Shake Generator Manual

Manual absolute-value producer:

- `frames_context`: frame count and resolution source only
- `frame_rate`: fallback fps, default `16.0`
- `pan`: yaw amplitude in degrees, `0.0..5.0`
- `tilt`: pitch amplitude in degrees, `0.0..5.0`
- `roll`: in-plane rotation amplitude in degrees, `0.0..5.0`
- `zoom`: log-scale zoom amplitude, `0.0..0.05`
- `drift_freq`: low-frequency holding sway in Hz, `0.05..2.0`
- `tremor`: high-frequency shake ratio relative to pan/tilt/roll amplitudes, `0.0..2.0`
- `tremor_freq`: high-frequency shake in Hz, `1.0..15.0`
- `jitter_rate`: impulse event rate per second, `0.0..3.0`
- `step`: walking bounce amplitude in degrees, `0.0..2.0`
- `randomness`: slow amplitude variation depth, `0.0..1.0`
- `virtual_fov`: projection FOV in degrees, `10.0..120.0`
- `amount`: global motion amplitude multiplier, `0.0..3.0`
- `pace`: global frequency multiplier, `0.1..3.0`
- `seed`: ComfyUI fixed seed control

Manual defaults match the `handheld` style exactly.

## Recipe Metadata

Both nodes write resolved absolute values to `motion_meta.generator.recipe`:

```json
{
  "node": "shake_generator",
  "style": "walking",
  "amount": 1.0,
  "pace": 1.0,
  "seed": 0,
  "recipe": {
    "pan": 0.46,
    "tilt": 0.60,
    "roll": 0.70,
    "zoom": 0.004,
    "drift_freq": 0.30,
    "tremor": 0.30,
    "tremor_freq": 5.0,
    "jitter_rate": 0.0,
    "step": 0.60,
    "randomness": 0.3,
    "virtual_fov": 60.0
  }
}
```

The recipe is recorded before `amount` and `pace` are applied. Users can copy it into the Manual node to continue from a style preset with visible, editable absolute values.

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

Every generated channel subtracts its first-frame value, so `per_frame[0].matrix` is identity.

## Styles

Values below are the amount `1.0` baseline. Pan, tilt, roll, and step are degrees; zoom is log scale.

| style | pan | tilt | roll | zoom | drift Hz | tremor | tremor Hz | jitter /s | step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tripod | 0.03 | 0.03 | 0.02 | 0.0002 | 0.20 | 0.15 | 4.0 | 0.0 | 0.0 |
| handheld | 0.40 | 0.33 | 0.50 | 0.0030 | 0.35 | 0.35 | 5.0 | 0.0 | 0.0 |
| walking | 0.46 | 0.60 | 0.70 | 0.0040 | 0.30 | 0.30 | 5.0 | 0.0 | 0.60 |
| action | 0.80 | 0.66 | 1.00 | 0.0060 | 0.50 | 0.80 | 6.0 | 0.5 | 0.0 |
| vibration | 0.15 | 0.15 | 0.10 | 0.0010 | 0.00 | 1.00 | 8.0 | 0.0 | 0.0 |

`style` controls the quality of motion. `amount` controls how much of that motion is applied.

## Determinism

The same `(frame_count, width, height, fps, recipe, amount, pace, seed)` must produce identical JSON. The implementation uses only a local NumPy generator created from `seed`; global random APIs, time, and hidden state are not allowed.

Frequencies are multiplied by `pace` and clamped to `fps / 2` to avoid aliasing.
