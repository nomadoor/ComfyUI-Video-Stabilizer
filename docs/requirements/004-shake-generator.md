# Shake Generator Requirements

Last updated: 2026-07-02 (UTC+9)

## Purpose

Video Stabilizer Shake Generator creates deterministic artificial handheld motion metadata. It does not alter or pass through frames; input frames are used only for frame count, resolution, and optional fps metadata.

## Controls

- `frames_context`: frame count and resolution source only
- `frame_rate`: fallback fps, default `16.0`
- `preset`: `handheld_subtle`, `handheld`, `handheld_rough`, `tripod_micro`, `telephoto`, `vibration`
- `strength`: `0.0..3.0`
- `speed`: `0.1..3.0`
- `detail`: `0.0..1.0`
- `seed`: ComfyUI fixed seed control

## Determinism

The same `(frame_count, width, height, fps, preset, strength, speed, detail, seed)` must produce identical JSON. The implementation uses only a local NumPy generator created from `seed`; global random APIs, time, and hidden state are not allowed.

## Presets

| preset | tx/ty amp (x base) | rot amp (deg) | zoom amp (log) | low/mid/high weights | low/mid/high Hz |
| --- | --- | ---: | ---: | --- | --- |
| handheld_subtle | 0.003 / 0.003 | 0.10 | 0.001 | 1.0 / 0.5 / 0.3 | 0.3 / 1.1 / 3.5 |
| handheld | 0.006 / 0.006 | 0.20 | 0.002 | 1.0 / 0.7 / 0.4 | 0.3 / 1.1 / 3.5 |
| handheld_rough | 0.012 / 0.010 | 0.45 | 0.004 | 0.9 / 1.0 / 0.7 | 0.4 / 1.3 / 4.0 |
| tripod_micro | 0.0008 / 0.0008 | 0.03 | 0.0003 | 1.0 / 0.3 / 0.2 | 0.15 / 0.9 / 3.0 |
| telephoto | 0.002 / 0.002 | 0.35 | 0.001 | 1.0 / 0.6 / 0.5 | 0.25 / 1.0 / 3.5 |
| vibration | 0.004 / 0.004 | 0.08 | 0.001 | 0.2 / 0.5 / 1.5 | 0.5 / 2.5 / 7.0 |

Frequencies are multiplied by `speed` and clamped to `fps / 2` to avoid aliasing.
