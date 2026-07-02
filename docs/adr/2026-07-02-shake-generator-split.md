# ADR: Split Shake Generator Preset and Manual Controls

Date: 2026-07-02

## Decision

Split shake generation into two nodes:

- `Video Stabilizer Shake Generator`: simple style-based controls.
- `Video Stabilizer Shake Generator Manual`: explicit absolute recipe controls.

Remove reliance on `advanced=True` for UX correctness. Motion Apply's `motion_blur_samples` is also a normal widget.

## Rationale

The previous one-node design assumed `style` could coexist with hidden advanced multipliers. That failed for two reasons.

First, Python-only ComfyUI nodes cannot update other widget values when a preset widget changes. Professional tools load preset values into editable controls, but doing that here would require custom frontend JavaScript, which this package avoids.

Second, multipliers are not a substitute for editable preset values. A list of `1.0` multipliers hides the real recipe and makes the controls look inert, especially in legacy frontend modes where advanced widgets are not hidden.

## Consequences

Style presets and manual editing are separate workflows. To bridge them, generated metadata records `generator.recipe`, the resolved absolute values before `amount` and `speed` are applied. Users can inspect that recipe and enter the same values in the Manual node.

This design works consistently across ComfyUI frontend generations because it does not depend on widget hiding or dynamic widget updates.
