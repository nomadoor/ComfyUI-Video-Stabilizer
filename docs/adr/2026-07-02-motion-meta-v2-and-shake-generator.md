# ADR: Motion Meta v2, Motion Apply, and Shake Generator

Date: 2026-07-02

## Decision

Add `motion_meta` v2 as a JSON metadata contract. Motion Apply consumes matrices directly with `input_to_output` absolute pixel coordinates. Classic and Flow continue to emit `stabilization_warp` unchanged and add `motion_meta` as an additive field.

Add Video Stabilizer Motion Apply as the general consumer and Video Stabilizer Shake Generator as a deterministic producer. Keep Video Stabilizer Inverse for one release, mark it deprecated, and register a node replacement to Motion Apply.

## Rationale

The producer/consumer split lets estimated motion and generated motion use the same apply path. Keeping matrices as the only truth avoids drift between decomposed parameters and the warp actually applied.

## Compatibility

The socket type remains `io.Custom("JSON")`. Existing workflows keep their node ids and output shapes. Inverse keeps its public inputs/outputs while delegating to Motion Apply.

## Rejected or Deferred

Motion blur is deferred because v1 only needs deterministic geometric application. Advanced shake controls such as separate translation scale or rotation scale are deferred to keep the public node small until real workflow feedback exists.
