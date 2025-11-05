# ComfyUI Video Stabilizer (Classic / Flow)

[![日本語版](https://img.shields.io/badge/README-日本語版-gray.svg)](README_ja.md)

## Overview

https://github.com/user-attachments/assets/598a407f-398e-4540-8d0c-c910f9bcc087

* A video stabilization node for ComfyUI
* Implements two approaches: **Classic (feature points + LK)** and **Flow (DIS Optical Flow)**
* Supports three framing modes:

  * **crop**: hide shake at the cost of field-of-view (FOV) by zooming/cropping
  * **crop_and_pad**: limit zoom as much as possible, then pad the remainder
  * **expand**: never crop; extend the canvas so all stabilized frames are fully contained
* Padding is **output as a mask**, so you can pass it to outpainting tools such as VACE

---

## Node List

* **Video Stabilizer (Classic)** - Lightweight, general-purpose stabilization using OpenCV / NumPy
* **Video Stabilizer (Flow)** - Higher-accuracy stabilization based on OpenCV **DIS Optical Flow** (somewhat heavier on CPU)

---

## Parameters (shared by Classic / Flow)

* **transform_mode**

  * `translation`: X/Y translation only (most robust and lightweight)
  * `similarity`: translation + rotation + uniform scale (recommended for many cases)
  * `perspective`: full projective transform (8 DoF). Often fragile; not generally recommended
* **framing_mode** (FOV handling)

  * `crop`: hide edges by zooming (narrows FOV)
  * `crop_and_pad`: avoid over-zooming and **pad** what still exceeds the frame
  * `expand`: never crop; add padding across all frames so the entire trajectory fits
* **camera_lock** (bool)

  * ON: enforce a tripod-like look
  * This is a separate solver from normal stabilization, so the two knobs below are disabled while ON
* **strength** (0.0 to 1.0)

  * **Removal gain** of the estimated camera motion (how much to take out)
* **smooth** (0.0 to 1.0)

  * Temporal **smoothing strength**. Higher values reduce jitter but yield a more "viscous" camera motion
* **keep_fov** (0.0 to 1.0, **used only when `framing_mode=crop`**)

  * **1.0 = preserve the input FOV (no zoom)**
  * **0.0 = allow maximum zoom to hide edges**
* **padding_color** (RGB)

  * Fill color for outer regions in `crop_and_pad` / `expand` (e.g., `127,127,127`)

---

## Outputs

* **frames_stabilized**: the stabilized video
* **padding_mask**: padding regions are emitted as a mask for `crop_and_pad` / `expand`
* **meta (JSON)**: diagnostics such as estimated/applied transforms, confidences, and zoom/padding ratios

---

## Using with VACE (outpainting)

* With `framing_mode=crop_and_pad` or `expand`, pass the emitted **padding_mask** to VACE (or similar) to **restore borders without sacrificing FOV** after stabilization

**Sample Workflow**

* [Wan2.1_VACE_outpainting_VideoStabilizer.json](example_workflows\Wan2.1_VACE_outpainting_VideoStabilizer.json)
* [Wan2.2-VACE-Fun_outpainting_VideoStabilizer.json](example_workflows\Wan2.2-VACE-Fun_outpainting_VideoStabilizer.json)
* [Sample_Video (Pexels)](https://www.pexels.com/ja-jp/video/29507473/)


