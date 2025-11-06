# ComfyUI Video Stabilizer

## 概要

https://github.com/user-attachments/assets/7da060c1-d775-47b7-91e6-f7a2ce147389

* ComfyUI 向けの動画手ぶれ補正ノードです
* **Classic（特徴点 + LK）** と **Flow（DIS Optical Flow）** の 2つの手法を実装しています
* 3方式のフレーミングがあります：

  * 画角をクロップで吸収する **crop**
  * 可能な範囲はズームで吸収し、残りをパディングする **crop_and_pad**
  * 入力を全くクロップしないようにキャンバスを拡張する **expand**
* パディングは**マスクとして出力されるため**、VACE 等の outpainting と連携できます

---

## ノード一覧

* **Video Stabilizer (Classic)** — OpenCV / NumPy による軽量・汎用のスタビライズ
* **Video Stabilizer (Flow)** — OpenCV **DIS Optical Flow** に基づく高精度スタビライズ（CPUでやや重め）

---

## パラメータ（UIは Classic / Flow 共通）

* **frame_rate** (float, default 16.0)

  * 平滑化に使う時間窓を入力映像の FPS に合わせて調整します。値を上げると 30/60/120fps の素材でも体感的な平滑度が一定になるように働きます。
* **transform_mode**

  * `translation`：平行移動のみ（最もロバスト・軽量）
  * `similarity`：平行移動 + 回転 + 等方スケール（多くのケースで推奨）
  * `perspective`：射影変換（8自由度）。破綻することが多く、おすすめしません
* **framing_mode**（画角処理）

  * `crop`：ズームで縁を隠します（画角は狭くなる）
  * `crop_and_pad`：ズームしないようにし、足りない分はパディングします
  * `expand`：全くクロップをせず、全フレームのブレを吸収できるよう、全フレームにパディングを追加します（キャンバスが拡張されるため、出力解像度は入力より大きくなることがあります）
* **camera_lock**（bool）

  * ON：三脚で撮ったような動画になるよう補正します
  * 通常スタビライズとは別処理のため、ON の間は下記2つのノブは無効化されます
* **strength**（0.0〜1.0）

  * 推定カメラ運動の **除去量**（どれだけ取り去るか）
* **smooth**（0.0〜1.0）

  * 時間方向の **平滑化強度**。大きいほどガタが減り、ネットリしたカメラワークになります
* **keep_fov**（0.0〜1.0、`framing_mode=crop`のときのみ使用）

  * **1.0 = 入力と同等の画角を維持（ズームしない）**
  * **0.0 = 縁を隠すため最大限ズーム許容**
* **padding_color**（RGB）

  * `crop_and_pad` / `expand` の外側塗りつぶし色（例 `127,127,127`）

---

## 出力

* **frames_stabilized**：補正済みの動画
* **padding_mask**：`crop_and_pad` / `expand` でのパディング領域がマスクとして出力されます
* **meta (JSON)**：推定/適用変換、信頼度、ズーム/パッド比率などの診断情報

---

## VACE との連携（outpainting）

* `framing_mode=crop_and_pad` または `expand` では、出力される **padding_mask** を VACE 等に渡すことで、**画角を犠牲にせず** 手ぶれ補正後の縁を補完できます

**サンプル ワークフロー**

* [Wan2.1_VACE_outpainting_VideoStabilizer.json](example_workflows/Wan2.1_VACE_outpainting_VideoStabilizer.json)
* [Wan2.2-VACE-Fun_outpainting_VideoStabilizer.json](example_workflows/Wan2.2-VACE-Fun_outpainting_VideoStabilizer.json)
* [Sample_Video (Pexels)](https://www.pexels.com/ja-jp/video/29507473/)

---
