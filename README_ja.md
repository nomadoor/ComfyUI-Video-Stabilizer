# ComfyUI Video Stabilizer

[![English](https://img.shields.io/badge/README-English-gray.svg)](README.md)

CPU で扱いやすい動画手ぶれ補正、padding mask 出力、元の動きへ戻す inverse workflow のための ComfyUI custom node です。

補正方法別に２種類のノードがあります。

- **Classic**: OpenCV / NumPy による特徴点トラッキング
- **Flow**: OpenCV DIS を標準で使う dense optical flow

https://github.com/user-attachments/assets/7da060c1-d775-47b7-91e6-f7a2ce147389

## インストール

- ComfyUI Manager からインストールしてください。

## ノード

| Node | 役割 |
| --- | --- |
| `Video Stabilizer (Classic)` | 特徴点トラッキングによる軽量な汎用 stabilizer。 |
| `Video Stabilizer (Flow)` | DIS optical flow による高精度 stabilizer。`cv2.optflow` が利用可能な場合のみ TV-L1 も使えます。 |
| `Video Stabilizer Inverse` | 補正した分の手ブレを、編集後のフレームに戻します。 |

Flow は通常 DIS optical flow を使います。使えない場合は TV-L1、平行移動推定、identity の順に自動で fallback します。

## 使い方

動画、またはバッチ画像を `Video Stabilizer (Classic)` もしくは `Video Stabilizer (Flow)` に入力します。

`padding_mask` は、手ブレ補正でできた余白を VACE などで補完したいときに使います。

## パラメータ

Classic / Flow 共通:

| Parameter | Default | 説明 |
| --- | ---: | --- |
| `frame_rate` | `16.0` | 時間方向の smoothing window を入力 FPS に合わせます。 |
| `framing_mode` | `crop_and_pad` | `crop`, `crop_and_pad`, `expand` から選択します。 |
| `transform_mode` | `similarity` | `translation`, `similarity`, `perspective` から選択します。 |
| `camera_lock` | `false` | 三脚で撮ったような結果へ強めに寄せます。 |
| `strength` | `0.7` | 推定したカメラ運動をどれだけ除去するか。`camera_lock` 中は無視されます。 |
| `smooth` | `0.5` | 時間方向の平滑化量。`camera_lock` 中は無視されます。 |
| `keep_fov` | `0.6` | crop mode の画角維持量。`1.0` はズームなし、`0.0` は最大ズーム許容。 |
| `padding_color` | `#7F7F7F` | padding 領域の HEX 色。core Color Picker の `hex` 出力を接続できます。 |

Framing mode:

| Mode | 動作 |
| --- | --- |
| `crop` | ズーム/クロップで縁を隠します。画角は狭くなります。 |
| `crop_and_pad` | ズームを抑え、不足分を padding します。 |
| `expand` | 全く crop せず、必要な分だけキャンバスを拡張します。 |

## 出力

| Output | 説明 |
| --- | --- |
| `frames_stabilized` | 補正済みフレーム。 |
| `padding_mask` | padding / 欠損領域の mask。 |
| `meta` | 推定 motion と実際に適用した補正行列を含む JSON 診断情報。 |

## Inverse stabilization

`Video Stabilizer Inverse` は、手ブレ補正した動画に後処理をしたあと、補正した分の手ブレをもう一度戻すためのノードです。

`crop` / `crop_and_pad` では仕組み上、最後にほぼ必ず隙間ができます。Inverse まで使う場合は `expand` がオススメです。

## サンプル Workflow

- [Wan2.1_VACE_outpainting_VideoStabilizer.json](example_workflows/Wan2.1_VACE_outpainting_VideoStabilizer.json)
- [Wan2.2-VACE-Fun_outpainting_VideoStabilizer.json](example_workflows/Wan2.2-VACE-Fun_outpainting_VideoStabilizer.json)
- [Sample Video (Pexels)](https://www.pexels.com/ja-jp/video/29507473/)

## License

MIT
