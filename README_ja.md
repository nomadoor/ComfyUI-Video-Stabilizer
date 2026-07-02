# ComfyUI Video Stabilizer

[![English](https://img.shields.io/badge/README-English-gray.svg)](README.md)

CPU で扱いやすい動画手ぶれ補正、padding mask 出力、再利用できる motion metadata、元の動きへ戻す workflow のための ComfyUI custom node です。

ノードは6種類です。

- **Classic**: OpenCV / NumPy による特徴点トラッキング
- **Flow**: OpenCV DIS を標準で使う dense optical flow
- **Motion Apply**: motion metadata をフレームへ適用
- **Shake Generator**: フレームを加工せず、style ベースの motion metadata を生成
- **Shake Generator Manual**: 明示的な recipe 値から同種の motion metadata を生成

https://github.com/user-attachments/assets/7da060c1-d775-47b7-91e6-f7a2ce147389

## インストール

- ComfyUI Manager からインストールしてください。

## ノード

| Node | 役割 |
| --- | --- |
| `Video Stabilizer Classic` | 特徴点トラッキングによる軽量な汎用 stabilizer。 |
| `Video Stabilizer Flow` | DIS optical flow による高精度 stabilizer。`cv2.optflow` が利用可能な場合のみ TV-L1 も使えます。 |
| `Video Stabilizer Motion Apply` | `motion_meta` JSON を crop / crop+pad / expand framing と optional motion blur でフレームへ適用します。 |
| `Video Stabilizer Shake Generator` | 決定的な shake `motion_meta` を出力します。`style` は揺れの種類、`amount` は強さです。 |
| `Video Stabilizer Shake Generator Manual` | pan / tilt / roll / zoom などの絶対値 recipe から shake `motion_meta` を出力します。 |
| `Video Stabilizer Inverse` | 元の手ブレを戻すための deprecated 互換ノード。 |

Flow は通常 DIS optical flow を使います。使えない場合は TV-L1、平行移動推定、identity の順に自動で fallback します。

## 使い方

動画、またはバッチ画像を `Video Stabilizer Classic` もしくは `Video Stabilizer Flow` に入力します。

`padding_mask` は、手ブレ補正でできた余白を VACE などで補完したいときに使います。

Classic/Flow の `motion_meta` と元フレームを `Video Stabilizer Motion Apply` へ接続すると同じ stabilization transform を再適用できます。補正済みフレームを編集したあと元キャンバスへ戻す場合も、埋め込まれた legacy warp metadata により Motion Apply で復元できます。生成した手持ち感を足す場合は Shake Generator の出力を Motion Apply へ接続します。

細かく調整する場合は、まず `Video Stabilizer Shake Generator` で style を試し、`motion_meta.generator.recipe` を確認して、その値を `Video Stabilizer Shake Generator Manual` に転記してから pan / tilt / roll / zoom などを調整します。

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
| `crop` | アスペクト比を保ったズーム/クロップで縁を隠します。画角は狭くなります。 |
| `crop_and_pad` | ズームを抑え、不足分を padding します。 |
| `expand` | 全く crop せず、必要な分だけキャンバスを拡張します。 |

Shake Generator:

| Parameter | Default | 説明 |
| --- | ---: | --- |
| `style` | `handheld` | `tripod`, `handheld`, `walking`, `action`, `vibration` から揺れの種類を選びます。 |
| `amount` | `1.0` | 揺れ全体の強さ。 |
| `speed` | `1.0` | 揺れ全体の速さ。 |
| `seed` | `0` | 決定的生成用の seed。 |

Shake Generator Manual は解決済み recipe を直接公開します: `pan`, `tilt`, `roll`, `zoom`, `drift_freq`, `tremor`, `tremor_freq`, `jitter_rate`, `step`, `randomness`, `virtual_fov`。

Motion Apply:

| Parameter | Default | 説明 |
| --- | ---: | --- |
| `framing_mode` | `crop_and_pad` | `crop`, `crop_and_pad`, `expand` から選択します。 |
| `interpolation` | `bilinear` | `bilinear` または `bicubic`。 |
| `motion_blur` | `0.0` | シャッター開角割合。`0.5` がだいたい 180度シャッター相当です。 |
| `motion_blur_quality` | `Standard` | `Draft`, `Standard`, `High`, `Ultra` から選びます。高品質ほど多くのシャッターサンプルを平均し、滑らかですが遅くなります。 |

## 出力

| Output | 説明 |
| --- | --- |
| `frames_stabilized` | 補正済みフレーム。 |
| `padding_mask` | padding / 欠損領域の mask。 |
| `meta` | 推定 motion、実際に適用した補正行列、Motion Apply 用の `motion_meta` block を含む JSON 診断情報。 |

## Inverse stabilization

`Video Stabilizer Motion Apply` が旧 inverse workflow の置き換えです。`Video Stabilizer Inverse` は deprecated 互換ノードとして残しています。

`crop` / `crop_and_pad` では仕組み上、最後にほぼ必ず隙間ができます。Inverse まで使う場合は `expand` がオススメです。

## サンプル Workflow

- [Wan2.1_VACE_outpainting_VideoStabilizer.json](example_workflows/Wan2.1_VACE_outpainting_VideoStabilizer.json)
- [Wan2.2-VACE-Fun_outpainting_VideoStabilizer.json](example_workflows/Wan2.2-VACE-Fun_outpainting_VideoStabilizer.json)
- [Sample Video (Pexels)](https://www.pexels.com/ja-jp/video/29507473/)

## License

MIT
