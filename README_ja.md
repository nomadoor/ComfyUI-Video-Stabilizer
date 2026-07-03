# ComfyUI Video Stabilizer

[![English](https://img.shields.io/badge/README-English-gray.svg)](README.md)

ComfyUI 上で動画の手ブレ補正を行うための custom node です。

主に以下の3つの機能があります。

- 動画の手ブレを補正する
- 手ブレ補正した動画から、元の手ブレを復元する
- 人工的な手ブレを加える

https://github.com/user-attachments/assets/7da060c1-d775-47b7-91e6-f7a2ce147389

## ノード

| Node | 役割 |
| --- | --- |
| `Video Stabilizer Classic` | 特徴点トラッキングによる軽量な stabilizer。 |
| `Video Stabilizer Flow` | dense optical flow による高精度な stabilizer。 |
| `Video Stabilizer Motion Apply` | `motion_meta` をフレームへ適用します(再適用・復元・生成した手ブレの付与)。 |
| `Video Stabilizer Shake Generator` | 決定的な手ブレ風 `motion_meta` を生成します。フレームには一切手を加えません。 |
| `Video Stabilizer Shake Generator Manual` | 同上、pan/tilt/roll/zoom を直接指定して細かく調整できます。 |

## インストール

- ComfyUI Manager からインストールしてください。

## 使い方

### 手振れ補正

動画、またはバッチ画像を `Video Stabilizer Classic` か `Video Stabilizer Flow` に入力するだけで補正できます。
`Classic` は特徴点トラッキングによる軽量な方式、`Flow` は dense optical flow を使うぶん重いですが精度が高い方式です。処理速度に大きな不満が無いのなら `Flow` を使っておけば間違いないでしょう。

主なパラメータ:

| Parameter | 説明 |
| --- | --- |
| `framing_mode` | 補正ではみ出した部分をどう処理するか。`crop`(ズームして隠す・画角が狭くなる)/ `crop_and_pad`(ズームを抑えて余白を padding する)/ `expand`(一切 crop せず、必要な分だけキャンバスを広げる)。 |
| `strength` | カメラの動きをどれだけ取り除くか。`1.0` に近いほど強く止まった映像になります。 |
| `smooth` | 時間方向の平滑化の強さ。大きいほどゆったりした動きだけが残ります。 |
| `camera_lock` | オンにすると三脚固定に近い、より強い補正になります。`strength` / `smooth` の代わりに使います。 |
| `keep_fov` | `crop` 系のモードでどこまで画角を犠牲にしてよいか。`1.0` はズームなし、`0.0` は必要なだけズームを許可します。 |
| `padding_color` | 余白を埋める色。 |

### 補正した手ブレを戻す

`Classic` / `Flow` の `meta` 出力には、取り除いたカメラの動きがそのまま記録されています。これを `Video Stabilizer Motion Apply` に接続すると、同じ動きを再現できます。

たとえば、一度手ブレを補正したフレームを動画生成 AI で編集・生成し直したあと、その結果に「元のカメラワーク」をもう一度乗せたい場合に使います。手ブレを消して作業しやすくしてから、最後に自然なカメラの揺れへ戻す、という流れです。

### 人工的な手ブレを追加する

逆に、静止した映像に手持ち感を足したいときは `Video Stabilizer Shake Generator` で揺れの `motion_meta` を作り、`Video Stabilizer Motion Apply` に接続して適用します。

主なパラメータ:

| Parameter | 説明 |
| --- | --- |
| `style` | 揺れの種類。`tripod`(ほぼ静止)/ `handheld`(自然な手持ち)/ `walking`(歩きながらの上下動)/ `action`(激しい動き)/ `vibration`(車内などの機械的な振動)。 |
| `amount` | 揺れ全体の強さ。 |
| `speed` | 揺れ全体の速さ。 |
| `seed` | 同じ値なら毎回同じ揺れを再現できます。 |

もっと細かく調整したい場合は `Video Stabilizer Shake Generator Manual` を使ってください。`style` を選ぶ代わりに、pan / tilt / roll / zoom や揺れの周波数などを数値で直接指定できます。

### 動画生成AIと組み合わせる

一般的に手振れ補正をすると、はみ出す部分がでてくるため、クロップすることでそれを隠します。が、当然ながら画角が犠牲になります。
このノードははみ出た部分をマスクとして出力するため、動画生成AIでこの部分をoutpaintingすれば、画角を犠牲にせず手振れ補正することが可能です。

サンプル workflow:

- [Wan2.1_VACE_outpainting_VideoStabilizer.json](example_workflows/Wan2.1_VACE_outpainting_VideoStabilizer.json)
- [Wan2.2-VACE-Fun_outpainting_VideoStabilizer.json](example_workflows/Wan2.2-VACE-Fun_outpainting_VideoStabilizer.json)

([サンプル動画](https://www.pexels.com/ja-jp/video/29507473/))

## より詳しく知りたい方へ

パラメータによる挙動の違いや簡単な workflow は、以下の記事で解説しています。

- [日本語](https://comfyui.nomadoor.net/ja/notes/comfyui-video-stabilizer/)
- [English](https://comfyui.nomadoor.net/en/notes/comfyui-video-stabilizer/)
- [中文](https://comfyui.nomadoor.net/zh/notes/comfyui-video-stabilizer/)

## License

MIT
