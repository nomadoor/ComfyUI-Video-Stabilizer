# ComfyUI Video Stabilizer

動画の手ぶれ補正を行う ComfyUI 向けカスタムノードです

## 概要

* 2種類のノードがあります

  * **Classic（VideoStabilizerClassic）**: 特徴点追跡と幾何変換（OpenCV/NumPy）による軽量な補正
  * **Flow（VideoStabilizerFlow）**: DIS Optical Flow（稠密オプティカルフロー）に基づく高精度補正です。計算コストは高め
* 画角は「クロップで吸収」または「クロップ最小 + パディング」で吸収します
* **CROP_and_PAD** ではパディング領域をマスクで出力するため、VACE 等の outpainting と連携できます

## ノード一覧

### 1) VideoStabilizerClassic

- OpenCV と NumPy による汎用的な手ぶれ補正です

**パラメータ**

* `method`（幾何モデル）

  * `translation`：平行移動のみです。最もロバストで軽量ですが回転やスケール変化には弱いです
  * `similarity`：平行移動と回転と等方スケールに対応します。多くのケースで推奨します
  * `perspective`：射影変換（8 自由度）に対応します。過補正や歪みのリスクが高いため基本は非推奨です
* `framing`（画角処理）

  * `CROP`：はみ出しをクロップで吸収。画角が狭くなります
  * `CROP_and_PAD`：クロップ量を抑え、はみ出しをパディングで補います
* `smoothness`（0.0〜1.0）

  * 軌跡平滑化の強度です。数値が大きいほど「ねっとり」としたカメラワークになります
  * 0.2〜0.5 が自然で、0.6 以上はジンバル風になります
* `stabilize_zoom`（0.0〜1.0）

  * クロップ許容量です
  * `CROP` で 0.0 にするとクロップが行われず、実質ほぼ無効化になります
  * `CROP_and_PAD` で 0.0 にするとクロップせずパディングのみで吸収します

**出力**

* 安定化済みフレーム列または動画を出力します
* `CROP_and_PAD` 選択時は **パディング領域マスク**（白=パディング/黒=元画）も出力します

### 2) VideoStabilizerFlow

- OpenCV の **DIS Optical Flow** による稠密フローで安定化します。Classic より高精度ですが計算は重いです

**パラメータ**

* `smoothness` / `framing` / `stabilize_zoom` は Classic と同様です

## VACE との連携（outpainting）

`CROP_and_PAD` ではパディング領域の **マスク** を同時出力します。これを WAN VACE 等の outpainting に渡すことで、**画角を犠牲にせず** 手ぶれ補正された動画を作成できます

**サンプル workflow**

1. `Load Video` → `VideoStabilizerClassic`（framing=`CROP_and_PAD`, stabilize_zoom=低）
2. 安定化動画とマスクを `VACE (outpainting)` に入力します
3. 必要に応じて `Composite/Blend` で合成します
