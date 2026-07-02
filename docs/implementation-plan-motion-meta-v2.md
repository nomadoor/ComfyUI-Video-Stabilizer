# ComfyUI-Video-Stabilizer: motion meta v2 / Motion Apply / Shake Generator 実装計画

実装者: Codex。この計画は設計を確定済みとして書いてある。実装中に矛盾を見つけた場合のみ設計に戻ること。

## Context

現状は「安定化(Classic/Flow)」と「逆安定化(Inverse)」の3ノード。これを「カメラモーションmetaの producer / consumer」構成に再編する:

- **Video Stabilizer Classic / Flow** — 安定化 + motion meta 出力(producer)
- **Video Stabilizer Shake Generator**(新規)— 人工手ブレ motion meta を生成(producer、フレームは加工しない)
- **Video Stabilizer Motion Apply**(新規、Inverseの一般化)— motion meta をフレームに適用(consumer)

旧 `Video Stabilizer Inverse` は deprecated として1リリース残し、Node Replacement API で Motion Apply へ移行させる。既存workflow・既存meta形式は壊さない(AGENTS.md の互換要件)。

### 調査済みの事実(実装前の再確認は不要)

- 既存metaは `meta["stabilization_warp"]`: `source_size`, `output_size`, `framing_mode`, `matrix_convention: "source_to_stabilized"`, `per_frame[].{index, applied_matrix(3x3)}`。ビルダーは `_build_stabilization_warp_meta` ([nodes/stabilizer_utils.py:868-888](nodes/stabilizer_utils.py#L868))。全framing_modeで applied_matrix は「元キャンバス→最終出力キャンバス(crop/expandのキャンバス変化込み)」の写像。
- 旧Inverseは applied_matrix を `np.linalg.inv` して `source_size` キャンバスへ warp する(`_apply_inverse_stabilization`, [nodes/stabilizer_utils.py:921-999](nodes/stabilizer_utils.py#L921))。入力フレームサイズ == `output_size` を要求。
- meta socket 型は `io.Custom("JSON")`。**新custom型は作らない**(既存接続が壊れるため)。
- Node Replacement API はローカルComfyUIに存在: `io.NodeReplace`(`comfy_api/latest/_io.py:2255`)、登録は extension の `on_load` 内で `await api.node_replacement.register(...)`(パターンは `comfy_extras/nodes_replacements.py`)。`input_mapping` は `{"new_id","old_id"}` / `{"new_id","set_value"}`、`output_mapping` は `{"new_idx","old_idx"}`。
- V3 seed: `io.Int.Input("seed", ..., control_after_generate=io.ControlAfterGenerate.fixed)`(例: `comfy_extras/nodes_seed.py:16`)。enumは fixed/increment/decrement/randomize。**独自seed機構は作らない。**
- `io.Schema` に `is_deprecated: bool = False` がある(`comfy_api/latest/_io.py:1591`)。
- 再利用ヘルパー: `_normalize_video_input`(dict/sequence→`VideoContext`、dictからfps抽出)、`_reconstruct_video`、`_convert_masks_for_output`、`_parse_padding_color`、`_compute_bounding_boxes`、warpパターン(`cv2.warpPerspective` + `BORDER_CONSTANT`、maskはonesをINTER_NEARESTでwarpして反転)。
- example_workflows は flow のみ使用、inverse を使うworkflowは無い。既存2ファイルは変更不要。
- テストは無し。検証は `scripts/validate_repo.py` が `scripts/check_*.py` をsubprocess実行する形式。新チェックもこの形式で足す。

---

## 1. 設計の核: motion_meta v2 の意味論

**v2 motion_meta は「Motion Apply が入力フレームにそのまま適用すべき変換」を表す。** 消費側に分岐を持たせないための唯一のルール。

```json
"motion_meta": {
  "version": 2,
  "source": "estimated_classic" | "estimated_flow" | "generated_shake" | "legacy_stabilization",
  "frame_count": 81,
  "fps": 16.0,
  "input_size": [1024, 576],
  "output_size": [1024, 576],
  "matrix_convention": "input_to_output",
  "per_frame": [
    {"index": 0, "matrix": [[1,0,0],[0,1,0],[0,0,1]]}
  ],
  "generator": {
    "node": "shake_generator",
    "style": "handheld",
    "amount": 1.0,
    "speed": 1.0,
    "seed": 0,
    "recipe": {
      "pan": 0.40,
      "tilt": 0.33,
      "roll": 0.50,
      "zoom": 0.0030,
      "drift_freq": 0.35,
      "tremor": 0.35,
      "tremor_freq": 5.0,
      "jitter_rate": 0.0,
      "step": 0.0,
      "randomness": 0.3,
      "virtual_fov": 60.0
    }
  }
}
```

決定事項(変更禁止):

- matrixは**絶対ピクセル座標の3x3**(既存metaと同じ流儀)。`center` や `params`(tx/ty/rot分解)は**入れない**。matrixが唯一の真実。
- `input_size` = Apply への入力フレームサイズ(要検証)、`output_size` = Apply の出力キャンバスサイズ。復元用途では両者が異なりうる。Shake用途では同一。
- `generator` は `source=="generated_shake"` のときのみ。
- **Classic/Flow が書き出す motion_meta の中身は「安定化時に実際に適用した変換」= `applied_matrix`**。つまり `input_size = stabilization_warp.source_size`、`output_size = stabilization_warp.output_size`、`per_frame[i].matrix = applied_matrix[i]`。これで「元フレーム + Classic/Flow の motion_meta を Motion Apply に繋ぐ = stabilizer出力の再適用」が成立する。旧Inverse互換は `stabilization_warp` fallback が `inv(applied_matrix)` を作ることで維持する。
- 旧 `stabilization_warp` は**そのまま残す**(削除・変更禁止)。v2は追加のみ。

### nodes/motion_meta.py(新規)

```python
@dataclass FrameTransform: index: int; matrix: np.ndarray  # 3x3 float64
@dataclass MotionMeta: source, frame_count, fps, input_size, output_size, per_frame, generator(optional)

build_motion_meta_v2(...) -> dict          # MotionMeta → JSON dict("motion_meta"ブロック)
resolve_motion_meta(meta: dict) -> MotionMeta  # 正規化の入口。消費側はこれだけ呼ぶ
validate_motion_meta(block: dict) -> None  # 型・サイズ・per_frame長・3x3・可逆性チェック(明確なValueError)
```

`resolve_motion_meta` の解決順:

1. `meta["motion_meta"]` があれば validate して MotionMeta へ。
2. なければ `meta["stabilization_warp"]`(`matrix_convention=="source_to_stabilized"` を要求)を読み、per-frame matrix を `np.linalg.inv` で反転、`input_size=output_size(warp)`, `output_size=source_size(warp)`, `source="legacy_stabilization"` として MotionMeta へ。既存 `_read_applied_matrix` / `_read_size_pair` のバリデーションと同等のエラーメッセージ品質を保つ(流用してよい)。
3. どちらも無ければ `ValueError("meta must contain motion_meta or stabilization_warp.")`。

非可逆matrix は `np.linalg.LinAlgError` を捕まえて frame index 付き ValueError(既存Inverseと同様)。

---

## 2. ノードID / 表示名

| node_id(不変 or 新規) | display_name | 備考 |
|---|---|---|
| `video_stabilizer_classic`(維持) | `Video Stabilizer Classic` | display_nameの括弧を外すだけ |
| `video_stabilizer_flow`(維持) | `Video Stabilizer Flow` | 同上 |
| `video_stabilizer_motion_apply`(新規) | `Video Stabilizer Motion Apply` | |
| `video_stabilizer_shake_generator`(新規) | `Video Stabilizer Shake Generator` | |
| `video_stabilizer_inverse`(維持・deprecated) | `Video Stabilizer Inverse` | `is_deprecated=True`、1リリース後に削除予定 |

カテゴリは全ノード既存どおり `Video/Stabilization`。

---

## 3. Video Stabilizer Motion Apply(新規ノード)

ファイル: `nodes/video_stabilizer_motion_apply.py` + 共有処理 `nodes/motion_apply.py`

### スキーマ

入力(この順):
1. `frames`: `io.Image.Input`(socket)
2. `motion_meta`: `io.Custom("JSON").Input`(socket)— 型名はJSONのまま
3. `framing_mode`: `io.Combo.Input(["crop_and_pad", "crop", "expand"], default="crop_and_pad")`
4. `interpolation`: `io.Combo.Input(["bilinear", "bicubic"], default="bilinear")`
5. `padding_color`: `io.Color.Input(default="#7F7F7F")`

出力: `frames`(Image, "Frames") / `padding_mask`(Mask, "Padding Mask") / `meta`(JSON, "Meta")

motion blur は Motion Apply の pixel application concern として実装する。

### 挙動(nodes/motion_apply.py の `apply_motion()`)

1. `resolve_motion_meta(meta)` で正規化。
2. 入力フレームサイズ == `input_size`、フレーム数 == `frame_count` を検証(不一致は明確なValueError。旧Inverseのメッセージ水準)。
3. **crop_and_pad モード**(デフォルト): 各フレームを `cv2.warpPerspective(frame, matrix, output_size, flags=interp, BORDER_CONSTANT, borderValue=padding)` で `motion_meta.output_size` キャンバスへ。はみ出しはclip、空きはpadding。padding_mask は ones を INTER_NEAREST でwarpして反転。**legacy meta + crop_and_pad + bilinear = 旧Inverseとピクセル一致**が必須要件。
4. **crop モード**: 全フレームの有効領域maskの論理AND(共通有効領域)を取り、出力アスペクト比を保った中心配置の最大内接矩形をスケール二分探索で求める(既存 `_refine_no_padding_crop` / keep_fov crop のアプローチを参照実装とする)。求めた `crop_matrix`(平行移動+等方スケールで `output_size` に引き伸ばし)を各 `matrix` に前乗せ(`crop_matrix @ matrix`)して warp。共通有効領域が退化する(内接矩形が極端に小さい: スケール>4x が必要)場合は crop_and_pad にフォールバックし、meta に `"framing_fallback": "crop_and_pad"` を記録(soft-fail方針)。padding_mask は crop 成功時は全ゼロ。
5. **expand モード**: 全フレームのwarp後コンテンツbboxのunionをキャンバスにし、translationを各matrixに前乗せしてwarpする。コンテンツはclipしないが、unionキャンバス内で各フレームごとのpaddingは出る。
6. 出力 meta = 入力 meta のコピー + `"motion_apply": {"input_size", "output_size", "framing_mode", "interpolation", "source"}`。

interpolation は `cv2.INTER_LINEAR` / `cv2.INTER_CUBIC`。maskは常に `INTER_NEAREST`。

### 旧 Inverse ノードの扱い

`nodes/video_stabilizer_inverse.py` は残すが薄いwrapper化:
- `define_schema()` に `is_deprecated=True` を追加。description に「Deprecated: use Video Stabilizer Motion Apply」を追記。入出力は現状のまま一切変更しない。
- `execute()` は `resolve_motion_meta` + `apply_motion(framing="crop_and_pad", interpolation="bilinear")` を呼ぶ実装に置換。ただし出力metaは既存の `inverse_stabilization` ブロック([stabilizer_utils.py:990-998](nodes/stabilizer_utils.py#L990))を維持すること(`scripts/check_inverse_stabilization.py` が通ること)。
- `_apply_inverse_stabilization` 本体は motion_apply.py への委譲に書き換えるか、そのまま残して wrapper 側だけ共通化するかは実装判断。**挙動が変わらないことが唯一の要件。**

---

## 4. Video Stabilizer Shake Generator(新規ノード)

ファイル: `nodes/video_stabilizer_shake_generator.py` + アルゴリズム `nodes/shake_noise.py`

### スキーマ

入力(この順):
1. `frames_context`: `io.Image.Input`(socket)— **バッチ数・解像度の取得のみに使う。フレームは加工しない。** tooltip/description に明記:
   > The input frames are used only to read frame count and resolution. This node outputs motion metadata only; connect it to Video Stabilizer Motion Apply to move pixels.
2. `frame_rate`: `io.Float.Input(default=16.0, min=1.0, step=0.1)` — Classic/Flow と同名・同デフォルトで統一。`_normalize_video_input` がdict入力から fps を拾えた場合はそちらを優先し、widget値はフォールバック(Classic/Flowの `fps_requested`/`fps_effective` と同じ流儀)。
3. `preset`: `io.Combo.Input([...6種...], default="handheld_subtle")`
4. `strength`: `io.Float.Input(default=1.0, min=0.0, max=3.0, step=0.05)` — 全振幅の一括スケール
5. `speed`: `io.Float.Input(default=1.0, min=0.1, max=3.0, step=0.05)` — 全周波数の一括スケール
6. `detail`: `io.Float.Input(default=0.35, min=0.0, max=1.0, step=0.05, display_mode=slider)` — 高周波帯の寄与
7. `seed`: `io.Int.Input(default=0, min=0, max=0xffffffffffffffff, control_after_generate=io.ControlAfterGenerate.fixed)`

出力: `motion_meta`(JSON, "Motion Meta")のみ。**フレームpassthroughは付けない。**

### 生成アルゴリズム(shake_noise.py)

決定性: `rng = np.random.default_rng(seed)` のみ使用。globalな `np.random.*`・時刻・`random` モジュール禁止。同一 `(frame_count, width, height, fps, preset, strength, speed, detail, seed)` → 完全同一出力。

チャンネル: `tx, ty, rotation_deg, log_scale` の4本。

帯域合成(smooth value noise):
- 各帯域の基準周波数(Hz): low / mid / high はpresetごとに持つ(下表)。実効周波数 = 基準 × `speed`、**ただし `fps_effective / 2` でクランプ**(エイリアシング防止)。
- 帯域ごとに: 時間軸 `t = frame_index / fps` 上に間隔 `1/freq` でrng制御点(標準正規)を置き、**Catmull-Rom(cubic)補間**で各フレーム値を得る。制御点は端の外挿用に前後2点余分に生成。
- `channel(t) = low(t)*w_low + mid(t)*w_mid + high(t)*w_high*detail`
- 合成後、**チャンネルごとに平均を引く**(クリップ全体の定常オフセット/ドリフト除去)。
- 振幅: `base = min(width, height)` として `tx = channel_tx * tx_amp * base * strength`(ty同様)、`rotation_deg = channel_rot * rot_amp * strength`、`log_scale = channel_scale * zoom_amp * strength`。
- 制御点生成は解像度に依存させない(振幅スケールのみ解像度依存)→ 同seedなら解像度違いでも同じ「動き」になる。
- rngの消費順序を固定(チャンネル順 × 帯域順)し、コメントで「順序変更は互換性破壊」と明記。

### プリセット定数表(初期値。実装後に目視チューニング可、ただしADRに最終値を記録)

| preset | tx/ty amp (×base) | rot amp (deg) | zoom amp (log) | 帯域重み low/mid/high | 帯域周波数 low/mid/high (Hz) |
|---|---|---|---|---|---|
| handheld_subtle | 0.003 / 0.003 | 0.10 | 0.001 | 1.0 / 0.5 / 0.3 | 0.3 / 1.1 / 3.5 |
| handheld | 0.006 / 0.006 | 0.20 | 0.002 | 1.0 / 0.7 / 0.4 | 0.3 / 1.1 / 3.5 |
| handheld_rough | 0.012 / 0.010 | 0.45 | 0.004 | 0.9 / 1.0 / 0.7 | 0.4 / 1.3 / 4.0 |
| tripod_micro | 0.0008 / 0.0008 | 0.03 | 0.0003 | 1.0 / 0.3 / 0.2 | 0.15 / 0.9 / 3.0 |
| telephoto | 0.002 / 0.002 | 0.35 | 0.001 | 1.0 / 0.6 / 0.5 | 0.25 / 1.0 / 3.5 |
| vibration | 0.004 / 0.004 | 0.08 | 0.001 | 0.2 / 0.5 / 1.5 | 0.5 / 2.5 / 7.0 |

設計意図: rotation 主体(translationだけだと平行移動にしか見えない)、zoom は極小(強いとエフェクト感が出る)。デフォルト `handheld_subtle` はAI生成動画に薄く撮影感を足す用途。

### matrix化

画像中心 `(cx, cy) = (w/2, h/2)` 回りの相似変換:
`M = T(cx + tx, cy + ty) @ R(rotation_deg) @ S(exp(log_scale)) @ T(-cx, -cy)`(3x3、絶対ピクセル座標)。
`input_size == output_size == (w, h)`、`source="generated_shake"`、`generator` ブロックに5パラメータ+seedを記録。`build_motion_meta_v2` で出力。

---

## 5. Classic / Flow の変更(最小限)

- `display_name` の括弧除去のみ(node_id・入出力・widget順は不変)。
- `_stabilize_frames` の meta 組み立て箇所(classic 512-549 / flow 581-621 と、空入力・単一フレーム・crop bypass の各return経路)で、`stabilization_warp` を作った直後に `motion_meta` ブロックを追加:
  - `per_frame[i].matrix = applied_matrix[i]`、`input_size = warp.source_size`、`output_size = warp.output_size`、`fps = fps_effective`、`source = "estimated_classic" / "estimated_flow"`。
  - 既存 `stabilization_warp` fallback は inverse restoration 用に `inv(applied_matrix)` を作るが、Classic/Flow が出力する top-level `motion_meta` は inverse ではない。
- 共通化のため、この変換は `motion_meta.py` に `applied_motion_meta_from_stabilization_warp(warp_meta, fps, source)` として実装し、Classic/Flow 両方から呼ぶ。

---

## 6. 登録と移行(__init__.py / node_replacements)

`nodes/node_replacements.py`(新規):

```python
from comfy_api.latest import ComfyAPI, io
api = ComfyAPI()

async def register_node_replacements() -> None:
    await api.node_replacement.register(io.NodeReplace(
        new_node_id="video_stabilizer_motion_apply",
        old_node_id="video_stabilizer_inverse",
        old_widget_ids=["padding_color"],
        input_mapping=[
            {"new_id": "frames", "old_id": "frames"},
            {"new_id": "motion_meta", "old_id": "meta"},
            {"new_id": "padding_color", "old_id": "padding_color"},
            {"new_id": "framing_mode", "set_value": "crop_and_pad"},
            {"new_id": "interpolation", "set_value": "bilinear"},
        ],
        output_mapping=[
            {"new_idx": 0, "old_idx": 0},
            {"new_idx": 1, "old_idx": 1},
            {"new_idx": 2, "old_idx": 2},
        ],
    ))
```

`__init__.py` の `VideoStabilizerSuiteExtension`:
- `get_node_list()` → Classic, Flow, MotionApply, ShakeGenerator, Inverse(deprecated) の5つ。
- `on_load()` を追加して `register_node_replacements()` を await(`comfy_extras/nodes_replacements.py` と同パターン)。

---

## 7. ファイル構成まとめ

```
nodes/
  video_stabilizer_classic.py      # display_name変更 + motion_meta v2追加
  video_stabilizer_flow.py         # 同上
  video_stabilizer_inverse.py      # deprecated wrapper化(挙動不変)
  video_stabilizer_motion_apply.py # 新規ノード
  video_stabilizer_shake_generator.py # 新規ノード
  motion_meta.py                   # 新規: v2 schema / resolve / validate / from_stabilization_warp
  motion_apply.py                  # 新規: apply_motion / crop計算 / padding mask
  shake_noise.py                   # 新規: noise生成 + matrix化
  node_replacements.py             # 新規: NodeReplace登録
  stabilizer_utils.py              # _apply_inverse_stabilization の委譲化(任意)以外は不変
```

## 8. ドキュメント / スクリプト更新

- `docs/requirements/003-motion-meta-and-apply.md` — v2 meta契約(意味論の一文: "motion_meta matrices are applied as-is by Motion Apply, input_to_output, absolute pixel coordinates")、Motion Apply挙動、旧meta互換。
- `docs/requirements/004-shake-generator.md` — パラメータ、プリセット表、決定性要件、Nyquistクランプ。
- `docs/adr/2026-07-02-motion-meta-v2-and-shake-generator.md` — 再編の決定、Inverse deprecation、motion blur/advancedパラメータ(translation_scale等)は将来拡張として却下理由付きで記録。
- `README.md` / `README_ja.md` — 4ノード構成の説明、Inverseはdeprecated節へ。Shake Generatorの「metaしか出さない」注意書き。
- `example_workflows/` — 既存2ファイルは変更不要(flowのnode_id不変)。可能なら Shake Generator → Motion Apply の最小サンプル `ShakeGenerator_MotionApply.json` を追加(実ComfyUIで保存したJSONを使う。手書きJSONは不可)。無理ならREADME記載のみでも可。
- `scripts/validate_repo.py` — py_compile対象に新規5ファイルを追加。新チェック `scripts/check_motion_meta.py` を登録。

## 9. 検証(scripts/check_motion_meta.py として実装、+手動確認)

repoにpytestは無いので、既存の `check_*.py` 形式(plain Python、ComfyUI非依存、exit code)で:

1. **決定性**: 同一入力+seed → `motion_meta` が完全一致(JSON比較)。
2. **seed差分**: seed違い → per_frame matrices が異なる。
3. **フレーム数一致**: N フレームの frames_context → `per_frame` 長 N、frame_count N。
4. **Nyquist**: fps=8, speed=3 でもクラッシュせず帯域が fps/2 にクランプされる。
5. **identity適用**: 全frame identity matrix の motion_meta → 出力≒入力(allclose)、padding_mask 全ゼロ。
6. **旧Inverse等価**: 合成した `stabilization_warp` meta に対し、新 `apply_motion(crop_and_pad, bilinear)` の出力が既存 `_apply_inverse_stabilization` と一致(frames/mask とも allclose)。既存 `scripts/check_inverse_stabilization.py` も無変更で通ること。
7. **crop フォールバック**: 極端な振幅で crop → crop_and_pad フォールバックし `framing_fallback` が記録される。
8. **隠れランダム性禁止**: shake_noise.py / ノード実装に `np.random.seed` / `np.random.rand` 系のglobal API・`random.` ・`time.` が無いことをgrepで確認する静的チェック。

最後に `python3 scripts/validate_repo.py` を実行。可能ならComfyUI実環境でノードロード + 旧 `video_stabilizer_inverse` を含むworkflowを開いて Motion Apply への置換が提示されることを確認。

## 10. 実装順(コミット単位の目安)

1. `motion_meta.py`(v2 schema / resolve / from_stabilization_warp)+ チェックスクリプト骨子
2. `motion_apply.py` + Motion Apply ノード + 旧Inverse等価テスト(検証6を先に通す)
3. Inverse の deprecated wrapper 化
4. Classic/Flow へ motion_meta v2 追加 + display_name変更
5. `shake_noise.py` + Shake Generator ノード + 決定性テスト
6. `node_replacements.py` + `__init__.py` 更新
7. docs / README / validate_repo.py 更新、全チェック実行
