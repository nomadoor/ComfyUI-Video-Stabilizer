# Shake Generator 再設計 + Motion Apply モーションブラー 実装指示

実装者: Codex。ブランチ `implement-plan-seq10` 上で作業する。設計は確定済み。矛盾を見つけた場合のみ質問すること。

## 前提(重要)

- 現在の Shake Generator / Motion Apply は **未マージ・未リリース**(`implement-plan-seq10` は origin/main より先行、1.2.0 は未公開)。したがって **両ノードのスキーマ・パラメータ・meta generator ブロックは互換性を気にせず変更してよい**。NodeReplace の追加は不要。
- 旧 Inverse 互換・`stabilization_warp` 互換・motion_meta v2 の意味論(`input_to_output`、絶対ピクセル座標3x3、Motion Apply がそのまま適用)は **一切変更しない**。
- 決定性要件は従来どおり: `np.random.default_rng(seed)` のみ。global `np.random.*` / `random` / 時刻依存は禁止。rng消費順序は固定しコメントで明記。

## 現行実装の問題(修正必須の根本原因)

1. **平均減算バグ**: `_channel`(nodes/shake_noise.py)がクリップ全体の平均を引いている。短いクリップでは低周波帯の制御点が2〜3個しかなく、平均減算でdrift成分がほぼ消える。→ **平均ではなく「フレーム0の値」を各チャンネルから引く**(先頭フレームがidentityになる。driftは残る)。
2. **振幅が5〜10倍小さい**: 576pで handheld が tx≈3.5px / rot 0.2° では知覚できない。下表の新基準に置き換える。
3. **4チャンネルが構造的に同一**(同じ3帯域の重み違い)なのでプリセットの性格差が出ない。→ 下記の新内部モデルに置き換える。

## 1. 新しい内部モデル: 角度ベース

チャンネルを画像空間(tx/ty)ではなく **カメラ角度空間** で生成する:

```text
pan_deg   (ヨー: 画面上ではほぼ横移動として現れる)
tilt_deg  (ピッチ: ほぼ縦移動)
roll_deg  (画面内回転)
zoom_log  (微ズーム)
```

画像空間への投影(similarity行列のまま。perspective化はしない):

```text
f  = 0.5 * min(width, height) / tan(radians(virtual_fov) / 2)
tx = f * tan(radians(pan_deg))
ty = f * tan(radians(tilt_deg))
rotation = roll_deg
scale    = exp(zoom_log)
M = T(cx+tx, cy+ty) @ R(roll) @ S(scale) @ T(-cx, -cy)   # 既存 _matrix と同形
```

`virtual_fov` はadvanced入力(デフォルト60.0、範囲10〜120)。狭くすると同じ角度揺れが画面上で拡大される=望遠の揺れが創発する(telephotoプリセットは廃止)。

### 揺れの構成要素(チャンネルごとに合成)

従来の「汎用3帯域」をやめ、性格の異なる4系統に分ける:

1. **drift** — 低周波の構え揺れ。value noise(Catmull-Rom補間、既存 `_smooth_value_noise` を流用)、基準周波数 0.2〜0.5 Hz。
2. **tremor** — 生理的な細かい震え。value noise、基準周波数 4〜8 Hz(生理的振戦は8-12Hzだが既存のNyquistクランプ `fps/2` に任せる)。
3. **jitter** — 突発的な小さい跳ね。決定的Poissonイベント: rngでイベント時刻を生成(レートλ回/秒)、各イベントはランダム方向のインパルス×指数減衰エンベロープ(τ≈0.1s)を pan/tilt/roll に加算。
4. **step**(walkingのみ) — 足のリズム。`f_step = 1.9 Hz × pace` の正弦波ベース: tilt に f_step のバウンス、pan/roll に f_step/2 の左右スウェイ。位相と振幅を遅いvalue noiseで揺らして機械感を消す。

合成(チャンネルごと):

```text
channel(t) = drift(t)*w_drift + tremor(t)*w_tremor + jitter(t)*w_jitter + step(t)*w_step
channel(t) -= channel(0)   # 先頭フレームidentity化
```

**randomness**(advanced、0〜1、デフォルト0.3): drift/tremor の振幅を別の遅いnoise(≈0.2Hz)で時間変調する深さ。0で完全に一定振幅、上げると揺れの「ムラ」が増える。

Nyquistクランプ・rng消費順固定・解像度非依存(振幅スケールのみ解像度依存)は現行実装の方針を維持。

## 2. スタイル(preset廃止 → style)

強さの序列ではなく撮影シーンで分ける。**styleは配合比(質)、amountは音量ノブ(量)**という役割分担を崩さないこと。

`style: io.Combo.Input(["tripod", "handheld", "walking", "action", "vibration"], default="handheld")`

定数表(amount=1.0 時の各系統の振幅。pan/tilt/roll は度、zoom は log):

| style | pan/tilt amp | roll amp | zoom amp | drift freq(Hz) | tremor amp比 | tremor freq(Hz) | jitter rate(/s) | step |
|---|---|---|---|---|---|---|---|---|
| tripod | 0.03 / 0.03 | 0.02 | 0.0002 | 0.2 | 0.15 | 4.0 | 0 | なし |
| handheld | 0.60 / 0.50 | 0.50 | 0.003 | 0.35 | 0.35 | 5.0 | 0 | なし |
| walking | 0.70 / 0.90 | 0.70 | 0.004 | 0.30 | 0.30 | 5.0 | 0 | あり(tilt主体) |
| action | 1.20 / 1.00 | 1.00 | 0.006 | 0.50 | 0.80 | 6.0 | 0.5 | なし |
| vibration | 0.15 / 0.15 | 0.10 | 0.001 | driftなし | 1.0(主体) | 8.0 | 0 | なし |

- 表の「tremor amp比」は pan/tilt/roll amp に対する tremor 系統の比率。
- vibration は drift・jitter・step を持たない高周波専用。
- walking の step 振幅: tilt バウンス ≈ tilt amp と同等、pan/roll スウェイはその半分程度。
- 数値は初期値。実装後に目視チューニングしてよいが、**最終値を requirements 004 に記録**すること。目安: handheld で画面上の移動が min(w,h) の 0.8〜1.2% 程度になること(fov 60° なら pan ±0.6° ≈ 0.9%)。

## 3. 新スキーマ(Shake Generator)

Basic(この順):

```text
frames_context : io.Image.Input          # 現行どおり。tooltipも維持
frame_rate     : io.Float.Input(16.0)    # 現行どおり
style          : io.Combo.Input(上記5種, default="handheld")
amount         : io.Float.Input(default=1.0, min=0.0, max=3.0, step=0.05)  # 全振幅の一括倍率
pace           : io.Float.Input(default=1.0, min=0.1, max=3.0, step=0.05)  # 全周波数の一括倍率
seed           : io.Int.Input(control_after_generate=fixed)  # 現行どおり
```

Advanced(すべて `advanced=True` フラグを付ける。V3 の Input が対応済み: comfy_api/latest/_io.py の `advanced` 引数):

```text
pan_amount    : Float(1.0, 0.0-3.0)   # styleに対する乗数
tilt_amount   : Float(1.0, 0.0-3.0)
roll_amount   : Float(1.0, 0.0-3.0)
zoom_amount   : Float(1.0, 0.0-3.0)
drift_amount  : Float(1.0, 0.0-3.0)   # 系統別乗数
tremor_amount : Float(1.0, 0.0-3.0)
jitter_amount : Float(1.0, 0.0-3.0)
randomness    : Float(0.3, 0.0-1.0)
virtual_fov   : Float(60.0, 10.0-120.0)
```

- advancedはすべて **乗数(デフォルト1.0)**。絶対値にしないこと(styleを変えてもadvancedの意味が保たれるように)。
- `preset` / `strength` / `speed` / `detail` は廃止。`strength→amount`, `speed→pace` は単なる改名ではなくスキーマ再構築として扱ってよい(未リリースのため)。
- `motion_blur` / `edge_mode` 系は **このノードには置かない**(Applyの責務)。
- 出力は現行どおり `motion_meta` のみ。

meta の `generator` ブロックは新パラメータを反映:

```json
"generator": {
  "style": "handheld", "amount": 1.0, "pace": 1.0, "seed": 0,
  "pan_amount": 1.0, "tilt_amount": 1.0, "roll_amount": 1.0, "zoom_amount": 1.0,
  "drift_amount": 1.0, "tremor_amount": 1.0, "jitter_amount": 1.0,
  "randomness": 0.3, "virtual_fov": 60.0
}
```

motion_meta の `version` / `matrix_convention` / `per_frame` 構造は不変。

## 4. Motion Apply: モーションブラー追加

入力の末尾に追加:

```text
motion_blur         : io.Float.Input(default=0.0, min=0.0, max=1.0, step=0.05)
                      # シャッター開角割合。0.5 ≈ 180°シャッター相当。0で無効
motion_blur_samples : io.Int.Input(default=9, min=3, max=33, advanced=True)
```

実装(nodes/motion_apply.py):

- `motion_blur == 0.0` → **既存パスを完全にバイパス**(現行とビット一致。テスト必須)。
- `motion_blur > 0.0` → フレーム i について、matrix_i と matrix_{i+1} を **行列の線形補間** で `t ∈ [0, motion_blur]` の N サンプルに展開し、各サンプルで warp して平均する。最終フレームは matrix_{i-1}→matrix_i の差分を外挿(フレームが1枚なら blur なし)。
  - 行列lerpの回転歪みはフレームあたり回転が数度以下(shakeの実域)では無視できる。パラメータ分解lerpはしない(perspective行列を持つlegacy metaで破綻するため)。
- padding_mask: 各サンプルの coverage(onesのwarp)を平均し、`mask = 1 - mean_coverage` のソフトマスクにする。微小値のゼロ化は既存慣例に従う。
- crop モードとの併用: crop_matrix 合成後の行列同士を補間すれば自然に整合する。
- meta の `motion_apply` ブロックに `motion_blur` / `motion_blur_samples` を追記。
- Apply側に乱数は一切入れない(完全決定的)。

旧 Inverse(deprecated wrapper)は blur なしの固定呼び出しのまま。挙動不変。

## 5. テスト更新(scripts/check_motion_meta.py + 必要なら新チェック)

既存チェックを新スキーマに追随させた上で、以下を追加:

1. **先頭identity**: どのstyle/seedでも per_frame[0].matrix ≈ identity。
2. **style判別性**: 同一seedで `vibration` は高周波帯エネルギーが `handheld` より大きい、`walking` は tilt 信号に f_step 近傍のスペクトルピークがある、`tripod` の総振幅は `handheld` より1桁小さい(FFTか差分統計で機械的に検証)。
3. **決定性/seed差分/フレーム数一致/Nyquist**: 既存項目を新パラメータで維持。
4. **advanced乗数**: `roll_amount=0` で roll 成分がゼロになる(行列の回転成分で確認)。
5. **blur=0等価**: `motion_blur=0` の出力が現行実装と一致(allclose)。
6. **blur決定性**: `motion_blur=0.5` で2回実行して完全一致。
7. **旧Inverse等価・隠れランダム性禁止**: 既存チェック維持。`check_inverse_stabilization.py` は無変更で通ること。

最後に `python3 scripts/validate_repo.py` を通す。

## 6. ドキュメント

- `docs/requirements/004-*.md` を新設計で書き直し(style表・最終定数・決定性・先頭identity・角度モデルとvirtual_fovの定義)。
- ADR 追加: `docs/adr/2026-07-02-shake-generator-redesign.md` — 旧 preset/strength/speed/detail を廃止した理由(強さの序列presetはamountと冗長でUX混乱を招く/シーン別styleに変更)、角度モデル採用理由、motion blurをApply側に置いた理由、平均減算→先頭基準化の理由。
- README / README_ja のパラメータ表を更新。Shake Generator の説明に「styleは揺れの種類、amountは強さ」を明記。

## 実装順

1. shake_noise.py 全面書き換え(角度モデル+4系統合成+先頭identity化)+ 定数表
2. Shake Generator ノードの新スキーマ(basic 6 + advanced 9)
3. Motion Apply の motion_blur(blur=0 バイパスを最初にテスト)
4. チェックスクリプト更新・追加、validate_repo 通過
5. docs / README 更新
