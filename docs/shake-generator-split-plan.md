# Shake Generator 分割: advanced乗数方式の撤回と2ノード化

実装者: Codex。ブランチ `implement-plan-seq10` 上で作業。未リリースのため互換対応(NodeReplace等)は不要。motion_meta v2 の意味論・旧Inverse互換・決定性要件は従来どおり不変。

## 0. 何が失敗だったか(本質)

前回の設計は「styleで簡単に、advancedでプロ向けに」を**1ノード**で成立させようとした。その実現手段として (a) `advanced=True` フラグで隠す、(b) パラメータを乗数(デフォルト1.0)にする、の2つを選んだ。両方とも失敗である。

**本質は、存在しない仕組みを前提に設計したことだ。**

ユーザーがstyle+個別パラメータのUIに期待する挙動は「styleを変えたら、pan や drift の欄に**そのstyleの値が読み込まれる**」である(Resolve/Sapphire等のプロツールはすべてそう動く)。しかしComfyUIのPython-onlyノードには、**あるwidgetの選択に連動して他のwidgetの値を書き換える手段が存在しない**(カスタムJSフロントエンド拡張が必要で、本repoはPython-onlyの方針)。

この制約を直視せず、2つのごまかしで回避した:

- **乗数方式**: 「連動できないなら、値ではなく倍率にすれば連動不要」という理屈。だが乗数はpresetの中身を**見せない・編集させない**。styleを変えても1.0のまま並ぶ欄は、ユーザーには「死んだパラメータ」に見える。プロ向けの操作性を提供した気になっていただけで、実際には何も提供していない。
- **advanced=True**: node 2.0フロントエンド限定の機能。レガシーモードは現在も広く使われており、そこでは意味不明な1.0が9個ベタ並びになる。**フロントエンドの世代によって挙動が変わるUI機構に、UXの成立自体を依存させてはならない。**

教訓(ADRに記録すること):

1. UI設計は「どのフロントエンドでも成立する形」だけで組む。表示制御フラグはあくまで装飾であり、設計の前提にしない。
2. 「presetが他のパラメータを書き換える」UXはPython-onlyでは実装不可能。**preset型とmanual型は同一ノードに共存できない。分けるのが唯一の正解。**
3. 連動できない代替としての乗数は、presetの中身を読む/編集するという要求の代替にならない。

## 1. 対応: 2ノードに分割

### Video Stabilizer Shake Generator(既存ノードを簡素化)

node_id `video_stabilizer_shake_generator` 維持。入力を以下**のみ**にする:

```text
frames_context : io.Image.Input      # 現行どおり
frame_rate     : io.Float.Input(16.0)
style          : io.Combo.Input(["tripod","handheld","walking","action","vibration"], default="handheld")
amount         : io.Float.Input(1.0, 0.0-3.0, step=0.05)
speed          : io.Float.Input(1.0, 0.1-3.0, step=0.05)
seed           : io.Int.Input(control_after_generate=fixed)
```

pan_amount 等の乗数9個は**全削除**。出力は `motion_meta` のみ(不変)。

### Video Stabilizer Shake Generator Manual(新規ノード)

node_id `video_stabilizer_shake_generator_manual`、display_name `Video Stabilizer Shake Generator Manual`、カテゴリ同じ。styleを持たず、**styleの定数表そのものを絶対値で公開**する。乗数は使わない。

```text
frames_context : io.Image.Input
frame_rate     : io.Float.Input(16.0)
pan            : io.Float.Input(0.60, 0.0-5.0, step=0.01)   # 振幅(度)
tilt           : io.Float.Input(0.50, 0.0-5.0, step=0.01)   # 振幅(度)
roll           : io.Float.Input(0.50, 0.0-5.0, step=0.01)   # 振幅(度)
zoom           : io.Float.Input(0.003, 0.0-0.05, step=0.001) # log-scale振幅
drift_freq     : io.Float.Input(0.35, 0.05-2.0, step=0.05)  # 低周波の構え揺れ(Hz)
tremor         : io.Float.Input(0.35, 0.0-2.0, step=0.05)   # 高周波の震え(pan/tilt/roll振幅に対する比率)
tremor_freq    : io.Float.Input(5.0, 1.0-15.0, step=0.5)    # (Hz)
jitter_rate    : io.Float.Input(0.0, 0.0-3.0, step=0.1)     # 突発的な跳ね(回/秒)
step           : io.Float.Input(0.0, 0.0-2.0, step=0.05)    # 歩行バウンス振幅(度)。>0でf_step=1.9Hz×speedのリズムが乗る
randomness     : io.Float.Input(0.3, 0.0-1.0, step=0.05)
virtual_fov    : io.Float.Input(60.0, 10.0-120.0, step=1.0)
amount         : io.Float.Input(1.0, 0.0-3.0, step=0.05)
speed          : io.Float.Input(1.0, 0.1-3.0, step=0.05)
seed           : io.Int.Input(control_after_generate=fixed)
```

デフォルト値は **handheld styleの定数と完全に一致**させる(下記テスト参照)。

### 内部共有

両ノードとも同一の内部関数を呼ぶ。shake_noise.py に:

```python
@dataclass(frozen=True)
class ShakeRecipe:  # 現行のShakeStyle定数表と同じフィールド構成
    pan, tilt, roll, zoom, drift_freq, tremor, tremor_freq, jitter_rate, step, randomness, virtual_fov

STYLES: Dict[str, ShakeRecipe]  # tripod/handheld/walking/action/vibration の定数表(現行値を維持)

generate_shake_motion_meta(recipe: ShakeRecipe, frame_count, width, height, fps, amount, speed, seed) -> dict
```

簡易版は `STYLES[style]` を、Manual版はwidget値から組んだ `ShakeRecipe` を渡すだけ。生成アルゴリズム本体(角度モデル、4系統合成、先頭identity化、Nyquistクランプ、rng消費順)は変更しない。

## 2. metaに解決済み値を書く(2ノードの接着剤)

`generator` ブロックを以下に統一する:

```json
"generator": {
  "node": "shake_generator" | "shake_generator_manual",
  "style": "walking" | "manual",
  "amount": 1.0,
  "speed": 1.0,
  "seed": 0,
  "recipe": {
    "pan": 0.70, "tilt": 0.90, "roll": 0.70, "zoom": 0.004,
    "drift_freq": 0.30, "tremor": 0.30, "tremor_freq": 5.0,
    "jitter_rate": 0.0, "step": 0.90,
    "randomness": 0.3, "virtual_fov": 60.0
  }
}
```

- `recipe` は amount/speed 適用**前**の絶対値(=Manualノードのwidgetにそのまま打ち込める値)。
- これにより「簡易版でstyleを試す → metaのrecipeを見る → Manualに転記して微調整」という移行パスが成立し、「presetの中身が見えない」問題も解決する。
- READMEにこのワークフローを1段落で記載すること。

## 3. advanced=True の全廃

- repo内の `advanced=True` をすべて削除する(Shake Generator の乗数はノードごと消えるが、**Motion Apply の `motion_blur_samples` も通常widgetに戻す**)。
- 方針として「レガシーフロントエンドで挙動・見え方が変わるUI機構には依存しない」をADRに明記。

## 4. 登録

`__init__.py` の `get_node_list()` に `VideoStabilizerShakeGeneratorManual` を追加(計6ノード)。NodeReplaceは不要(未リリース)。

## 5. テスト(scripts/check_motion_meta.py を更新)

1. **デフォルト等価**: 簡易版 `style="handheld"` と Manual版デフォルト値が、同一 frames/fps/amount/speed/seed で**完全に同一の motion_meta** を出す。
2. **recipeラウンドトリップ**: 簡易版の任意style(全5種)で出力した `generator.recipe` をManual版の入力に与えると、同一seedで per_frame matrices が完全一致する。
3. 既存項目の維持: 決定性 / seed差分 / フレーム数一致 / 先頭identity / style判別性 / Nyquist / blur=0等価 / 隠れランダム性禁止。乗数削除に伴い `roll_amount=0` テストは Manual版 `roll=0.0` に置き換え。
4. `python3 scripts/validate_repo.py` 通過(py_compile対象に新ファイルを追加)。

## 6. ドキュメント

- `docs/adr/2026-07-02-shake-generator-split.md` — §0の失敗の本質と教訓をそのまま記録する(乗数方式・advanced=Trueを撤回した理由、preset型とmanual型を分離した理由)。
- `docs/requirements/004-*.md` — 2ノード構成・Manualのパラメータ定義・generator.recipe契約に更新。
- README / README_ja — ノード表を6ノードに、簡易→Manual転記ワークフローを追記。
