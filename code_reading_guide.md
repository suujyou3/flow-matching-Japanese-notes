# 教材コードの読み方

このファイルは、`src/` 配下のPyTorchコードを読むための案内である。本文の数式を先に読んだあと、この順番でコードを見ると対応を追いやすい。

## 1. 最初に読む流れ

まずは次の6ファイルを見る。

1. `src/fm_minimal/data.py`
2. `src/fm_minimal/time_samplers.py`
3. `src/fm_minimal/paths.py`
4. `src/fm_minimal/models.py`
5. `src/fm_minimal/losses.py`
6. `src/fm_minimal/solvers.py`

この6つで、データ作成から学習・生成までの最小単位がそろう。`couplings.py` は独立ペア以外を試す段階で追加して読む。

```text
学習: data -> coupling -> time sampler + path -> model -> loss -> optimizer
生成: 学習済みmodel -> solver
```

`time sampler` は学習時刻 $t$ を選ぶ。`path` は $x_t$ と教師速度 $u_t$ を作る。`model` は $(t,x_t)$ から速度を予測し、`loss` が予測速度と教師速度を比較する。`optimizer` はlossから得たgradientでmodelを更新する。生成時の`solver`は、学習済みmodelを繰り返し呼んでsource点をtarget側へ動かす。

## 2. ファイル別の役割

| ファイル | 主な役割 | 本文で対応する内容 |
|---|---|---|
| `src/fm_minimal/data.py` | toy source/target分布を作る | 第1部、第10部 |
| `src/fm_minimal/paths.py` | Linear pathとGaussian pathを定義する | 第1部、第5部 |
| `src/fm_minimal/time_samplers.py` | 学習時刻を一様・中央重視・両端重視で標本化する | 第1部、第7部 |
| `src/fm_minimal/losses.py` | CFM lossを計算する | 第1部、第2部 |
| `src/fm_minimal/models.py` | 2D toy用の時刻条件付きMLPを定義する | 第3部、第4部 |
| `src/fm_minimal/solvers.py` | Euler/Heun/RK4でODEを解く | 第1部、第8部 |
| `src/fm_minimal/couplings.py` | independent/mini-batch couplingを作る | 第6部 |
| `src/fm_minimal/diffusion_basics.py` | Diffusionのnoising、epsilon loss、DDIM型生成を最小実装する | 第4部、第5部 |
| `src/fm_minimal/reflow.py` | Reflow用ペアと軌道直線性を扱う | 第7部 |
| `src/fm_minimal/evaluation.py` | biased MMD²、mode coverage、endpoint errorを計算する | 第10部 |
| `src/fm_minimal/image_models.py` | 画像用U-Net、patch操作、最小DiT速度場を示す | 第9部 |
| `src/fm_minimal/__init__.py` | 各部品を `from fm_minimal import ...` で読み込めるよう公開する | 全実行スクリプト |
| `src/train_minimal_2d.py` | toy CFMを学習する実行スクリプト | 第3部 |
| `src/train_diffusion_toy.py` | toy Diffusion epsilon predictionを学習する実行スクリプト | 第4部 |
| `src/sample_diffusion_toy.py` | 学習済みDiffusionから点群と逆向き軌道を生成する | 第4部 |
| `src/evaluate_diffusion_toy.py` | Diffusion生成をNFE別に数値評価する | 第4部、第10部 |
| `src/check_image_model_shapes.py` | 画像U-Net/DiTの入出力shapeとgradientを確認する | 第9部 |
| `src/train_rectified_flow_2d.py` | toy Rectified Flowを学習し、straightness ratioを保存する実行スクリプト | 第7部 |
| `src/train_reflow_2d.py` | 学習済みflowでペアを作り直し、次のRectified Flowを学習する | 第7部 |
| `src/plot_minimal_2d.py` | source/target/generated/trajectoryを描く | 第1部、第3部 |
| `src/evaluate_minimal_2d.py` | checkpointをsolver/NFE別に評価し、CSVと比較図を保存する | 第10部 |

## 3. 数式とコードの対応

### path

`LinearPath.sample` は

```math
x_t = (1-t)x_0 + tx_1
```

に対応する。`LinearPath.velocity` は

```math
u_t = x_1 - x_0
```

に対応する。

`TrigGaussianPath` では

```math
x_t = \alpha(t)x_1 + \sigma(t)x_0
```

を使う。pathを変えると、主に `sample` と `velocity` が変わる。lossやtraining loopの外形はあまり変わらない。

どちらのpathも、time samplerが返す `t: [B,1]` を入力データの次元数に合わせて `[B,1,...,1]` へ内部で拡張する。このため2D点 `[B,D]` だけでなく画像 `[B,C,H,W]` にも同じpathを適用できる。独自pathも `sample` と `velocity` を持つ `ConditionalPath` の契約に合わせる。

### loss

`conditional_flow_matching_loss` は

```math
\mathbb{E}_{t,x_0,x_1}
\left[
\|v_\theta(t,x_t)-u_t\|^2
\right]
```

をミニバッチで近似する。重要なのは、モデルに `x0` と `x1` を渡さないことである。生成時に使える速度場は、$t$ と $x_t$ だけから速度を返す必要がある。

実装は誤差tensorを `flatten(1)` で各サンプルのベクトルへ直してから二乗和を取る。したがって2D点と画像で同じloss関数を使える。画像で要素平均へ変更する場合はlossの尺度が変わるため、学習率と比較条件も記録する。

### time sampler

`get_time_sampler("uniform")` は全区間を一様に選ぶ。`center` は中央、`endpoint` は両端を多く選ぶ。lossへ関数として渡すため、pathやmodelを変えずに時刻分布だけを比較できる。

```python
time_sampler = get_time_sampler("uniform")
loss = conditional_flow_matching_loss(
    model, path, x0, x1, time_sampler=time_sampler
)
```

### model

`models.py` の `MLPVelocity` は、時刻 $t$ をsin/cosの時刻埋め込みへ変換し、状態 $x_t$ と結合してMLPへ渡す。

```text
t -> SinusoidalTimeEmbedding --+
                                +-> MLP -> xと同じshapeの出力
x_t ---------------------------+
```

`hidden_dim` は各隠れ層の幅、`depth` は隠れ層数、`time_dim` は時刻埋め込みの次元、`activation` は各隠れ層の非線形関数を決める。`depth` は1以上、`time_dim` は正の偶数でなければならず、教材コードは不正値を早期にエラーにする。出力の意味はモデルクラス名だけでは決まらない。同じMLPをCFM lossで学習すればvelocity、Diffusionのepsilon lossで学習すればnoise予測になる。

### solver

`euler_solve` は

```math
x_{t+\Delta t} = x_t + \Delta t\,v_\theta(t,x_t)
```

を繰り返す。`heun_solve` は1 stepで2回、`rk4_solve` は1 stepで4回モデルを呼ぶ。したがって、solver比較ではstep数だけでなくNFEも見る。

## 4. 実行スクリプトの役割

### `train_minimal_2d.py`

2D toy分布でモデルを学習する。学習ループの1 stepは次の順番である。

```text
x0をsampleする
x1をsampleする
tの前に必要ならcouplingでペアを並べ替える
tをsampleする
x_tとu_tを作る
model(t, x_t)を計算する
MSE lossで更新する
```

保存される `minimal_2d.pt` には、モデル重み、モデル構成、確認用生成サンプル、path、coupling、time sampler、seedが入る。可視化・評価スクリプトは `model_config` を読み、学習時と同じ幅・深さ・活性化関数のMLPを復元する。

### `plot_minimal_2d.py`

checkpointを読み、4つのパネルを作る。

```text
source
target
generated
trajectory
```

`trajectory` は、学習済み速度場で点がどのように動くかを見るためのもの。lossだけでは分からない曲がり方や発散を確認できる。

`--seed` を固定すると、source、target、軌道用noiseを同じ条件で再生成できる。複数checkpointの図を比較するときは同じseedを使う。

### `train_diffusion_toy.py`

2D toy分布をclean dataとして、Diffusionのepsilon predictionを学習する。Flow Matchingとの違いを読むための比較用コードである。

```text
x_dataをsampleする
tとnoiseをsampleする
x_t = alpha(t) x_data + sigma(t) noise を作る
model(t, x_t)でnoiseを予測する
epsilon MSEで更新する
```

学習後は `sample_diffusion_toy.py` が決定論的なDDIM型更新でnoiseから点群を生成し、`evaluate_diffusion_toy.py` がbiased MMD²、mode coverage、endpoint errorをNFE別に計算する。

### `train_reflow_2d.py`

base checkpointを凍結し、そのflowでsourceを輸送してReflowペアを作る。新しいモデルはそのペアの直線速度を学ぶ。`pair_steps` はペア生成用ODEの精度、`steps` は新モデルの学習回数であり、別の引数である。

### `check_image_model_shapes.py`

ランダム画像をTiny U-NetとTiny DiTへ通し、出力shapeが入力画像と一致すること、仮lossから有限gradientが得られることを検査する。画像品質ではなく、実画像学習前のインターフェース確認である。

### `train_rectified_flow_2d.py`

1-Rectified Flowを学習する。lossは直線補間の速度回帰であり、学習後に生成サンプルとstraightness ratioを保存する。

```text
x0をsourceからsampleする
x1をtargetからsampleする
x_t = (1-t)x0 + t x1 を作る
u_t = x1 - x0 を教師にする
学習後にODE軌道のstraightness ratioを見る
```

### `evaluate_minimal_2d.py`

checkpointを読み、solverとstep数を変えて数値評価する。全条件で同じsource noiseとtarget標本を使うため、乱数標本の違いではなくsolverとNFEの差を比較しやすい。

```text
solver,steps,nfe,mmd,coverage,endpoint_error,mode_counts,...
```

CSV列名は `mmd` だが、実装が計算するのはRBF kernelを使ったbiased MMD²推定値である。小さいほど二つの点群が近いが、値は `bandwidth` に依存する。`coverage` は8個のmodeのうち指定半径内に生成点が1個以上入った割合、`mode_counts` は各mode付近の点数である。`endpoint_error` は各生成点から最も近いtarget点までの平均距離であり、target側のmodeをすべて覆うことは保証しない。三つを併用する。

結果は `_outputs/minimal_2d_evaluation.csv` に保存され、3指標のNFE別比較図は `figures/minimal_2d_solver_comparison.png` に保存される。

## 5. 実験を実行する

以下のコマンドは、カレントディレクトリの直下に `src`、`_outputs`、`figures` がある状態で実行する。リポジトリを取得した場合は、`code_reading_guide.md` と `src` が見えるディレクトリへ移動してから実行する。Pythonの起動コマンドは環境によって `python`、`python3`、`py` のいずれかになるため、本書では `python` と表記する。例はPowerShell形式である。Bashやzshでは、パス区切りの `\` を `/` に、行末のバッククォートを `\` に置き換えるか、コマンドを1行で入力する。初回は短い設定で処理全体が動くことを確認し、その後で学習stepやサンプル数を増やす。

### 5.1 最小CFMを学習する

まず、一様なtime samplerを使うbaselineを学習する。

```powershell
python .\src\train_minimal_2d.py `
  --steps 2000 `
  --batch 512 `
  --lr 0.001 `
  --device cpu `
  --seed 0 `
  --time-sampler uniform `
  --path linear `
  --coupling independent `
  --hidden-dim 128 `
  --depth 3 `
  --time-dim 32 `
  --activation silu `
  --out .\_outputs\baseline
```

| 引数 | 既定値 | 意味 |
|---|---:|---|
| `--steps` | `2000` | optimizerを更新する学習step数。ここではODEの分割数ではない |
| `--batch` | `512` | 1 stepで使うsource/targetペア数 |
| `--lr` | `0.001` | AdamWの学習率 |
| `--device` | `cpu` | `cpu` または利用可能な `cuda` device |
| `--seed` | `0` | 学習データ、初期重み、時刻標本化に使う乱数seed |
| `--time-sampler` | `uniform` | `uniform`, `center`, `endpoint`から学習時刻の分布を選ぶ |
| `--path` | `linear` | `linear` または三角関数係数の `trig` |
| `--coupling` | `independent` | 独立ペアまたは教材用の `greedy` mini-batch coupling |
| `--hidden-dim` | `128` | MLP各隠れ層の幅 |
| `--depth` | `3` | MLPの隠れ層数 |
| `--time-dim` | `32` | 時刻埋め込みの次元。偶数を指定する |
| `--activation` | `silu` | `models.py` の隠れ層で使う `silu`, `relu`, `tanh` |
| `--out` | `_outputs` | checkpointを保存するディレクトリ。この中に `minimal_2d.pt` が作られる |

動作確認だけなら、次のように小さくする。

```powershell
python .\src\train_minimal_2d.py `
  --steps 20 --batch 32 `
  --out .\_outputs\smoke_test
```

20 stepでは生成品質を判断しない。これはimport、shape、backward、checkpoint保存までが動くかを確認するsmoke testである。

### 5.2 checkpointを可視化する

学習済みcheckpointから、source、target、生成点、粒子軌道を描く。

```powershell
python .\src\plot_minimal_2d.py `
  --checkpoint .\_outputs\baseline\minimal_2d.pt `
  --out .\figures\baseline_result.png `
  --device cpu `
  --samples 2048 `
  --trajectory-samples 64 `
  --steps 64 `
  --seed 0
```

| 引数 | 既定値 | 意味 |
|---|---:|---|
| `--checkpoint` | `_outputs/minimal_2d.pt` | 読み込む学習済み重み |
| `--out` | `figures/minimal_2d_result.png` | 保存する画像ファイル |
| `--device` | `cpu` | 推論を実行するdevice |
| `--samples` | `2048` | source、target、generatedへ描く点の数 |
| `--trajectory-samples` | `64` | 軌道を線として描く粒子数。多すぎると図が読みにくくなる |
| `--steps` | `64` | Euler法でODEを分割するstep数。学習step数とは別 |
| `--seed` | `0` | source、target、軌道用noiseの乱数条件 |

### 5.3 solverとNFEを数値評価する

同じsource noiseとtarget標本を使い、solverとstep数を変えて比較する。

```powershell
python .\src\evaluate_minimal_2d.py `
  --checkpoint .\_outputs\baseline\minimal_2d.pt `
  --device cpu `
  --samples 2048 `
  --target-samples 2048 `
  --solvers euler heun rk4 `
  --steps 8 16 32 64 `
  --seed 0 `
  --csv-out .\_outputs\baseline\evaluation.csv `
  --plot-out .\figures\baseline_solver_comparison.png
```

| 引数 | 既定値 | 意味 |
|---|---:|---|
| `--checkpoint` | `_outputs/minimal_2d.pt` | 評価するcheckpoint |
| `--device` | `cpu` | 評価を実行するdevice |
| `--samples` | `2048` | 各条件で生成するsource標本数 |
| `--target-samples` | `2048` | 指標計算に使うtarget標本数 |
| `--solvers` | `euler heun` | 比較するsolver。`euler`, `heun`, `rk4`を指定できる |
| `--steps` | `8 16 32 64` | 各solverで試すODE step数のリスト |
| `--nfe-budgets` | 指定なし | solverごとに同じNFE予算となるstep数を自動計算する。`--steps`とは同時指定しない |
| `--bandwidth` | `0.5` | MMDのGaussian kernel幅 |
| `--coverage-radius` | `0.35` | 生成点がmodeを覆ったと判定する半径 |
| `--seed` | `0` | sourceとtargetの乱数条件。同じ比較では固定する |
| `--csv-out` | `_outputs/minimal_2d_evaluation.csv` | 数値結果を保存するCSV |
| `--plot-out` | `figures/minimal_2d_solver_comparison.png` | NFE別比較図の出力先 |

`--steps 16`でもsolverごとのNFEは同じではない。モデル呼び出しはEulerが1 step当たり1回、Heunが2回、RK4が4回なので、CSVの `nfe` 列で計算量を比較する。

同じNFE予算でsolverを比較する場合は、`steps_from_nfe_budget`を使う次の指定に切り替える。

```powershell
python .\src\evaluate_minimal_2d.py `
  --checkpoint .\_outputs\baseline\minimal_2d.pt `
  --solvers euler heun rk4 `
  --nfe-budgets 16 32 64 `
  --seed 0 `
  --csv-out .\_outputs\baseline\equal_nfe.csv `
  --plot-out .\figures\baseline_equal_nfe.png
```

たとえばNFE予算16なら、Eulerは16 step、Heunは8 step、RK4は4 stepになる。割り切れない予算では、予算を超えない最大の整数step数を使う。

### 5.4 time samplerだけを変えて比較する

time samplerの効果を見るときは、出力ディレクトリを分け、その他の引数を固定する。

```powershell
python .\src\train_minimal_2d.py `
  --steps 2000 --batch 512 --lr 0.001 --time-sampler uniform `
  --out .\_outputs\time_uniform

python .\src\train_minimal_2d.py `
  --steps 2000 --batch 512 --lr 0.001 --time-sampler center `
  --out .\_outputs\time_center

python .\src\train_minimal_2d.py `
  --steps 2000 --batch 512 --lr 0.001 --time-sampler endpoint `
  --out .\_outputs\time_endpoint
```

三つのcheckpointを5.2節と5.3節のコマンドで個別に可視化・評価する。比較時には同じ `--seed`、`--samples`、solver、NFEを使う。training lossの大小だけではなく、MMD、mode coverage、軌道図を合わせて見る。

### 5.5 pathだけを変えて比較する

第11.3節のpath差し替え実験を行う。time sampler、coupling、model、seedを固定し、`--path` と出力先だけを変える。

```powershell
python .\src\train_minimal_2d.py `
  --steps 2000 --batch 512 --seed 0 `
  --path linear --coupling independent --time-sampler uniform `
  --out .\_outputs\path_linear

python .\src\train_minimal_2d.py `
  --steps 2000 --batch 512 --seed 0 `
  --path trig --coupling independent --time-sampler uniform `
  --out .\_outputs\path_trig
```

`linear` は $x_t=(1-t)x_0+tx_1$、`trig` はdata係数をsin、noise係数をcosで変えるGaussian風pathである。両者は途中状態と教師速度が異なるため、training lossの値を直接比べるだけでなく、5.2節と5.3節の生成評価を行う。

### 5.6 couplingだけを変えて比較する

```powershell
python .\src\train_minimal_2d.py `
  --steps 2000 --batch 512 --seed 0 `
  --path linear --coupling independent `
  --out .\_outputs\coupling_independent

python .\src\train_minimal_2d.py `
  --steps 2000 --batch 512 --seed 0 `
  --path linear --coupling greedy `
  --out .\_outputs\coupling_greedy
```

`independent` は `random_coupling` でtarget側をランダムに並べ替え、sourceとtargetの対応に距離情報を使わない。`greedy` はmini-batch内で距離の近い点を重複なしに対応させる。これはcouplingを差し替える位置を理解する教材用近似であり、厳密なOptimal Transport solverではない。比較では同じbatch sizeを使う。batchが変わるとgreedy couplingが探索できる候補数も変わるためである。

### 5.7 `models.py` のモデル変更とseedのablation

`models.py` のMLP設計は、CLIから幅、深さ、時刻埋め込み、活性化関数を変更できる。まずモデル容量だけを変える。

```powershell
python .\src\train_minimal_2d.py `
  --steps 2000 --batch 512 --seed 0 `
  --hidden-dim 64 --depth 2 --time-dim 32 `
  --activation silu `
  --out .\_outputs\model_small

python .\src\train_minimal_2d.py `
  --steps 2000 --batch 512 --seed 0 `
  --hidden-dim 128 --depth 3 --time-dim 32 `
  --activation silu `
  --out .\_outputs\model_baseline
```

活性化関数だけを変更する例:

```powershell
python .\src\train_minimal_2d.py `
  --steps 2000 --batch 512 --seed 0 `
  --hidden-dim 128 --depth 3 --time-dim 32 --activation relu `
  --out .\_outputs\model_relu

python .\src\train_minimal_2d.py `
  --steps 2000 --batch 512 --seed 0 `
  --hidden-dim 128 --depth 3 --time-dim 32 --activation tanh `
  --out .\_outputs\model_tanh
```

| 変更箇所 | 主な影響 | 固定するもの |
|---|---|---|
| `hidden_dim` | 1層が保持できる特徴量数とパラメータ数 | depth、activation、学習条件 |
| `depth` | 非線形変換を重ねる回数 | width、activation、学習条件 |
| `time_dim` | 時刻依存性を表す埋め込み容量 | MLP容量、path、time sampler |
| `activation` | 勾配の流れ方と表現する関数の滑らかさ | width、depth、seed |

checkpointには `model_config` が保存されるため、5.2節・5.3節は変更後の構造を自動復元する。手で `models.py` に新しい層やskip connectionを追加した場合は、その引数も `model_config` に保存し、読込側で同じコンストラクタを呼べるようにする。

乱数によるばらつきを見る場合は、モデル設定を固定して `--seed 0`, `--seed 1`, `--seed 2` を別々の出力先で実行する。平均だけでなく各seedの評価値も残す。モデル構造とseedを同時に変えると、差の原因を分離できない。

### 5.8 Rectified Flowを学習する

```powershell
python .\src\train_rectified_flow_2d.py `
  --steps 2000 `
  --batch 512 `
  --lr 0.001 `
  --device cpu `
  --seed 0 `
  --time-sampler uniform `
  --hidden-dim 128 `
  --depth 3 `
  --time-dim 32 `
  --activation silu `
  --out .\_outputs\rectified_flow
```

各引数の意味は5.1節と同じである。出力される `rectified_flow_2d.pt` には、モデル重み、モデル構成、生成サンプル、平均straightness ratio、time sampler名が保存される。

### 5.9 Reflowでペアを作り直して再学習する

5.8節で学習したcheckpointをbase modelとして使う。

```powershell
python .\src\train_reflow_2d.py `
  --base-checkpoint .\_outputs\rectified_flow\rectified_flow_2d.pt `
  --steps 2000 `
  --batch 512 `
  --lr 0.001 `
  --device cpu `
  --seed 0 `
  --pair-steps 64 `
  --time-sampler uniform `
  --out .\_outputs\reflow
```

| 引数 | 既定値 | 意味 |
|---|---:|---|
| `--base-checkpoint` | `_outputs/rectified_flow_2d.pt` | Reflowペアを生成する学習済みモデル |
| `--steps` | `2000` | 新しいモデルを更新する学習step数 |
| `--batch` | `512` | 1 stepで生成・学習するReflowペア数 |
| `--lr` | `0.001` | 新しいモデルの学習率 |
| `--device` | `cpu` | ペア生成と学習を行うdevice |
| `--seed` | `0` | source、初期重み、時刻の乱数seed |
| `--pair-steps` | `64` | base modelを積分してReflow終点を作るODE step数 |
| `--time-sampler` | `uniform` | 新しいモデルの学習時刻分布 |
| `--out` | `_outputs/reflow` | `reflow_2d.pt` の保存先ディレクトリ |

`--pair-steps` はReflowペアの品質と作成コストを変える。新モデルの学習step数 `--steps` や、評価時のODE step数とは別である。生成後は5.2節と5.3節の `--checkpoint` に `reflow_2d.pt` を指定し、base modelと同じNFEで比較する。

新モデルの幅、深さ、時刻埋め込み、活性化関数はbase checkpointの `model_config` を引き継ぐ。Reflowとモデル構造変更を同時に行わず、まず同じ構造でcoupling更新の効果だけを測るためである。

### 5.10 比較用Diffusionを学習する

```powershell
python .\src\train_diffusion_toy.py `
  --steps 2000 `
  --batch 512 `
  --lr 0.001 `
  --device cpu `
  --seed 0 `
  --hidden-dim 128 `
  --depth 3 `
  --time-dim 32 `
  --activation silu `
  --out .\_outputs\diffusion
```

| 引数 | 既定値 | 意味 |
|---|---:|---|
| `--steps` | `2000` | epsilon predictionを学習するoptimizer step数 |
| `--batch` | `512` | 1 stepのclean data数 |
| `--lr` | `0.001` | AdamWの学習率 |
| `--device` | `cpu` | 学習device |
| `--seed` | `0` | データ、noise、初期重み、時刻の乱数seed |
| `--hidden-dim` | `128` | MLP隠れ層の幅 |
| `--depth` | `3` | MLP隠れ層数 |
| `--time-dim` | `32` | 時刻埋め込み次元 |
| `--activation` | `silu` | MLPの活性化関数 |
| `--out` | `_outputs` | `diffusion_toy.pt` の保存先ディレクトリ |

このスクリプトは学習とcheckpoint保存を担当する。生成と評価は次の2スクリプトへ分ける。

### 5.11 Diffusion生成点群と軌道を可視化する

```powershell
python .\src\sample_diffusion_toy.py `
  --checkpoint .\_outputs\diffusion\diffusion_toy.pt `
  --out .\figures\diffusion_toy_result.png `
  --device cpu `
  --samples 2048 `
  --trajectory-samples 64 `
  --steps 64 `
  --seed 0
```

| 引数 | 既定値 | 意味 |
|---|---:|---|
| `--checkpoint` | `_outputs/diffusion_toy.pt` | noise prediction modelのcheckpoint |
| `--out` | `figures/diffusion_toy_result.png` | 生成点群と軌道を保存する画像 |
| `--device` | `cpu` | 生成を実行するdevice |
| `--samples` | `2048` | source、target、generatedへ描く点数 |
| `--trajectory-samples` | `64` | 逆向き生成軌道を描く粒子数 |
| `--steps` | `64` | DDIM型更新のstep数。1 stepでモデルを1回呼ぶ |
| `--seed` | `0` | source noiseとtarget標本の乱数条件 |

### 5.12 Diffusion生成をNFE別に評価する

```powershell
python .\src\evaluate_diffusion_toy.py `
  --checkpoint .\_outputs\diffusion\diffusion_toy.pt `
  --device cpu `
  --samples 2048 `
  --target-samples 2048 `
  --steps 8 16 32 64 `
  --bandwidth 0.5 `
  --coverage-radius 0.35 `
  --seed 0 `
  --csv-out .\_outputs\diffusion\evaluation.csv `
  --plot-out .\figures\diffusion_toy_evaluation.png
```

| 引数 | 既定値 | 意味 |
|---|---:|---|
| `--checkpoint` | `_outputs/diffusion_toy.pt` | 評価するDiffusion checkpoint |
| `--device` | `cpu` | 評価device |
| `--samples` | `2048` | 各NFEで生成する標本数 |
| `--target-samples` | `2048` | 指標計算に使うtarget標本数 |
| `--steps` | `8 16 32 64` | 比較するDDIM型更新回数。ここではNFEと等しい |
| `--bandwidth` | `0.5` | MMDのGaussian kernel幅 |
| `--coverage-radius` | `0.35` | modeを覆ったと判定する半径 |
| `--seed` | `0` | 全条件で固定する乱数seed |
| `--csv-out` | `_outputs/diffusion_toy_evaluation.csv` | 数値評価の保存先 |
| `--plot-out` | `figures/diffusion_toy_evaluation.png` | NFE別評価図の保存先 |

Flow Matching評価と同じ標本数、seed、指標設定を使えば、生成分布とNFEの関係を横並びで比較できる。ここで生成するのは2D点群であり、画像生成やFID評価ではない。

CSVには各NFEのbiased MMD²、coverage、endpoint errorに加え、`mode_counts`、`bandwidth`、`coverage_radius` も保存する。Flow Matching側と同じ列の意味で比較できるため、評価設定をファイル外の記憶に頼らず追跡できる。

### 5.13 画像速度場モデルのshapeとgradientを確認する

実画像datasetを接続する前に、U-NetとDiTが画像と同じshapeの速度を返し、backwardできることを確認する。

```powershell
python .\src\check_image_model_shapes.py `
  --batch 4 `
  --channels 3 `
  --image-size 32 `
  --patch-size 4 `
  --device cpu `
  --seed 0
```

| 引数 | 既定値 | 意味 |
|---|---:|---|
| `--batch` | `4` | テスト画像のbatch size |
| `--channels` | `3` | 入出力画像のchannel数 |
| `--image-size` | `32` | 正方形画像の高さと幅 |
| `--patch-size` | `4` | DiTが1 tokenにまとめるpatchの一辺。画像サイズを割り切る必要がある |
| `--device` | `cpu` | forward/backwardを行うdevice |
| `--seed` | `0` | 入力と初期重みの乱数seed |

成功すると、各モデルについてinput/output shape、仮loss、パラメータ数、`gradients=ok` が表示される。これは画像生成品質を測る実験ではなく、長時間学習の前にshape不一致、patch設定、gradient断絶を検出するsmoke testである。

## 6. 研究用に改造するときの入口

| やりたい変更 | 触る場所 |
|---|---|
| pathを変える | `paths.py` |
| 学習する時刻分布を変える | `time_samplers.py` と `--time-sampler` |
| coupling差し替えを試す | `couplings.py` と `--coupling` |
| MLPの幅・深さ・時刻埋め込み・活性化関数を変える | `models.py` と各CLI引数 |
| U-Net/DiTへ進む | `image_models.py` |
| 画像モデルのshapeとgradientを確認する | `check_image_model_shapes.py` |
| Diffusion学習と比較する | `diffusion_basics.py`, `train_diffusion_toy.py` |
| Diffusionで生成・評価する | `diffusion_basics.py`, `sample_diffusion_toy.py`, `evaluate_diffusion_toy.py` |
| Rectified Flowを学習する | `models.py`, `losses.py`, `train_rectified_flow_2d.py` |
| solver比較をする | `solvers.py`, `evaluate_minimal_2d.py` |
| Reflowを試す | `reflow.py` |
| Reflowを一連の実験として実行する | `train_rectified_flow_2d.py`, `train_reflow_2d.py` |
| 評価指標を追加する | `evaluation.py` |

改造するときは、まず1つだけ変更する。path、coupling、model、solverを同時に変えると、結果が変わった原因を説明しにくくなる。
