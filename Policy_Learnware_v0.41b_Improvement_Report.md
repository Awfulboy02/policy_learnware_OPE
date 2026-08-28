# Policy Learnware v0.41b 方法增强报告

## 1. 结论与验收边界

v0.41b 已在 OPE companion 仓库的 `v041b` 分支完成一个诚实、可执行、可复现的 method-level synthetic MVP。最初方法增强 commit 为 `31c1648b68a89792a7a57d9bc14dc7630aa522e9`，数值门禁与 candidate-key 纠偏后的干净代码/测试快照为 `79ce56f3c388aa289f3dc624cecf88f13b6d7921`，均由冻结的 `v04b@637e6650b5ae419919b9ea65137bdc896bbfd6be` 普通派生；本报告作为后续 release commit 的 tracked 文件进入同一分支。

本轮完成两项方法增强：

- `FH_KMIFQE` 从静态 kernel/ridge proxy 升级为 `B20_PROTOCOL_ADAPTATION`。
- `ETM_MBOPE` 从 shuffle-negative proxy 升级为 `PROJECT_ETM_PROTOCOL_ADAPTATION`。

这里的 “protocol adaptation” 不等于官方论文数值复现。两条方法均能在轻量 synthetic fixture 上真实 fit/estimate；真实资产仍因 actor authority、精确 behavior density、折扣 oracle 及 Raw operator authority 缺失而 fail closed。本轮没有补采日志、重训 FPO、改写旧 oracle、运行全量论文实验或触碰 v0.4a。

## 2. 冻结来源核对

| 来源 | 锁定证据 | 使用方式 |
|---|---|---|
| B20 PDF：`B20_2024_Kernel_Metric_Learning_for_In_Sample_OPE.pdf` | SHA-256 `9e9edf0279bbd261b5aeec189c1a9b110eabf007859f975680b4e6267818cd67` | 核对 Eq. 7、Eq. 11、Proposition 3 与 Algorithm 1 |
| KMIFQE official | commit `070f121d29f05638221695690d5b0d1f0e2bf75b`，MIT | 只读审计训练循环、Hessian、带宽、重采样与 target update；未复制源码 |
| B22 PDF：`B22_2024_Offline_Transition_Modeling_via_Contrastive_Energy_Learning.pdf` | SHA-256 `bcaada273215fa3133ffee6aa187e2b3d0e4877f3f012dee889301ecc008cf6f` | 核对 Eq. 22、Eq. 23 与 Table 2 |
| ETM official | commit `2a2c780c0da074b6e7733a3cb6b40b2444452de6`，Apache-2.0 | 只读审计 Langevin schedule、noise/clip/detach 与 GP reduction；未复制源码 |

本轮实现是 clean-room NumPy adaptation。`THIRD_PARTY.md` 已记录出处、commit、许可证和 parity 否认。临时官方源码 clone 在核对完成后删除。

## 3. 代码与依赖

初始算法 commit 的 diff 为 10 个文件、`2186 insertions(+), 197 deletions(-)`；后续聚焦纠偏 commit 只修改 4 个既有模块和 4 个既有测试，`352 insertions(+), 71 deletions(-)`：

- 唯一新增算法模块：`src/policy_learnware_ope/kmifqe.py`。
- 唯一新增聚焦测试：`tests/test_kmifqe_protocol.py`。
- 其余增强集中在既有 `fqe.py`、`mbope.py`、`cli.py` 和既有 tests。
- 未新增 registry、factory、contract、schema scaffold；未改 `core.py`、ranking seal、oracle join 或 Raw adapter ABI。
- 包版本从 `0.4.0b0` 升为 `0.4.1b0`。
- 运行依赖仍只有 `numpy==2.5.2`；测试依赖仍只有 `pytest==9.1.1`。

方法身份变化：

| 方法 | v0.4b | v0.41b |
|---|---|---|
| KMIFQE | `KMIFQE_PROJECT_ADAPTATION` | `B20_PROTOCOL_ADAPTATION`，`official_parity=false` |
| ETM | `PROJECT_CONTRASTIVE_ENERGY_ADAPTATION_PROXY` | `PROJECT_ETM_PROTOCOL_ADAPTATION`，`upstream_parity_claimed=false` |

实际 `method_id` 始终由运行时 gamma/H 派生，例如 toy 为 `FH_KMIFQE_G099_H5` 和 `ETM_MBOPE_G099_H5`，不会用 H=5 的结果冒充 H1000。

## 4. KMIFQE：B20 机制映射

| B20 机制 | v0.41b 实现与证据 |
|---|---|
| candidate-specific nonlinear Q / target-Q | 每个 candidate 独立初始化固定 seeded tanh feature critic，并分别维护 fitted Q 与 hard target-Q 系数；不是原先全局对角 quadratic proxy |
| local action Hessian | 对 tanh critic 给出精确解析 action Hessian，在 `pi(s')` 逐状态计算；非 informative action 维不凭随机特征伪造曲率 |
| Mahalanobis metric | 按 Hessian 正/负 eigenspace 构造 SPD metric，施加 eigen floor/regularization，并执行 `det(A)=1` normalization |
| Eq. 11 bandwidth | 每轮从 TD-MSE bias²、variance、action dimension 与 active sample count 计算 `h*`；正式路径拒绝固定 median bandwidth |
| kernel / mu ratio | 使用 local Mahalanobis Gaussian kernel 与 arbitrary-action exact behavior density，应用显式 ratio clipping；以 `p=w/sum(w)` 重采样 |
| replacement resampling | seeded common-random uniforms 每轮按更新后的 p 重新映射为 replacement indices；记录 ESS、clip fraction、duplicate/unique fraction 和离散 index digest；机制测试令 draw count 大于 active count，由鸽巢原理确定性证明 replacement duplicate |
| in-sample TD | bootstrap 使用同一被抽中日志行的 `next_behavior_action`，不使用 `pi(s')`；连续行真实性按 `float64` 的 `atol=32*eps*max(1,max_abs_successor), rtol=32*eps` 核验并记录 max abs/max rel drift，所有 active row 均须由物理相邻行验证，否则 `NO_GO_MISSING_NEXT_BEHAVIOR_ACTION` |
| mean-weight correction | critic 目标为 `mean(w) * MSE + ridge * ||theta||²`，因此相同 p、不同 `mean(w)` 会改变 regularized critic；测试已隔离证明 |
| alternating loop | 每轮完整执行 Q/target-Q → Hessian/L → h → ratio/p → replacement resampling → logged-a' TD → target update，不再 H+1 后冻结 panel |
| fail closed | exact-density、support/ESS、adjacency authority、finite state/Q、target lag 和 convergence 任一不满足均不发布 value；陈旧 target 的错误 PASS 已有攻击型回归 |

KMIFQE 不再继承 FQE 的 `1e-9` 默认收敛阈值：方法自身默认 Q-space relative tolerance 为 `0.003`，replacement probability L1 tolerance 为 `0.015`。Q gate 使用 `tolerance * (1 + max_abs_Q)` 的明确尺度；不可辨识 fixed-feature coefficient 的绝对 L2 不再作为跨后端失败门禁，finite Q/TD 与收敛不变量仍严格检查。

二维 curved fixture 的机制证据：

- 状态：`PASS`；68 次 iteration，68 次 Hessian 与 target update。
- `h` 从 `10.0` 更新到 `0.4974388405`，轨迹 range 为 `9.5025611595`；测试逐轮从 bias²/variance 独立重算 Eq. 11。
- local metric state variation `0.2051753335`，off-diagonal Frobenius mean `0.0473973262`，最大 determinant error `1.11e-15`。
- active rows 144，replacement draw count 145，duplicates 63，unique fraction `0.5655172414`。
- final Q update delta `0.0119668360 <= 0.0131476875`，p-L1 delta `0.0005638082 <= 0.015`。
- 对称 `a'/-a'` 测试保持前两轮 h 与 resampled index digest 相同，同时第二轮 TD target 改变，隔离证明 logged adjacent action 真正进入 bootstrap。

toy 的 action dimension 为 1；determinant normalization 在一维必然把 metric 归一为 1，所以 toy 的 metric state variation 约为 `1.22e-16`。非退化 local metric 的证据来自上述二维 fixture，不能把一维 toy 写成 metric-learning parity。

## 5. ETM：Langevin negatives 与 gradient penalty

| B22/official 机制 | v0.41b 实现与证据 |
|---|---|
| conditional negatives | 训练期每个 mini-batch 从 `U[-1,1]` box coordinates 初始化 conditional Langevin chains；正式路径不再使用 shuffle/replay negatives |
| normalization | ETM target 使用 official-style zero-mean max-abs box normalization，scale 小于 `1e-12` 时取 1 |
| release chain convention | 采用官方 release 的 polynomial `0.1 → 0.001` schedule、noise `0.5`、gradient clip `10`、drift clip `0.5`、sample clip `1.1`，每步 NumPy value replacement 等价于 detach |
| B22 Eq. 22 GP | 实现 `sum_j relu(||dE/dy_j||-M)^2` 后按 batch mean；对 RFF theta 给出精确解析 VJP，并真实加入 Adam update |
| mini-batch/config | 支持 epochs、batch size、max optimizer steps、K、L_train、temperature、noise、clip、GP margin/weight；actual config 完整进入 artifact |
| finite gate | Langevin、NCE、GP/VJP、parameter gradient、Adam moments/bias-corrected state、theta 和 final panel 任一非有限立即抛错，失败 refit 保持 unfitted |

机制测试确认：

- train 与 final-panel chain 的位置和能量均发生变化，不依赖某个偶然精确 loss 常数。
- 聚焦抽查前 3 个 RFF theta 维的解析 GP VJP 与 finite difference 一致；GP 超 margin 时非零，并实际改变 theta update。
- tiny temperature 与极大 GP weight 两条路径均在发布模型前原子失败。
- fixed seed 的 theta/value 在本地锁定 `float64` 测试中显式使用 `rtol/atol` 比较；离散身份与 seed/config digest 仍精确。本轮没有建立跨 backend golden 或 tolerance registry。

ETM 推断 sampler 不在本轮训练机制验收范围内，且与论文/官方 release 均存在重要漂移：当前采用 conditional center + Gaussian initialization、noise `1`、clip `[-10,10]`、默认 20 步（toy 10 步）；论文 OPE 描述约 100 步/noise `0.1`，官方 release 又采用 zero initialization、默认零噪声并沿用 polynomial schedule/clips。该差异可能改变正式 OPE value，因此在正式 benchmark 前保持 `NO_GO_ETM_INFERENCE_PROTOCOL_ALIGNMENT`。当前计算计数也未覆盖 final diagnostic panel 与 inference 的全部 gradient evaluations，发表成本表前必须补齐；本轮不为此扩写 sampler。

## 6. 实际 toy 配置与结果

统一设置为 seed 41、gamma 0.99、H=5、48 episodes、5 candidates。主要实际配置：

- FQE：ridge `1e-7`，max iterations 2500，tolerance `1e-8`。
- KMIFQE：32 tanh features，ridge `1e-7`，max iterations 200，Q tolerance `0.003`，p-L1 tolerance `0.015`，critic damping `0.1`，eigen floor `1e-6`，metric regularization `0.1`，h range `[0.001,10]`，ratio clip `[0.001,2]`，target update interval 1，min ESS fraction `0.01`。
- ETM：24 hidden center features、48 RFF energy features、3 generated negatives + 1 positive、12 epochs、batch 48、最多 40 Adam updates、learning rate `0.01`、L_train 5、training schedule `0.1→0.001`、noise `0.5`、GP margin/weight `5/1`；inference L=10、step `0.025`、noise 1、clip 10。
- model-based common：ridge `1e-4`、12 rollouts/initial state、horizon-only termination。

两次独立 CLI 运行均为 `TOY_MVP_PASS`，并共同得到：

- implementation commit `79ce56f3c388aa289f3dc624cecf88f13b6d7921`，tree `b01192c9a137424c40ca4f83e107c331cbff8434`，worktree `CLEAN`。
- config SHA-256 `ff94a4f6389bb1c775ed5f6307c0ebab0fc4ce5fccab32aaad1d051e3ff45f08`；其中 candidate key 由 `SHA256(seed, method_id, candidate_id, phase)` 的前 64 bit 与 row index XOR 派生，不再依赖 candidate list position。
- discrete reproducibility identity SHA-256 `fd45d519bd8f149f4ded269238552131f105ced90bfe7d318588a90c41d52b61`。该 digest 只绑定 implementation/config/dataset/candidate IDs/status/ranking/winner 等离散语义，不哈希浮点 estimates。
- 两个 `run.json` 因 seal 外 runtime 不同而分别为 `d368b820...` 与 `9db0f29f...`；每次 ranking seal 仍由各自精确 digest 做单 artifact integrity 校验，但跨合法 backend 不要求浮点 payload 的 seal digest 相等。

toy 排名结果如下；这是 synthetic fixture 结果，不是论文 benchmark：

| 方法 | ranking | Hit@1 | 备注 |
|---|---|---:|---|
| FQE | 0,1,2,3,4 | 1 | value MAE `0.00722` |
| KMIFQE | 1,2,0,3,4 | 0 | 五 candidate 均 PASS；value MAE `0.17326`，诚实保留失败的 top-1 决策 |
| ETM | 1,0,2,3,4 | 0 | value MAE `0.03877`；不把 synthetic top-1 失败隐藏为浮点噪声 |
| DOPE-style MB-FF | 0,1,2,3,4 | 1 | project-defined reference |
| AR-MBOPE | 0,1,2,3,4 | 1 | independent reference |
| RAW_ADAPTER_FIXTURE | 3,2,4,1,0 | 0 | 仅 toy fixture；production Raw 仍 NO_GO |

KMIFQE toy candidate 0：value `4.8106561999`，ESS fraction `0.66891`，clip fraction `0.10938`，unique resampled fraction `0.510417`，76 次完整 alternating updates。ETM toy candidate 0：value `5.0205986053`，40 optimizer updates，training chain mean energy `32.2173 → 3.2593`、逐链 mean absolute energy change `28.9580`，final panel `33.3759 → 3.8654`，GP last `442.0883`，active rate `0.3529`。

候选池反序回归使用相同 seed 重新运行全部方法：按 candidate ID 对齐的 value 在显式 `rtol/atol` 内一致，所有 ranking 与 winner 精确不变。该测试直接覆盖了旧 `candidate_index*offset` 导致 KMIFQE winner 随候选顺序变化的语义错误。

## 7. 数值 gate 原则

本轮没有引入 tolerance framework。规则集中在现有比较与聚焦测试中：

- commit/tag、asset/config/protocol digest、candidate/context/membership/seed、shape/schema、权限与 seal expected digest 使用精确比较。
- 浮点结果使用测试中显式、dtype/尺度相称的 `atol/rtol`，并同时检查 finite、SPD/界限、机制调用计数、相对趋势、排序和最终决策；不以 derived-float hash 作为跨 backend 数值 gate。
- 本轮没有跨 backend golden/tolerance registry。合法 backend 改变时，每次 artifact 仍绑定自己的精确 digest，但跨跑数值用显式容差比较，ranking/winner 作为决策语义精确比较。
- adjacent action真实性同时要求全部 active row 有物理连续 authority；容差只吸收序列化末位漂移，max abs/max rel drift 与 tolerance 一并导出，实质错配仍 `INVALID_DATA`。
- NaN/Inf、数量级异常、动作/状态语义错误、资产混用或排名改变不能被容差掩盖。

## 8. 测试与命令

算法 commit 的干净工作树验证：

```text
.venv/bin/python -m pytest -q
93 passed in 10.57s

.venv/bin/python -m pytest tests/test_fqe.py tests/test_kmifqe_protocol.py tests/test_mbope.py tests/test_cli.py -q
38 passed in 10.25s

git diff --check
PASS
```

覆盖范围包括 finite-horizon known MDP、native t/H 与 termination/dataset-cut mask、exact density/support、adjacency roundoff/authority、stale-target/nonconvergence/nonfinite-Q fail-closed、Eq. 11、local Hessian metric、replacement resampling、logged-a'、mean-weight Q-space effect、ETM conditional Langevin、GP VJP/真实参数影响、candidate permutation invariance、两次 fixed-seed toy、ranking/metrics export、CLI smoke 与 real-preflight。

## 9. Production fail-closed preflight

`real-preflight` artifact SHA-256 为 `d68b32c8e63cc5deadf59a80f7570af6f6b66270d2b6f47565ac117d47e6379c`，config SHA-256 为 `00f00677f7d9e4fe64ba572501037e20b33fe05db337badc3511c6ca42a3e2e5`。状态为 `NO_GO`，且：

- `production_training_started=false`
- `asset_mutation_started=false`
- actor authority：`NO_GO_ACTOR_AUTHORITY`
- discounted oracle：`NO_GO_ORACLE_DISCOUNTED_VALUE`
- exact behavior density：`NO_GO_EXISTING_LOG_DENSITY`
- production Raw：`NO_GO_RAW_OPERATOR_AUTHORITY`

因此本轮没有用 toy capability 或 truthy manifest 字段冒充生产权限。

## 10. 冻结性证明

OPE 仓受保护引用在算法提交前后保持：

- `main == origin/main == 6f5dc2fabdc73aa2ad22ab06c1a404874c3bdf48`
- `v04b == origin/v04b == 637e6650b5ae419919b9ea65137bdc896bbfd6be`
- annotated tag `ope-bootstrap-v0` object `3de93ce3745f5e78f5ed1a3d9dce3351542c56f4`，peeled commit `6f5dc2f...`

旧仓只读核验：

- `v03 == origin/v03 == 8b979f08c1d67e0eabfbda53b539ce67f21a6cfb`
- `v03^{tree} == 7802977523fe2b0334b2f041a9fd8874e68f4aee`
- `v0.3.1` peeled commit 同为 `8b979f08...`
- v03 冻结路径 scoped diff 为空。

任务中间快照曾观察到并行 v0.4a 工作对
`server/repro_fpo_ppo_v04a/bpr_runner.py` 的修改；最终复核时旧仓 HEAD 已由该并行工作推进到
`de184808bc83fdeaba9ed81bdf548e364345402a` 且 porcelain 为空。这个并发状态变化不属于本轮；贯穿两次快照的冻结事实是 v03 ref/tree 与 scoped paths 均未变化，本轮没有写入旧仓。

本地小型 manifest 锚点：pool `8cb95e...`、inventory `c21e15...`、championization `ce6b8e...`、returns `853a68...`、protocol manifest `e58928...`、selector pool `7f9d78...`。交叉 digest 一致，inventory 为 complete、60 items/0 rejected，其中 FPO 30、6 tasks × 5 seeds；30 个本地 FPO candidate manifest complete，30 个 parity report passed/raw_checked。远端巨型 bundle/log 本轮未逐字节重哈希，所以结论限定为“v03 Git 冻结 + 本地 manifest/cross-link/digest 未见变化”。

## 11. 已复原、剩余漂移与延期项

已复原到 method-level：B20 local Hessian metric、Eq. 11 h、kernel/mu、replacement IR、logged-a' TD、target alternation与严格收敛；B22 conditional Langevin training negatives、official release schedule/clip/detach，以及会真实改变参数的 Eq. 22 GP。

仍有公开漂移：

- KMIFQE 使用 fixed seeded tanh basis + damped analytic regularized output fit，而非官方 fully-trained 2×256 PyTorch critic/Adam。
- KMIFQE 使用每轮 remap 的 common-random replacement uniforms，而非每轮 fresh multinomial minibatch。
- KMIFQE 使用 project overall scalar ratio clip `[1e-3,2]`，未复刻官方更宽及逐维 clipping 语义；这是重要科学漂移，不得用 toy PASS 淡化。
- KMIFQE 使用项目 finite-H native-time mask 与 raw `J_gamma,H`，不是官方 normalized continuing convention。
- KMIFQE 当前 materialize `N×d×d` Hessian、metric 与 transform；在 `N=10^6,d=17` 量级峰值可能超过 13 GB。未在无证据下仓促加入 chunk framework，正式 OPS-DS 前状态为 `NO_GO_OPS_DS_DENSE_HESSIAN_PANEL`。
- ETM 使用 RFF energy + ridge center，而非官方 4×200 ReLU MLP。
- ETM training chain选择官方 release 的 polynomial schedule；该 release 与论文 Eq. 23/Table 2 在 drift/step 表述上存在差异。
- ETM inference 的 initialization/noise/clip/step count 同论文与官方 release 都未对齐，正式 value benchmark 前为 `NO_GO_ETM_INFERENCE_PROTOCOL_ALIGNMENT`；当前成本计数也未包含 final diagnostic/inference 的全部 gradient evaluations。
- synthetic linear fixture 不验证论文 discontinuity/OOD 场景，也不继承 DOPE claim。
- 没有 MuJoCo/D4RL 或论文 benchmark numerical parity claim。

真实实验延期原因不变：旧 clipped-Gaussian 日志无 arbitrary-action exact density；旧 oracle 无 per-step reward，不能构造锁定的 `J_0.99,H1000`；现有 FPO actor 尚无可验证 deterministic authority；Raw-Delta/RKME operator 尚未形成 digest-locked production adapter response。

## 12. 报告副本

canonical tracked copy 位于 OPE 仓库根目录。另两份同字节副本放置于 workspace 根目录和服务器 `/share/songyf/RL_Learnware/` 文档目录。报告自身不嵌入自身 SHA-256，以避免自引用；三处 hash 在发布后由外部 `sha256sum` 核验并在最终交付信息中报告。
