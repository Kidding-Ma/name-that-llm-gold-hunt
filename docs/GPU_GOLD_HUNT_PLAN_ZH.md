# 冲击 0.66 的 GPU 研究与执行计划

> 这是一份给下一位 GPU AI/算法工程师的作战计划。它不能保证金牌；每个结论都必须以完整 hidden-test Kaggle 提交和分数为准。

## 一、当前最重要的事实

- 用户当前 CPU 提交真实分数约 `0.365`。
- 私有训练快照上，四视图 style-aware SVC 的 prompt-grouped OOF 为 `0.384484`。
- 加入 25 个数值风格统计后，严格 5-fold GroupKFold OOF 为 `0.388936`；证据在 `evidence/metrics/cpu_numeric_style_local_metrics.json`。
- 公开 two-stage Qwen3 8B/14B LoRA + public pool 报告 public LB `0.592`、grouped CV `0.593`，但本仓库尚未独立复现。
- 公开源码审计中，最高可信的可追溯目标是 prvsiyan v4：报告 `0.604`，并且源码绑定 COMPLETE submission `57428928` 和 `scriptVersionId=360252286`；这仍然是上游证据，不是本地复现。
- 用户看到的第一名约 `0.667` 尚未绑定到公开 notebook、版本或代码。不能从用户名/分数推断方法。

## 二、建议硬件和成本顺序

### 不要先租两张 4090

先用 Kaggle GPU 试跑。最小有效信息是一次完整 hidden-test 提交，而不是本地 CV。

| 路线 | 推荐资源 | 目的 |
|---|---|---|
| CPU | 当前电脑 | 控制组和格式检查；预计无法接近 0.66 |
| Qwen3-1.7B | 1×T4/16GB 或更高 | 低成本 embedding 控制，公开笔记本约 0.431 CV 级别 |
| Nishant 0.592 | Kaggle 2×T4，或同机 2×4090 | 首个大模型控制 |
| prvsiyan/Joseph/Kirill 多输入 | 2×T4 或 24–48GB GPU，RAM 64GB+，NVMe 200GB+ | 0.596–0.604 研究目标 |
| 自研 LoRA/多折训练 | 2×4090/A100 更合适 | 只有在已知推理路线有效后再投入 |

2×4090 不是“保证 0.66”的硬件门槛；它只是让复杂路线更容易运行。不要为了尚未验证的代码直接包月。

## 三、严格执行顺序

### Phase 0：提交记录基线

1. 使用 `scripts/kaggle_submit.py` 的 CPU 方案作为控制提交。
2. 保存 notebook 版本、提交文件 hash、public/private 分数和预测计数。
3. 用户报告的 `0.365` 作为真实基线；不能用本地 3 行 test 代替。

### Phase 1：复现 compact 0.592 控制

输入：

- competition `name-that-llm`；
- `nishantdahal/ntl-two-stage-lora`；
- Qwen3 8B Base；
- Qwen3 14B Base；
- 推荐 2×T4。

运行 `scripts/kaggle_best_two_stage_lora.py`，然后分别运行：

```text
--only-8b
--no-twin
--no-balance
```

每个开关单独一个 notebook 版本，不能同时改动。先得到真实 LB，才能判断输入、版本和 GPU 是否正确。

### Phase 2：先恢复历史依赖，再考虑 0.604 v4

**重要更新：** 当前 v10 的最新源码头明确表示：它是 unscored candidate，历史 branch B 的 Lightning checkpoint 不可用，因此禁用了该分支；没有宣称 v4 parity。下列当前输入清单不能补全历史 v4。先核实/恢复确切的 branch B 和所有版本；无法恢复时跳到 Phase 3，或命名为新的未评分实验。

不要先重写成一个“看起来类似”的脚本。打开原始版本，检查实际 cells 和 inputs 后才使用 Copy & Edit：

- [prvsiyan v4](https://www.kaggle.com/code/prvsiyan/the-prompt-is-not-the-author?scriptVersionId=360252286)
- 记录 `scriptVersionId=360252286` 和 COMPLETE submission `57428928`；这是当前 source 引用的作者历史证据，不是本地复现；
- 确认历史 branch B 的 Lightning checkpoint 是否存在、版本是否正确、是否可合法使用；
- 运行 `scripts/prvsiyan_route_preflight.py`；
- 挂载 v4 source 实际需要的所有输入，并私下记录每个版本号、许可证和 SHA-256；当前 v10 pull metadata 的名称只能作为线索；
- 优先运行作者的原生 source；
- 只有生成完整 hidden-test submission 并获得 LB 后，才写“复现”。

当前公开 v10 pull metadata 的输入清单（尚不能代替 v4 的精确 input version 清单）：

```text
josephadamski91/ntl-atlas-qwen3-8b-v1
prvsiyan/ntl-public-in-style-pool-v1
prvsiyan/ntl-two-stage-lora-adapters-v1
nishantdahal/ntl-two-stage-lora
qwen-lm/qwen-3/Transformers/8b-base/1
qwen-lm/qwen-3/Transformers/14b-base/1
```

### Phase 3：0.596/0.600 旁路

如果 v4 输入不可用，分别尝试，不能混合后把分数归因给某个方案：

1. [Joseph ATLAS + early layers](https://www.kaggle.com/code/josephadamski91/two-stage-lora-atlas-and-early-layers?scriptVersionId=360008790)，报告约 `0.596`；
2. [Kirill Teach the Model, Remove the Prompt](https://www.kaggle.com/code/kirilltym/0-600-teach-the-model-remove-the-prompt)，报告约 `0.600`；
3. [Kirill Gaussian Readout](https://www.kaggle.com/code/kirilltym/two-stage-lora-gaussian-readout) 只当方法参考，因为其源码明确没有该改版的已知分数。

### Phase 4：从 0.604 到 0.66 的实际研究

只有 Phase 1–3 取得真实分数后，才开始新研究。推荐每次只添加一个成员：

1. **Logit ensemble**：保存 Nishant、ATLAS、early-layer、style SVC、Qwen1.7B、DeBERTa/ELECTRA 各自 logits；用 outer prompt-grouped OOF + inner weight selection 训练融合。
2. **家族内 gate**：先做 provider gate，再在 Claude/GPT/GLM/Kimi/Qwen 内部分类；gate 只在训练侧 prompt-balanced 数据上拟合。
3. **额外公开数据**：优先使用每个 prompt 都包含全家族标签的 public pool；先做去重、prompt overlap 和 fidelity 检验，避免 topic shortcut。
4. **ATLAS/早层特征**：把 Qwen/encoder 的早层、中层、末层分别作为独立 head，不要只保留最终 embedding；计算均值池化、attention pooling、协方差/Gaussian readout。
5. **LoRA 多折**：如果有 2×4090/A100，再训练 5 个 prompt-fold LoRA heads；训练期间固定 epoch/权重，不能用外层验证集反复挑参数。
6. **centered twins**：只在 pseudo-test 中验证；用多个随机 prompt draw 估计是否真的提升，而非在一个 OOF 上调阈值。
7. **class calibration**：balanced accuracy 对弱类敏感。只能用训练侧/outer-fold 校准；不要把隐藏 test 类别配额当真。

## 四、最可能有效的模型组合

### 组合 A：简单高性价比

```text
Nishant full 8B+14B
+ style SVC
+ no-twin/full-twin isolated choice
+ no-balance/full-balance isolated choice
```

预计目标是先接近公开 `0.592`，不是直接 0.66。

### 组合 B：历史 v4 与当前 v10 必须分开

```text
历史 v4（作者报告 0.604）:
  双分支完整融合（包含目前不可用的 branch B）
  + branch A 的公开/自训 LoRA
  + 给各 two-stage LoRA 增加 shared-prompt Gaussian heads
  + branch A 自己的 TF-IDF grouping

当前 v10（明确未评分）:
  仅 branch A + Gaussian heads
  + 10% ATLAS
  + tokenizer/provenance 修复
  - 历史 branch B
```

ATLAS 与 early-layer 不能归因给 v4 的 0.604；early-layer 出现在 v5 历史条目，ATLAS 属于后续候选。当前源码保留多个历史头部，阅读时以明确版本为准。无法恢复历史依赖时，不能把删掉 branch B 的代码叫作 v4 复现。

### 组合 C：独立 ensemble

```text
public v4 prediction logits
+ Nishant prediction logits
+ Qwen3-1.7B layer-17/20 logits
+ CPU style-aware logits
```

不要简单平均概率。对每个成员进行 temperature/column calibration，并在 outer prompt-grouped OOF 上选择权重；若没有 OOF logits，就先做单独 Kaggle submission，不要凭感觉融合。

### 组合 D：家族 gate

基础 18 类模型先判断 provider，再使用 family-specific classifier：

- Claude 5-way：stylegram + DeBERTa；
- GPT 5-way：stylegram + DeBERTa + public GPT passages；
- GLM 2-way：style/encoder ensemble；
- Gemini 4-way：保持基础模型或轻量 gate；
- Kimi/Qwen：增加 frozen embedding 和 public pool。

家族 gate 可能提高 balanced accuracy，但也可能把 provider 错误放大。必须保留全局模型 fallback，并通过 outer OOF 比较。

## 五、评分和消融纪律

每次实验记录：

```text
commit:
notebook_url:
script_version_id:
gpu:
input identifiers and versions:
license/attribution:
train/test/pool hashes:
code variant:
runtime:
submission hash:
public LB:
private LB:
change from previous:
```

分数标签：

- A：本地 prompt-grouped OOF；
- B：公开 source/input 已检查，未复现；
- C：作者报告或用户观察；
- D：自己的完整 hidden-test Kaggle 分数。

任何“有效果”都必须意味着 D 级分数高于控制提交，或者至少有严格 outer-fold 证据支持；本地高 CV 不等于排行榜提高。

## 六、禁止事项

- 不要使用 hidden labels；
- 不要手工给 test 文本贴标签；
- 不要把每个 hidden prompt 恰好一个模型的假设当作事实；
- 不要默认 Hungarian assignment、Sinkhorn 或 test quota；
- 不要将二分类 detector AUROC 当成 18 类 balanced accuracy；
- 不要把作者报告的 0.604/0.600/0.592 写成本地复现；
- 不要把数据、权重、adapter、public pool 或 token 放进 GitHub。

## 七、给后续 AI 的一句话任务

先在 Kaggle 2×T4 验证 Nishant 原生 0.592 路线，再读取并运行 prvsiyan v4 实际 source/输入；所有结果按版本和提交记录。不要从 CPU TF-IDF 继续盲目调参，也不要为了“看起来接近 0.66”引入未经 pseudo-test 验证的 prompt assignment。
