# GPU 冲金牌运行手册

## 目标

本手册不是保证分数的教程，而是把当前公开、依赖明确、最值得首先复现的 GPU 方案交给下一位 AI/算法工程师。

第一优先级是：

> 先完整复现公开的 two-stage LoRA + public pool 方案，得到真实 Kaggle 分数，再决定是否继续改模型。

对应脚本：[`../scripts/kaggle_best_two_stage_lora.py`](../scripts/kaggle_best_two_stage_lora.py)

更高分的公开研究目标已经审计并记录在 [`PUBLIC_GPU_ROUTES.md`](PUBLIC_GPU_ROUTES.md)：

> prvsiyan 当前 v10 源码把历史 v4 的 **0.604** 绑定到 COMPLETE submission `57428928` 和版本 URL `scriptVersionId=360252286`。但 v10 同时明确：历史双分支方案的 Lightning checkpoint 不可用，当前候选删除了 branch B 且未评分。不能把当前输入栈直接叫作 0.604 复现。

公开来源：

- [Two-stage LoRA on the public pool](https://www.kaggle.com/code/nishantdahal/two-stage-lora-on-the-public-pool)
- [配套数据集 nishantdahal/ntl-two-stage-lora](https://www.kaggle.com/datasets/nishantdahal/ntl-two-stage-lora)
- [The Prompt Is Not the Author — prvsiyan v4](https://www.kaggle.com/code/prvsiyan/the-prompt-is-not-the-author?scriptVersionId=360252286)
- [0.600 Teach the Model, Remove the Prompt](https://www.kaggle.com/code/kirilltym/0-600-teach-the-model-remove-the-prompt)
- [Two-stage LoRA + ATLAS and early layers](https://www.kaggle.com/code/josephadamski91/two-stage-lora-atlas-and-early-layers?scriptVersionId=360008790)

Nishant 上游作者报告 grouped CV `0.593`、public LB `0.592`。这是上游报告，不是本仓库独立复现结果。

## 推荐环境

### Kaggle

- Accelerator：优先 `2×T4`；单卡不一定能容纳 14B 截断模型。
- Internet：如果所有输入已挂载，建议关闭。
- 运行时间：公开说明约 1–2 小时，取决于资源和缓存。
- 不需要把数据下载到本地，也不要把权重提交到 GitHub。

### 私有服务器

建议最低：

- GPU：2×T4 16GB，或一张显存更大的 24–48GB GPU；
- RAM：32GB 以上；复杂 prvsiyan/Joseph/Kirill 组合建议 64GB+；
- 磁盘：100GB 以上；复杂多输入路线建议 200GB+ NVMe；
- Python：3.10–3.12 更稳妥；
- PyTorch、Transformers、Accelerate、safetensors、SciPy、scikit-learn、PyArrow。

如果只有单张 T4，先做 `--only-8b` smoke/ablation，不要声称它等于完整方案。复杂的 prvsiyan/Joseph/Kirill 组合不能按“单脚本、单数据集”估计资源。

## Kaggle 操作步骤：先复现 0.592 控制

1. 打开 [Name That LLM](https://www.kaggle.com/competitions/name-that-llm)，加入比赛。
2. 创建 Code Notebook，选择 GPU，优先 2×T4。
3. Add Input → Competition → `Name That LLM`。
4. Add Input → Dataset → `nishantdahal/ntl-two-stage-lora`。
5. Add Input → Models：
   - Qwen3 8B Base，Transformers 格式；
   - Qwen3 14B Base，Transformers 格式。
6. 将 `scripts/kaggle_best_two_stage_lora.py` 全部复制到一个代码单元格。
7. 首次运行默认完整模式：

   ```text
   python kaggle_best_two_stage_lora.py
   ```

   在 Notebook 单元格中直接运行时，复制文件内容即可；不要复制 shell 命令本身。

8. 等待生成 `/kaggle/working/submission.parquet`。
9. 检查：
   - 列严格为 `id,model`；
   - 行数等于完整 test；
   - ID 顺序与 test 相同；
   - model 全部属于 18 个合法标签；
   - 预测分布没有异常地只剩一个类别。
10. Save Version / Run & Save All。
11. Submit to Competition。
12. 记录 Notebook 版本、GPU、运行时间、public/private 分数和提交备注。

## Kaggle 操作步骤：再研究 0.604 目标

只有 0.592 控制已成功运行并提交后，再进行此路线。不要把当前 v10 页面直接当作“运行 v4 就能得到 0.604”。

1. 打开 [prvsiyan v4 source URL](https://www.kaggle.com/code/prvsiyan/the-prompt-is-not-the-author?scriptVersionId=360252286)，优先使用作者的 **Copy & Edit**，而不是从本文档重写整套 notebook。
2. 记录原始版本 URL、notebook id、版本号、运行状态和输入版本。**先确认历史 branch B 的 Lightning checkpoint 能合法挂载。** 当前源码说它不可用；若无法恢复，不继续声称复现 v4，改测 Joseph 0.596 或明确命名的新候选。
3. 核实 v4 实际依赖；[`PUBLIC_GPU_ROUTES.md`](PUBLIC_GPU_ROUTES.md) 中四个数据集、Qwen3 8B/14B 清单只来自当前 v10 metadata，**不包含已确认可用的历史 branch B**。
4. 运行 [`scripts/prvsiyan_route_preflight.py`](../scripts/prvsiyan_route_preflight.py)。缺项先修复；即使 exit 0，也只证明当前 metadata 的粗预检通过，不证明历史依赖齐全。
5. 先使用作者的原生版本/source，保存新版本前记录每一个输入的 dataset/model/kernel version。
6. 运行完成后只提交完整 hidden test。保存：submission 文件 hash、行数、预测计数、GPU、时间、public/private LB。
7. 如果要把该路线移植到本仓库，先单独建立私有实验分支；在没有确认外部 checkpoint、数据许可、完整 source 和输入版本前，不把大型 notebook/blob/权重复制进 GitHub。

### 当前 v10 的输入清单（不是 v4 完整依赖）

当前 metadata 报告的主要输入为：

```text
josephadamski91/ntl-atlas-qwen3-8b-v1
prvsiyan/ntl-public-in-style-pool-v1
prvsiyan/ntl-two-stage-lora-adapters-v1
nishantdahal/ntl-two-stage-lora
qwen-lm/qwen-3/Transformers/8b-base/1
qwen-lm/qwen-3/Transformers/14b-base/1
```

当前 v10 候选包含 style/n-gram、ATLAS、Gaussian、文本分组和 balancing 等组件，但它明确是 **unscored candidate**，与历史双分支 v4 不同；Branch B 的 Lightning checkpoint 被禁用/不可用。当前仓库只提供 preflight 和 provenance；没有声称已经移植或复现完整 v4。

## 预检

### 0.592 控制

在运行完整模型前，把 `scripts/lora_kaggle_preflight.py` 粘贴到独立单元格运行。它只检查：

- CUDA/GPU 是否可见；
- competition train/test 是否存在；
- public pool 是否存在；
- Qwen 模型配置是否存在；
- adapter/head 文件是否存在。

如果大量输入显示 `MISSING`，不要直接运行完整模型。先修复挂载。

### 0.604 研究目标

使用 [`prvsiyan_route_preflight.py`](../scripts/prvsiyan_route_preflight.py)。它不会训练或预测，只检查比赛 train/test、Nishant 标记、Qwen 8B/14B config，以及 ATLAS/in-style 输入的可见性。它是路径/结构预检，不是成功或合法性的证明。

如果大量输入显示 `MISSING`，不要直接运行完整模型。先修复挂载，并记录精确版本/许可证。

## 独立消融（先针对 0.592 控制）

完整运行成功后，再分别测试以下版本，每个版本单独保存和提交：

```text
--only-8b
--only-14b
--no-twin
--no-balance
```

解释：

- `--only-8b`：只用 8B adapter，诊断显存和 8B 分支；不是完整报告方案。
- `--only-14b`：只用 14B adapter，隔离 14B 分支；需要足够显存，不是完整报告方案。
- `--no-twin`：关闭 cluster-centered twin，检查 transductive center 是否有帮助。
- `--no-balance`：关闭 Sinkhorn，检查 class balancing 是否导致分布偏差。

不要同时改多个开关，否则无法知道是哪一部分造成分数变化。

为了后续分析，可在脚本模式增加 `--scores-output /kaggle/working/private_scores.npz`。它私下保存对齐的 SVC/LoRA 概率、twin 加权贡献、平衡前后的融合分数与分组；不是 OOF，也不是可提交的预测文件。不要上传 GitHub；没有独立 OOF 时不能据此反复调权重并宣称验证提升。

## 如果完整方案失败

按错误类型处理：

### 显存不足

1. 先运行 `--only-8b`。
2. 减小 batch size（需要记录为改动）。
3. 改用 24–48GB GPU 或 2×T4。
4. 不要把 CPU fallback 当作 LoRA 结果。

### 找不到 pool/checkpoint

确认已添加 `nishantdahal/ntl-two-stage-lora`，并检查是否有：

- `pool_v1.parquet`；
- `lora8b_v1/meta.json`；
- `lora14b_v1/meta.json`；
- 对应 `adapter_model.safetensors` 和 `heads.safetensors`。

### 标签顺序不匹配

脚本会拒绝运行。不要强行改标签顺序；先检查 checkpoint metadata 和比赛数据版本。

### 输出行数错误

最常见原因是使用了本地 3 行 test。必须从 Kaggle Notebook 的完整 competition input 读取 test。

## 继续冲击 0.667 的研究顺序

若 two-stage LoRA 实际分数远低于榜首：

1. 搜索更近期的公开 Notebook，而不是盲目调参。
2. 对每个候选方案记录：公开 URL、版本、输入、GPU、CV、LB、许可证。
3. 先增加一个独立 encoder/embedding head，再重新做 prompt-grouped OOF 融合。
4. 对 Claude/GPT/GLM/Kimi/Qwen 做 family-specific gate，但训练数据必须有 prompt-balanced 的额外样本。
5. 对 centered twin、Sinkhorn 和 test clustering 做 pseudo-test stress test。
6. 保存每个成员的 logits，避免只保存最终标签而无法分析。

不建议默认：

- Hungarian assignment；
- 强制每个 prompt 每个模型只出现一次；
- 用隐藏 test 的类别配额；
- 使用未授权或未记录来源的外部数据；
- 在同一个 OOF 上无限搜索融合权重。
