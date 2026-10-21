# CPU 运行手册

## 适用场景

没有 GPU、需要快速建立控制提交、或想先确认 Kaggle 数据/提交流程时使用。

## 推荐脚本

基础方案：[`../scripts/kaggle_submit.py`](../scripts/kaggle_submit.py)

当前发布版脚本包含四路 style-aware TF-IDF 和 25 个数值 style statistics；工作区记录的 grouped OOF 约为 `0.388936`。它仍然必须在完整 hidden test 上真实提交验证。

它使用：

- raw char 3–5 TF-IDF；
- word/punctuation 1–3 TF-IDF；
- case-shape char 2–6；
- function-word/punctuation 1–3；
- `LinearSVC(C=0.5)`。

本地私有快照的 5-fold prompt-grouped OOF balanced accuracy 为 `0.384484`。这不是 hidden-test 分数保证。

第二候选：[`../scripts/kaggle_submit_blend.py`](../scripts/kaggle_submit_blend.py)

它把 style-aware SVC 与原始 char+word+scalar-style SVC 做 row-softmax 融合。已有折外探针约 `0.393`，但不是完整 nested CV，因此必须单独作为 Kaggle 实验验证。

## Kaggle 提交步骤

1. 在比赛页创建 Code Notebook。
2. 加入 `Name That LLM` competition data。
3. 保持 CPU 即可。
4. 将目标脚本完整复制到一个代码单元格。
5. 运行并等待 `/kaggle/working/submission.parquet`。
6. 检查列、行数、ID 顺序和标签。
7. Save Version / Run & Save All。
8. Submit to Competition。

## 不要做的事情

- 不要上传本地 3 行 `test.csv` 预测文件。
- 不要把 `sample_submission.parquet` 当预测结果。
- 不要把随机行 CV 当主结果。
- 不要在 CPU 脚本中添加 prompt assignment 或隐藏 batch 配额。

## 本地运行

如果你已经有 Python 科学环境：

```powershell
python scripts/final_inference.py --data-dir D:/path/to/private/data --output submission.parquet
```

本地 test 仅能验证代码和输出格式；不能获得真实 leaderboard 分数。

## 研究验证

本地实验必须使用 5-fold `GroupKFold(prompt_id)`，并且每个 fold 内单独拟合 TF-IDF。若修改特征或权重，参数选择应在 inner grouped CV 完成。
