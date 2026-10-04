# Qwen 提示词基线

本阶段只运行预训练指令模型，不更新参数、不租 GPU。模型为 Qwen/Qwen2.5-0.5B-Instruct；官方来源 https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct ，Apache 2.0，许可证随下载保留。

## 模型与环境

- 固定 revision：`7ae557604adf67be50417f59c2c2f167def9a775`，见 `model_lock.json`。
- 权重和 tokenizer 位于 `models/Qwen2.5-0.5B-Instruct/`，文件大小和 SHA-256 见 `model_files.json`。
- 本次使用系统 Python `D:\Programs\Python312\python.exe`，不是课程 `.venv`。系统环境已有 torch 2.11.0+cpu、transformers 5.5.4、huggingface_hub 1.11.0、sklearn 1.8.0，未安装新包。
- 之前 TF-IDF 使用课程 `.venv` 的 sklearn 1.9.0；本次配对比较直接读取该基线已保存的预测，没有在另一环境重新训练分类器。

## 提示词与公平比较

`prompt_labels.json` 是助手编写的中文标签释义，并非 MASSIVE 官方标签定义。它是实验设计的一部分，存在解释偏差。当前是 zero-shot：提供全部 60 个标签及其释义，没有加入训练样例；使用官方 chat template。

要求模型只生成 `{"intent":"标签"}`。原始输出直接评测，不剥离 Markdown，不修复 JSON，不把别名自动映射为标签。这样可以区分 JSON 合法、标签合法、分类正确三个层次。

固定生成：CPU float32、6 线程、greedy decoding、最多 40 个新 token。下载模型附带的 temperature/top_p/top_k 不用于 greedy 解码；运行时出现的忽略采样参数提示不是训练或推理失败。

初步评测按照固定 ID 哈希顺序，从 dev 每个存在的类别选 2 条，得到 118 条。**dev 缺少 audio_volume_other，实际只有 59 类**，不是两个类别各只有一条。全部标签仍作为模型候选并进入固定 60 类 Macro-F1，缺失类按 0 处理。该样本近似类别均衡，不代表完整验证集的自然分布，不能直接拿它与全 dev 的 82.98% 比较。

配对 TF-IDF 使用完全相同的 118 个 ID。此试验比较的是“训练过的任务分类器”和“未做任务微调的语言模型”，不能据此断言模型架构孰优。小样本方差较大，结果用于判断下一步，不作为最终成绩。

## 复现

从课程根目录运行：

```powershell
# 本地文件已下载，推理不需联网
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
python Code/intent_router/prompt_baseline.py
```

每次生成新 `runs/qwen-*` 目录，包含 config（模型版本、源代码/提示词/数据哈希、样本 ID）、完整 prompt、选中 gold、逐条原始预测和耗时、统一评测、配对 TF-IDF 指标。不要将 3 条 smoke 测试的指标作为性能结果。

进程中断后可以在相同配置下续跑，逐条结果已经写入磁盘：

```powershell
python Code/intent_router/prompt_baseline.py --resume Code/intent_router/runs/qwen-20261005-040049
```

程序拒绝配置或脚本哈希不匹配的续跑。不使用 `--per-class 0`，除非确实要跑全 dev；它可能耗时明显更长。本阶段没有运行测试集。

## 阅读任务

打开此轮 `metrics.json` 的 errors，再回到 `predictions.jsonl` 找相同 ID：区分“JSON 坏了”“生成了非法标签”“合法标签但语义错了”。先理解这三类错误，再考虑训练。当前提示词未做搜索；不要为了更好看的数字不断试提示词并把最好一次当作无偏成绩。

## 本轮结果

完整运行：`runs/qwen-20261005-040049/`，2026-10-05，由助手执行。前面的 3 条 smoke 没有用于改写提示词；正式 pilot 使用同一个固定提示词。

| 同一组 118 条 dev 样本 | Accuracy | 固定 60 类 Macro-F1 |
| --- | ---: | ---: |
| Qwen zero-shot | 24.58% | 22.17% |
| 已训练的 TF-IDF + LinearSVC | 73.73% | 71.17% |

Qwen 正确 29 条；输出不满足 schema 43 条（其中 2 条不是合法 JSON）；合法标签但分类错误 46 条。JSON 合法率 98.31%，完整 schema 合法率 63.56%。schema 要求对象只含 intent 且值属于合法标签集合，不接受额外字段。

118 条生成合计约 199.89 秒，均值约 1.69 秒/条；输入 563–582 token，没有样本触及 40 新 token 上限。时间不含模型加载、数据读取，且本机负载会影响速度。没有 GPU 费用。

结论限于：当前小模型、固定中文 zero-shot 提示词与这一 pilot，不能可靠执行 60 标签分类协议。不能推出 Qwen 全系列能力、优化后提示词表现或微调收益。全 dev 与测试集还未跑；不把此成绩与其他公开榜单比较。

下一步建议先准备一致的 SFT 数据和小规模微调对照；如果另做 few-shot 提示词，它必须作为独立实验保留，示例仅来自训练集。最终保留测试集用于冻结方案后的比较。
