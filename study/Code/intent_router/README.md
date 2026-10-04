# 中文助手指令路由：本机基线

云端训练入口：[Colab 使用说明](COLAB.md)，先免费 GPU 短测，再正式训练。

状态：2026-10-05 已完成官方数据准备、重复检查、统一评测和 CPU 基线。Qwen 模型已下载并接入本机推理，见 [提示词基线](PROMPT_BASELINE.md)。已运行两步 LoRA 检查，尚未正式微调或租卡。此轮由助手执行，需要你自行复现与解释后才能作为个人经历。

后续进展：已完成 [微调数据与本机检查](SFT_PREPARATION.md)：冻结 1000/5000 条嵌套训练集、检查答案 loss mask，并实际通过两步 CPU LoRA 更新和保存重载检查。尚未进行正式微调或租卡。

最新进展：[LLaMA-Factory 验证](FACTORY_VALIDATION.md)已通过：两组完整训练数据和 dev 的实际框架预处理检查、两步 CPU 框架训练及 adapter 保存完成。框架环境版本已记录，正式 GPU 训练尚未开始。

## 从这里开始

在课程根目录 `D:\cuhksz\CIE6032` 的 PowerShell 中运行：

```powershell
.venv\Scripts\python.exe Code/intent_router/test_evaluate.py
.venv\Scripts\python.exe Code/intent_router/baseline.py
```

数据已经准备好。每次基线运行创建新目录，入口是 `runs/latest.json`，里面有报告、逐条预测、错误案例与逐类别指标。不会覆盖旧结果。

## 本次结果（验证集，非最终测试结果）

固定配置：中文字符 1–3 gram TF-IDF，min_df=2，sublinear_tf=True；LinearSVC C=1、seed=42、max_iter=5000。词表与 IDF 仅用训练集拟合，没有搜索超参数。

| 数据 | 样本数 | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| 官方 dev | 2033 | 82.98% | 80.15% |
| dev 中排除训练集已出现的规范化文本 | 1889 | 82.27% | 77.91% |

本机本次拟合约 0.56 秒，dev 批量预测约 0.019 秒，22471 个特征；不包含 Python 启动和数据加载，不是通用速度承诺。结果目录：`runs/20261005-034909-401906/`。

传统分类器的 JSON 是程序包装的，合法率不代表模型学会了结构化生成；以后评测 LLM 时才衡量其实际输出是否合法。所有非法输出均保留在准确率和 F1 分母中。

## 数据来源与审计

- 发布方：Amazon；[官方仓库](https://github.com/alexa/massive)、[数据卡](https://huggingface.co/datasets/AmazonScience/massive)。
- 使用官方 MASSIVE 1.1 压缩包的 zh-CN 文件，CC BY 4.0。原始许可证保存在 `data/raw/LICENSE`。
- 原始下载地址及 SHA-256 见 `data/audit.json`。保留原始压缩包和中文 JSONL；转换版仅保留 id、utt、intent，并按官方 partition 分开。
- train/dev/test = 11514/2033/2974，60 类，ID 唯一，文本非空，dev/test 标签均在训练标签集合中。
- NFKC、转小写、去空白后，共发现 571 组重复文本、53 组存在标签冲突的文本、319 组跨划分重复文本。这里只发现字面重复，不代表已排除语义泄漏。
- 官方划分不改；额外报告 dev 中未在 train 出现的文本子集。子集类别分布改变，因此两个 Macro-F1 不能直接解释为纯粹的泄漏影响。
- 测试集只做结构和重复审计，本轮没有推理、算分或查看其预测错误。最终统一比较时再使用。

若需重新下载和准备：

```powershell
Invoke-WebRequest -Uri 'https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz' -OutFile 'Code/intent_router/data/raw/massive-1.1.tar.gz' -UseBasicParsing
.venv\Scripts\python.exe Code/intent_router/prepare.py
```

首次下载需联网。prepare.py 只从压缩包读取指定文件，不执行数据集代码，不解压任意路径。

## 文件与下一步

- `prepare.py`：划分、检查、重复审计、来源哈希。
- `evaluate.py`：按 ID 严格对齐；JSON/schema 合法率、准确率、60 类 Macro-F1、错误案例。
- `test_evaluate.py`：验证非法输出不被漏算、预测顺序无关、缺失/重复 ID 拒绝。
- `baseline.py`：训练集拟合、验证集评测，记录参数、环境、耗时、代码与数据哈希。
- `requirements-local.txt`：本次实际环境中的版本，现有课程 .venv 已具备，不需要重新安装。

你本阶段只需完成三件事：运行一次 baseline；读 10 条 dev 错误；解释为何词表只能在 train 拟合、为何 Macro-F1 低于 Accuracy。先不要调多个参数。

下一工程阶段：将同一评测接到 Qwen 提示词基线，并准备固定的嵌套 1000/5000 条训练子集；之后才决定 GPU 短测。传统模型与 LoRA 最终还应在相同训练子集上比较，本次全训练集基线只是参照。

执行记录：首次训练后的报告写入因版本字符串被误调用而失败，保留在 `runs/20261005-034858-567904/`，标记为不完整。修正后重新执行成功；以 latest.json 指向的完整运行作为依据。
