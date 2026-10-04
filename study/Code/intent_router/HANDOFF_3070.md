# 在 RTX 3070 机器继续

## 当前做到哪里

截至 2026-10-05，官方 MASSIVE 中文数据、TF-IDF 基线、Qwen 提示词 pilot、LoRA 数据准备、CPU PEFT 检查和 LLaMA-Factory 两步 CPU 训练均完成。正式 1000/5000 条 LoRA 尚未训练。Colab 当前只确认 Linux 内核 CUDA=False，没有得到 GPU 训练结果；现在优先改用自己的 RTX 3070，不再租卡。

结果入口：README.md → PROMPT_BASELINE.md → SFT_PREPARATION.md → FACTORY_VALIDATION.md。`runs/latest*.json` 分别指向各类检查，不能把 smoke adapter 当作正式模型。118 条 pilot 的 Qwen Accuracy 24.58%，同样本 TF-IDF 73.73%。验证集自然分布上的 TF-IDF 为 82.98%，两组分数口径不同。

## 机器准备

仓库：`git@github.com:SatouJotaro/CIE6032-Assignment.git`。本次成果放在 `study/` 下；另一台机器应进入 `study/Code/intent_router`。原电脑继续工作也优先编辑仓库内 study 版本，避免与外部原目录形成两份不同进度。

Git 保存固定训练数据和轻量日志，不保存模型权重、CPU smoke adapter、虚拟环境或缓存。旧日志提及的 adapter 路径是原机器的本地产物；需要时重跑检查生成，不是克隆丢失。学习站的原课件链接依赖另外拷贝课程材料，核心 HTML 仍可离线使用。

1. 克隆仓库，进入包含本文件的 `Code/intent_router` 目录。不要复制原电脑的 `.venv` 或 `.venv-lf`。
2. 运行 `nvidia-smi` 检查驱动、GPU 型号、总/空闲显存。不要只凭“3070”推定当前可用显存。此配置以 batch=1、LoRA rank=8、cutoff=768 为起点，仍须短测。
3. 用 Python 3.11/3.12 建独立虚拟环境，先按 PyTorch 官方安装入口 https://pytorch.org/get-started/locally/ 安装适配驱动的 CUDA 版本。不能照搬旧电脑的 CPU torch 或整个 requirements-factory-local.txt。
4. 在新环境安装 LLaMA-Factory 0.9.4；用 constraints 固定刚装好的 torch/torchvision/torchaudio，避免安装框架时替换 CUDA 构建。安装后 `python -m pip check`，以及 `python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"`。
5. `python download_model.py` 下载 model_lock.json 中固定 revision 的 Qwen 模型。约 1GB 权重不入 Git。运行 `python audit_factory_data.py` 检查本机框架与固定数据。

依赖参考：LLaMA-Factory 0.9.4、Transformers 4.57.1、PEFT 0.17.1、datasets 4.0.0、accelerate 1.11.0。旧电脑 CPU 环境验证记录不能替代新机 CUDA 验证。

## GPU 短测和正式训练

`colab_train.py` 虽然名字含 Colab，内部是普通 GPU 训练脚本，能在自己的机器运行；不需要挂载 Drive，也不调用 Colab API。

```powershell
# 工作目录：Code/intent_router。每次使用未存在的新输出目录。
python colab_train.py --smoke --output runs/3070-smoke-01
```

先查看 `gpu_measurement.json` 的峰值显存、耗时和 `run_config.json` 的实际精度。脚本按 GPU 能力选择 BF16/FP16。10 步短测正常后：

```powershell
python colab_train.py --size 1000 --output runs/3070-train-1000-01
# 1000 条完成并检查日志后，再运行下面这一组
python colab_train.py --size 5000 --output runs/3070-train-5000-01
```

每 25 步保存 checkpoint，保留最近两个。中断后指定同一个输出目录及其中一个完整 checkpoint：

```powershell
python colab_train.py --size 1000 --output runs/3070-train-1000-01 --resume runs/3070-train-1000-01/checkpoint-100
```

正式训练结束仅有 dev loss，还需接 adapter 到统一的生成评测；这部分尚待实现。不要声称已经取得微调准确率，也不要用 test 调参。两组相同 epoch、不同数据量，因此计算量不等；先保持同配方再解释结果。

## 交接给下一次助手

先读本文件和 ../PROJECT_DECISIONS.md；检查 `nvidia-smi`、torch CUDA 与环境版本，跑 10 步短测，再执行 1000 条训练。之后补充 adapter 生成评测，使用固定 prompt、pilot ID 和严格 evaluate.py；必要时在同一新环境重跑未微调模型，隔离环境/精度变化的影响。保留所有日志与配置，并区分助手执行和用户独立掌握。

暂不添加 RAG、Agent、新模型或其他任务。预算目标仍为控制成本；自有 GPU 无租卡费用，但耗电/时间不为零。当前没有任何 GPU 已成功执行的证据。
