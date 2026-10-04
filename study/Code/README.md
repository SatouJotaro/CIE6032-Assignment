# CIE6032 算法实现与实验工作区

这里用于把课件里的算法变成自己能解释、能验证的 Python 实现。这个设计合理；不需要把每个模型从底层重新实现。

现有 `learnbycode/LSA.py`、`LSA-C.py` 保留。它们是 sklearn API 应用示例，不是手写 SVD；学习目标应区分“理解并调用算法”和“独立实现核心步骤”。

## 三层学习法

| 层次 | 建议内容 | 实现与验证方式 | 本机 CPU |
| --- | --- | --- | --- |
| 手写核心 | 线性/逻辑回归、稳定 softmax、交叉熵、MLP 反传、SGD/动量/Adam、单步 RNN、attention、IoU/NMS | Python/NumPy 小输入 → 数值梯度或 PyTorch 对照 | 适合 |
| 框架复现 | 小 CNN、残差块、LSTM、小 Transformer、MNIST GAN/小 DDPM | 用 nn.Module/autograd 组合；自己写训练和评估流程 | 小规模适合，耗时实测 |
| 使用与改造 | 预训练检测器、Stable Diffusion、LLM/LoRA | 理解输入输出与损失，做评测或有限适配 | 推理/训练依模型而定；较大实验再租 GPU |

不要为了“手写”重写 CUDA 卷积、完整自动微分引擎或从零预训练大模型。单步卷积用循环理解即可，实际训练使用经过优化的框架实现。

## 当前入口

- [中文指令项目](intent_router/README.md)：已运行的 MASSIVE 中文数据审计与 CPU 分类基线；包含统一评测脚本和复现步骤。
- [综合项目选型与范围记录](PROJECT_DECISIONS.md)：候选框架、模型、数据来源、取舍、预算与未验证事项；区别于已运行的实验。
- `learnbycode/`：原有 LSA 学习脚本。
- `experiments/optimizer_study.py`：本次新增的真实 CPU 对照实验，比较固定配置下的 SGD、momentum、Adam 与学习率。
- `experiments/README.md`：运行、控制变量、指标解释与面试证据边界。
- `runs/`：实验实际输出，包含配置、逐 epoch CSV、汇总和检查结果；每次运行新建独立目录，不覆盖旧实验。
- `../Learning/labs/`：上次已有的梯度检查、CNN 形状、attention 与 MLP 入门脚本。先用这些验证核心知识，再进入对照实验。

## 后续逐步补充的模块

建议按 `linear_models → mlp_backprop → optimizers → cnn_ops → sequence_attention → tiny_language_model → peft_evaluation` 学习。每个模块实际开始时再创建目录，避免先堆一批空文件夹。

每个已完成算法应有：公式与形状说明、最小输入例子、独立实现、与可信实现对照、失败/边界案例和自己的复述。不要把看到解析后复制出的实现写成“独立完成”。

## 一次值得面试讲的实验

问题 → 基线 → 控制变量 → 可重复日志 → 观察 → 解释 → 局限 → 下一次实验。

“loss 下降了”不足以支持结论。至少同时看：按 optimizer step/epoch 的曲线、达到预先指定目标的时间、验证性能、梯度范数、配置和多随机种子波动。不能跨不同 loss 定义直接比较数值，也不能从一个玩具数据集推出某优化器普遍更好。
