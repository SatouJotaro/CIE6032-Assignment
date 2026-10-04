# 微调数据与本机检查

2026-10-05：数据准备和两步 CPU LoRA 检查已完成。正式 1000/5000 条微调尚未运行，没有租卡或付费。

## 固定数据

`data/sft_v1/` 已生成训练 1000、训练 5000、官方 dev 2033 条 Alpaca 格式 JSON，以及 LLaMA-Factory 的 `dataset_info.json`。

1000 条是 5000 条的严格子集。每类先保证 1 条，再按原训练集类别规模近似比例分配，类内以 SHA256(sft42:id) 固定排序，无放回抽样。两个训练子集都覆盖 60 类，但不是类别均衡抽样；极少数类受最低 1 条的约束，不能完全维持原比例。所有 ID、各类数量、文件哈希均在 `manifest.json` 中。该方法没有读取 test，train/dev ID 不相交。

字面重复样本仍遵循官方划分，未在此阶段另行清洗；前一阶段发现的文本重叠风险依然存在。后续评测需补充排除训练已见文本的结果，不能只凭 ID 不相交宣称无泄漏。

训练中的 system 提示词直接复用已完成 Qwen pilot 的 prompt.txt，并核对哈希。用户输入是原始中文指令，答案是唯一 intent JSON；只在答案和结束标记上计算 loss，提示词 token 为 -100。分词逐条检查前缀边界、答案可还原、至少有一个监督 token、没有截断。

| 数据 | 数量 | 最长 token 数 | 总 token 数 | 答案监督 token 数 |
| --- | ---: | ---: | ---: | ---: |
| train_1000 | 1000 | 590 | 575892 | 7567 |
| train_5000 | 5000 | 598 | 2880618 | 37880 |
| dev | 2033 | 598 | 1170866 | 15346 |

长度包含模板和 EOS；统计按本机官方 tokenizer 与本脚本的答案边界，不等同于尚未运行的 LLaMA-Factory 内部分词统计。cutoff_len 设为 768，当前检查无截断。

## 两步 CPU 检查已实际通过

使用现有系统 Python + PEFT 0.19.1，在第一条训练样本上执行两次 AdamW 更新，LoRA rank=8、alpha=16、dropout=0，目标为所有 q_proj/v_proj，学习率 1e-4。检查以下条件：

- loss、梯度有限，梯度非零；所有可训练参数都是 LoRA 参数。
- 更新前后所有冻结权重的 SHA-256 一致，adapter 确有更新。
- adapter 保存后卸载并重新载入，相同样本 loss 差小于 1e-6。

记录：`runs/lora-smoke-20261005-040844/report.json`，适配器保存在同目录 `adapter/`。两步训练 loss 约 0.10057 → 0.05591；这只是同一训练样本上的流程检查，不是验证集进步，不能作为微调有效的证据。不要用 smoke adapter 代替正式实验的模型。

这是独立的 PyTorch/PEFT 检查，**不是 LLaMA-Factory 已通过运行验证**。没有安装 LLaMA-Factory，正式训练环境尚未锁定，GPU 显存/吞吐未知。

## 正式配置草案

`configs/lf_train_1000.json` 与 `configs/lf_train_5000.json` 除数据名和输出路径外保持一致：Qwen 模板、LoRA q_proj/v_proj、rank 8、alpha 16、lr 1e-4、1 epoch、micro batch 1、梯度累积 8、seed 42、cosine schedule、warmup 10%。不上传日志或模型。

同 epoch 下两组约为 125 与 625 个优化器 step，因此同时改变了数据量和计算量。这是“相同训练配方下扩大数据”的比较，不是等算力比较，warmup 的绝对步数也随之改变。

配置是 JSON（框架支持的配置形式），从 `Code/intent_router` 目录启动，路径相对该目录。设置了 bf16，需要支持它的 GPU；不能直接拿当前 CPU 环境运行这些正式配置。参考了官方 [训练示例](https://github.com/hiyouga/LlamaFactory/blob/main/examples/train_lora/qwen3_lora_sft.yaml)、[数据格式](https://github.com/hiyouga/LlamaFactory/blob/main/data/README.md) 和 [v0.9.4 模板源码](https://github.com/hiyouga/LlamaFactory/blob/v0.9.4/src/llamafactory/data/template.py)，未声称已固定或验证框架版本。

下一步在隔离环境中固定 LLaMA-Factory 版本、解析配置并检查实际 loss mask，与本机 tokenizer 对照，再短测训练成本。成功加载配置并不等于 GPU 上已验证。

## 本机复现

从课程根目录使用系统 Python，现有依赖已具备：

```powershell
$env:HF_HUB_OFFLINE='1'
python Code/intent_router/prepare_sft.py
python Code/intent_router/lora_smoke.py
```

第一条会重建相同数据并拒绝覆盖内容不同的冻结数据；第二条新建独立运行目录。不要手工修改冻结文件，若改提示词/筛选规则应创建新的数据版本。

本阶段学习任务：看一条数据，说明 input_ids 与 labels 的区别，解释为什么提示词需要屏蔽；看报告，说明为什么两步 loss 下降不能证明泛化提升。暂时不用增加新的模型或任务。
