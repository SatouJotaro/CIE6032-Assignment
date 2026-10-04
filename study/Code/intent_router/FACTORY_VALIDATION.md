# LLaMA-Factory 本机验证

2026-10-05：已通过实际框架的数据处理和两步 CPU 训练。没有进行正式 1000/5000 条训练、GPU 测量或付费。

## 环境

项目环境：`.venv-lf`。使用 `--system-site-packages` 复用部分系统依赖，因此它不是完全独立的可移植环境；兼容旧版本的包安装在项目虚拟环境中，不修改系统包。运行后检查系统 Transformers 5.5.4、PEFT 0.19.1、torch 2.11.0+cpu 保持原版本。

实际框架环境：LLaMA-Factory 0.9.4、Transformers 4.57.1、PEFT 0.17.1、datasets 4.0.0、accelerate 1.11.0、torch 2.14.1、tokenizers 0.22.2、huggingface-hub 0.36.2。`pip check` 通过。完整已安装版本快照为 `requirements-factory-local.txt`，它是本机环境记录，不保证可直接用于 Linux/CUDA。正式 GPU 环境需匹配 CUDA 构建后重新验证。

安装中遇到临时目录权限限制，通过环境授权完成安装；没有禁用版本检查或修改上游代码。

## 数据与 loss mask

`audit_factory_data.py` 实际调用 LLaMA-Factory 的配置解析、tokenizer、模板、数据加载和预处理。

| 实际加载的数据 | 条数 | 最长 token 数 |
| --- | ---: | ---: |
| train_1000 | 1000 | 591 |
| train_5000 | 5000 | 599 |
| dev（两份配置分别检查） | 2033 | 599 |

每条均检查提示词前缀、被屏蔽的 labels、答案 token 与先前数据准备的一致性，无截断。框架 Qwen 模板在 EOS 后多监督一个换行 token；这是已确认的唯一尾部差异。因此框架 loss 与前一轮独立 PEFT smoke 的 loss 不能直接比较。

完整结果：`runs/factory-data-audit.json`；执行日志：`runs/factory-data-audit.log`。没有读取 test。

## 两步真实训练

运行目录：`runs/factory-check-20261005-041933/`，入口记录：`runs/latest_factory_check.json`。

从正式 1000 条配置派生短测配置：4 条训练数据、max_steps=2、batch=1、累积=1、不运行 dev 评价、不保存中间 checkpoint；保存最终 adapter。核心 LoRA 为 q_proj/v_proj、rank 8、alpha 16，540672 个可训练参数。

框架实际完成 2 步优化，报告训练耗时 11.4144 秒，训练均值 loss 0.039485。两步使用不同样本，不把数值高低解释为收敛趋势。此 adapter 仅用于集成检查，不是项目成果模型。

注意实际运行行为：即使训练参数 bf16=false，框架仍从模型配置自动选择了 bfloat16 底座，并将可训练参数转为 float32；框架还自动启用了 gradient checkpointing。`gradient_checkpointing=false` 不等于该版本模型加载阶段禁用了 checkpointing（另有 `disable_gradient_checkpointing` 参数）。本次检查记录的是这套实际行为，不宣称是纯 float32，也不拿它与先前独立 PEFT 时长作性能比较。

验证文件 `verification.json` 保存版本、配置哈希、脚本哈希与训练结果。`adapter_model.safetensors` 已生成。前一轮独立 PEFT 已验证保存重载，本轮只验证框架训练和保存链路。

## 复现

从课程根目录运行：

```powershell
Code/intent_router/.venv-lf/Scripts/python.exe Code/intent_router/verify_factory.py
Code/intent_router/.venv-lf/Scripts/python.exe Code/intent_router/audit_factory_data.py
```

`verify_factory.py` 每次生成独立目录；数据审计报告更新到固定路径。离线读取本地模型，缓存放在项目目录。

## 下一步与停止条件

本机数据、模板、梯度更新和框架保存链路已经验证。下一阶段才是正式 1000/5000 条对照。两步短测不够估算 GPU 预算，也没有证明模型效果改善。

如继续使用 CPU，按此次短测每样本约 5.7 秒粗算，1000 条仅训练约 1.6 小时、5000 条约 8 小时，另加评测与保存；此为极粗外推，不是承诺。GPU 短测应先测几十步的实际吞吐、峰值显存、评测开销及平台单价，遵守首轮 20 元短测、总额 100–200 元上限，不自动开通服务。
