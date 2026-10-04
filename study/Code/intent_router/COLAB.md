# Colab 运行入口

## 优先使用 VS Code（已安装官方 google.colab 插件）

直接打开 `CIE6032_Qwen_LoRA_VSCode.ipynb`，右上角选择 Colab 内核，运行第一个代码格查看 GPU 名称和显存。本机只确认插件目录存在（含 0.9.7），尚未确认活动连接或执行远程代码。

这一版内嵌约 364 KB 的公开数据/脚本包，不需手动上传 Notebook 或 ZIP 到网页。代码单元运行时仍会将内嵌包传到远程内核；远程机器不能直接读取本机 D 盘。已替换网页专用 files.upload/download 调用；Drive 挂载由官方扩展支持，授权在 VS Code 完成。若已有连接，直接选用，不必重新创建。

运行顺序：GPU 检查 → 自动解包 → 安装 → 下载模型 → Drive → 10 步短测。不要直接 Run All。官方说明：https://github.com/googlecolab/colab-vscode/wiki/Known-Issues-and-Workarounds 。

## 浏览器入口（备用）

2026-10-05：优先尝试免费 Colab，再考虑原定 100–200 元租卡预算。当前未在 Colab 执行、未登录账号、未付费。

1. 打开 https://colab.research.google.com/ ，上传 `CIE6032_Qwen_LoRA.ipynb`。
2. 选择 GPU 运行时，逐格执行；上传文件时选择 `intent_router_colab.zip`（约 362 KiB，不含模型）。
3. 安装固定 LLaMA-Factory 0.9.4，保留 Colab CUDA torch；下载固定 revision 的 Qwen 0.5B，挂载自己的 Drive。
4. 先跑 10 步短测查看 `gpu_measurement.json`。通过后手动运行 1000 条，最后再运行 5000 条。
5. 结果和每 25 步 checkpoint 保存到 Drive；可按 Notebook 中说明恢复，保留最近两个 checkpoint。结果下载格会打包最近一次输出。

精度通过 GPU 能力检测：支持 BF16 则启用 BF16，否则 FP16。batch=1、累积=8、cutoff=768、LoRA rank=8。0.5B、约 1GB 半精度底座权重只是部分显存，还需要激活、梯度和临时张量；16GB GPU 适合先尝试的判断不是实测保证。短测失败则先查错误，不自动升级付费套餐。

官方限制：https://research.google.com/colaboratory/faq.html 。免费 GPU 型号、配额与时长不保证，虚拟机可能回收。checkpoint 降低进度损失，不能保证永不断线；不使用保活或绕配额手段。

Notebook 已本地逐格 Python AST 语法检查，上传 ZIP 含 SHA-256 校验。Colab 依赖兼容、GPU显存、速度、断线恢复仍待实际验证；本机 LLaMA-Factory 检查不能替代云端验证。正式配置当前只计算 dev loss，生成准确率仍需后续统一评测。

上传包数据来自 Amazon MASSIVE 1.1 zh-CN，CC BY 4.0；源：https://github.com/alexa/massive 。派生处理为官方划分内抽样与 Alpaca 格式转换，许可证随包保留。模型来源：https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct 。
