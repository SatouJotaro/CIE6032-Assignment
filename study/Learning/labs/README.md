# 配套 CPU 实验

这些是新增的教学练习，不是原作业答案。无需 GPU，也不下载任何数据或预训练权重。

在终端进入 `D:\cuhksz\CIE6032\Learning\labs` 后运行：

```powershell
python gradient_check.py
python tensor_shapes.py
python attention_cpu.py
python mlp_cpu.py --epochs 240 --lr 0.02 --seed 7
```

- `gradient_check.py` 仅依赖 Python 标准库，比较解析梯度与中心差分。
- 其余脚本需要当前 Python 环境已安装 `torch`。课程根目录 `.venv` 与系统 Python 是不同环境，勿默认两者依赖相同。可先执行 `python -c "import torch; print(torch.__version__)"` 检查。
- `mlp_cpu.py` 使用合成双环分类数据；默认把每个 epoch 的真实指标保存到本目录 `results/learning_curve.csv` 和 `results/metrics.json`。再次运行会更新这两个文件；比较实验时用 `--output results/lr-001` 指定不同目录。
- MLP 采用一次随机训练/验证划分，没有独立最终测试集，不可把该玩具任务的分数当成真实数据上的性能结论。

建议操作：先预测输出形状或梯度方向，再运行；每次只改变一个参数；把配置、曲线及解释记入实验模板。

浏览器中的可视化由 JavaScript 计算，不运行这些 Python 文件。点击下载后需在 Python/Jupyter 环境执行。原 Notebook 可能依赖更多包、数据集、网络或 GPU，请先阅读各自说明。
