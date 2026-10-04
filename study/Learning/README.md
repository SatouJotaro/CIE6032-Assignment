# Deep Learning Studio

直接打开 [index.html](index.html)，不需要启动服务器。首页有推荐路线和 8 题起点自测。

## 内容

- 15 个中文学习模块：直觉、公式、形状、例题、误区、实践与完成标准。
- 45 道自测题，提交后显示解析，自动维护当前与历史错题。
- 17 个可调参数的浏览器实验：原有 8 个，加上激活导数、链式法则、二维动量、Dropout、BN/LN、时间梯度、NMS、真实训练曲线和 LoRA 参数量。
- 15 组分步推导与 PyTorch 片段，分别标明假设、公式约定和验证方法；可从侧栏“推导与 PyTorch”集中阅读。
- 全部主课件的逐页 PDF 入口；明确区分本站已展开主线与尚待深读的拓展，不假称覆盖全部课件。
- 本机实际执行的 5 配置 × 3 种子 CPU 优化器对照，及固定下一 batch 的 checkpoint 恢复检查。页面嵌入真实日志，不在浏览器后台训练。
- 通用算法基础 + LLM 应用/微调路线；CPU 优先，首轮 GPU 预算上限 100–200 元，未租卡或付费。
- 每课链接到课程原始 PDF、Notebook 或源码。
- 4 个 CPU Python 练习以及实验记录模板，见 [labs/README.md](labs/README.md)。

## 保存与使用

学习记录存在当前浏览器的 localStorage，包含答题、完成状态和笔记。不同浏览器、隐私模式或文件位置可能使用不同存储空间。切换环境前，可在“学习笔记”页导出 JSON，再导入恢复。浏览器拒绝保存时，页面会提示导出备份。

完成模块需要三题当前均答对，并自行确认实践与复述标准；此状态记录学习活动，不等同于独立能力评定。

网页不执行 Python，也不自动安装依赖或下载模型。请复制原 Notebook 后练习；点击 IPYNB 链接可能触发下载，需要用 Jupyter 或 Colab 打开。所有课程资料的相对链接依赖上级目录结构，请保留 `Learning` 与 `Lectures`、`Tutorials`、`Assignments` 等的相对位置。

## 内容边界

以系统梳理和实践迁移为目标，不是全部课件的逐页转录。高级模型训练细节、RAFLCC 项目算法、大型预训练模型适配仍需回到原材料深入学习。网页中的加噪示意不是图像生成模型；attention 使用可解释的玩具向量。

## 维护

`index.html` 是完整离线文件，内嵌 CSS、JavaScript 和课程内容，不依赖 CDN。

- 内容源：`course.js`
- 推导与代码片段：`depth.js`；新增实验与扩展页面：`extension.js`
- 页面与交互：`shell.html`、`style.css`、`app.js`
- 构建：`node Learning/build.cjs`（从课程根目录执行）
- 同步生成：`tutorial.md`（可离线阅读的课程文本）、`practice.md`（练习题）
- 同步生成：`derivations.md`（逐步推导与代码）；真实实验读取 `../Code/runs/latest.json` 指向的报告。
- `source_index.json` 是本次梳理的课件文本索引，不是网页运行依赖。

修改内容源后应重新构建，避免与 HTML 不一致。
