# 分步推导与 PyTorch

## 01 从独立样本似然到平均交叉熵

约定：监督分类；样本在给定模型下独立，目标是 one-hot 标签。

1. **定义单样本概率**：模型给出 pθ(yᵢ|xᵢ)。所有样本同时出现的条件似然为 ∏ᵢ pθ(yᵢ|xᵢ)。

2. **把乘积转为求和**：取负对数：NLL=−Σᵢ log pθ(yᵢ|xᵢ)。log 单调递增，因此最大化似然等价于最小化 NLL。

3. **用 one-hot 写成统一形式**：−log pθ(yᵢ|xᵢ)=−Σ꜀ yᵢ꜀ log pᵢ꜀。仅正确类别的 y 为 1，其余为 0。

4. **明确归约与泛化边界**：除以 N 得平均交叉熵；最优点在不含其他项时不变，但梯度尺度改变。损失降低是训练目标改善，泛化仍要通过独立数据评价。

PyTorch：数据集与 DataLoader；在训练集拟合预处理；batch 与 epoch 的关系。

```python
from torch.utils.data import TensorDataset, DataLoader
# X_train: [N,D] float；y_train: [N] long
loader = DataLoader(TensorDataset(X_train, y_train),
                    batch_size=64, shuffle=True, num_workers=0)
for xb, yb in loader:
    logits = model(xb)  # [B,C]
    loss = torch.nn.functional.cross_entropy(logits, yb)
```

验证：核对首尾 batch 大小；验证集不 shuffle 也不重新拟合均值方差。Windows 脚本先用 num_workers=0，避免把多进程问题混进模型学习。

## 02 线性层的形状、梯度与自动微分

约定：行样本 X:[B,D]；数学权重 W:[D,H]；Y=XW+b。

1. **微分乘积**：dY=dX·W+X·dW+db。把三个变化来源分开，避免只记形状猜公式。

2. **引入上游梯度**：令 G=∂L/∂Y:[B,H]。标量损失的微分是 dL=Σᵢⱼ GᵢⱼdYᵢⱼ。

3. **按系数收集**：代入后收集 dX、dW、db 的系数：∂L/∂X=GWᵀ，∂L/∂W=XᵀG，∂L/∂b=Σ_batch G。

4. **对应框架约定**：nn.Linear(D,H) 的 weight:[H,D]，所以 weight.grad=GᵀX。不是推导矛盾，而是存储约定转置了。

PyTorch：叶子张量、requires_grad、grad 累加、detach 与 no_grad。

```python
import torch
x = torch.tensor(2.0, requires_grad=True)
y = x * x + 3 * x
y.backward()
print(x.grad)  # 7 = 2x+3
# nn.Linear(D,H).weight 的形状是 [H,D]
# backward 计算梯度；optimizer.step 才更新参数。
```

验证：自己将 y 改为 x³−2x，先预测 x=2 的导数，再与 autograd 对照。不要通过 detach().numpy() 构造需要继续反传的损失。

## 03 真正推出 softmax + CE 的 p−y

约定：单样本；z:[C]；pⱼ=exp(zⱼ)/Σₖexp(zₖ)；Σⱼyⱼ=1。

1. **求 softmax 的局部导数**：商法则给出 ∂pᵢ/∂zⱼ=pᵢ(𝟙[i=j]−pⱼ)。i=j 时为 pᵢ(1−pᵢ)，否则为 −pᵢpⱼ。不能只取对角项。

2. **交叉熵对概率求导**：L=−Σᵢyᵢlog pᵢ，所以 ∂L/∂pᵢ=−yᵢ/pᵢ。

3. **链式法则并化简**：∂L/∂zⱼ=Σᵢ(−yᵢ/pᵢ)·pᵢ(𝟙[i=j]−pⱼ)=−yⱼ+pⱼΣᵢyᵢ=pⱼ−yⱼ。

4. **扩展到 MLP 与 batch**：batch 平均得到 δ₂=(P−Y)/B；W₂ 梯度为 Hᵀδ₂；δ₁=(δ₂W₂ᵀ)⊙σ′(A₁)。偏置沿 batch 求和。若两路共享 W，来自两路的 W 梯度相加。

PyTorch：CrossEntropyLoss 的 logits/long 标签契约；reduction；梯度检查。

```python
import torch
z = torch.tensor([[1., 0., -1.]], requires_grad=True)
target = torch.tensor([1])
loss = torch.nn.functional.cross_entropy(z, target)
loss.backward()
expected = z.detach().softmax(-1) - torch.tensor([[0.,1.,0.]])
assert torch.allclose(z.grad, expected)
print(loss.item(), z.grad)
```

验证：把 batch 复制两遍并采用 mean，观察每个样本梯度缩小一半；改成 sum 再比较。课程目录下已有 gradient_check.py 可检验两层 MLP 的解析梯度。

## 04 卷积为什么能共享参数？反传又如何累加？

约定：先用一维、单通道、stride=1、padding=0 的互相关说明。

1. **写出前向和合法位置**：yₜ=Σⱼ₌₀ᵏ⁻¹ wⱼxₜ₊ⱼ。一个宽 k 的窗口在宽 n 的输入里有 n−k+1 个合法位置。

2. **对同一个核权重求导**：若上游梯度为 gₜ，则 ∂L/∂wⱼ=Σₜgₜxₜ₊ⱼ。wⱼ 在所有位置使用，所以所有位置贡献必须求和。

3. **对一个输入位置求导**：∂L/∂xᵢ=Σₜgₜwᵢ₋ₜ，只累加满足 0≤i−t<k 的项。同一个输入可能被多个窗口覆盖。

4. **推广维度并检验**：二维多通道再对核的两个空间维和输入通道求和。有效核宽 d(k−1)+1；输出宽度为 floor((n+2p−d(k−1)−1)/s)+1。

PyTorch：NCHW、Conv2d 权重 [Cout,Cin/groups,kH,kW]、flatten 与池化。

```python
import torch
from torch import nn
x = torch.randn(2,3,32,32, requires_grad=True)
conv = nn.Conv2d(3,16,3,padding=1)
y = conv(x)
y.square().mean().backward()
print(y.shape, conv.weight.grad.shape, x.grad.shape)
# [2,16,32,32], [16,3,3,3], [2,3,32,32]
```

验证：用一个 5×5 输入手写单通道前向，与 F.conv2d 对照；先不写高性能卷积，重点验证边界和梯度累加。

## 05 残差、归一化与 Dropout 的数学边界

约定：残差分支维度匹配；倒置 Dropout 使用独立 Bernoulli mask；归一化忽略具体 running-stat 更新细节。

1. **残差的直接梯度**：y=x+F(x)，Jacobian 是 I+J_F。上游行梯度 g 变为 g(I+J_F)，其中 g 可以沿恒等路径直接传播。

2. **归一化不是只有除标准差**：x̂=(x−μ)/√(σ²+ε)。训练阶段 μ、σ²依赖输入，因此对输入求导也要包含均值与方差的路径；不能把它们都当常数。

3. **区分 BN 与 LN 的统计轴**：对 [B,D]，BN 通常跨 B 统计每个特征；LN 通常在每个样本内部跨 D 统计。对卷积 BN2d，统计轴还包括空间维。两者不是只换名字。

4. **推出 Dropout 的期望补偿**：m~Bernoulli(1−p)，训练输出 mx/(1−p)。E[mx/(1−p)]=x；p<1 时 Var= x²p/(1−p)。单次采样不等于期望。

PyTorch：train/eval 与梯度开关；BN buffers；Dropout；state_dict。

```python
import torch
from torch import nn
x = torch.ones(8)
drop = nn.Dropout(p=0.5)
drop.train(); print(drop(x))  # 元素为 0 或 2
drop.eval(); print(drop(x))   # 全 1
# eval 不等于 no_grad；BN 的 running_mean/var 是 buffer。
```

验证：在归一化实验里只放大一个样本，比较另一个样本的 BN/LN 输出。再用 Dropout 实验比较单次输出和多次平均。

## 06 从二次函数稳定性到 Adam 偏置修正

约定：稳定性推导只针对确定性标量二次函数；Adam 按常见 EMA 约定。

1. **求更新递推式**：L=aw²/2，∇L=aw。GD 得 wₜ₊₁=(1−ηa)wₜ，于是 wₜ=(1−ηa)ᵗw₀。

2. **得到稳定区间**：趋于 0 需要 |1−ηa|<1，即 0<ηa<2。0<ηa<1 同号衰减；1<ηa<2 交替衰减；边界不衰减。

3. **解释动量的状态**：本页二维实验采用 vₜ=βvₜ₋₁+gₜ，θₜ₊₁=θₜ−ηvₜ。它与带 (1−β) 的 EMA 约定需换算步长后比较。

4. **推出 EMA 零初始化偏差**：若梯度恒为 g，m₀=0，则 mₜ=(1−β₁ᵗ)g；除以 1−β₁ᵗ 恢复 g。Adam 对 v 同样修正，然后用 m̂/(√v̂+ε) 更新。

PyTorch：optimizer 状态、学习率组、scheduler 调用位置、梯度范数与性能记录。

```python
# model、xb、yb 已定义
optimizer.zero_grad(set_to_none=True)
logits = model(xb)
loss = torch.nn.functional.cross_entropy(logits, yb)
loss.backward()
grad_norm = torch.sqrt(sum(p.grad.square().sum()
                          for p in model.parameters() if p.grad is not None))
optimizer.step()
# 比较 epoch、optimizer step 和实际耗时，不能只看 loss 的终值。
```

验证：运行 Code/experiments/optimizer_study.py。比较相同步数与相同时间两个视角，记录目标阈值没有达到的情况，不替它编一个达到时间。

## 07 把 BPTT 的“未来梯度”写出来

约定：列向量约定：hₜ=tanh(Wxₜ+Uhₜ₋₁+b)，zₜ=Vhₜ+c；单序列，每步 CE 求和。

1. **输出层误差**：令 eₜ=pₜ−yₜ。当前时刻输出损失传回隐藏状态的贡献是 Vᵀeₜ。

2. **加入未来贡献**：隐藏状态还影响 hₜ₊₁。若 δₜ=∂L/∂aₜ，aₜ 是 tanh 前输入，则 ∂L/∂hₜ=Vᵀeₜ+Uᵀδₜ₊₁。末端未来项为 0。

3. **通过激活局部导数**：δₜ=(Vᵀeₜ+Uᵀδₜ₊₁)⊙(1−hₜ²)。必须逆时间计算，不能只把当前输出梯度往回传一次。

4. **累加共享参数贡献**：∂L/∂U=Σₜδₜhₜ₋₁ᵀ，∂L/∂W=Σₜδₜxₜᵀ，∂L/∂b=Σₜδₜ。这里与前面行样本公式约定不同，已明确标注。

PyTorch：Embedding、batch_first、padding mask、hidden.detach 与截断 BPTT。

```python
import torch
from torch import nn
rnn = nn.GRU(input_size=8, hidden_size=16, batch_first=True)
x = torch.randn(2,5,8)
output, h = rnn(x)
print(output.shape, h.shape)  # [2,5,16], [1,2,16]
# 连续序列分块训练时可在分块边界 h=h.detach() 截断图。
# 不相关样本不应无意继承前一序列隐藏状态。
```

验证：先在两步手写 RNN 上对 U 做数值梯度检查；再进入课程的 rnn_step_backward/rnn_backward。

## 08 Attention 前向与四步反向

约定：单 head、单 batch；Q:[Tq,d]，K:[Tk,d]，V:[Tk,dv]；mask 固定。

1. **明确四个中间量**：S=QKᵀ/√d+M；A=softmax_row(S)；O=AV。每个 query 行的 A 对可见 key 和为 1。

2. **穿过加权和**：给定 G=∂L/∂O，则 ∂L/∂V=AᵀG，∂L/∂A=GVᵀ。

3. **穿过逐行 softmax**：每行 gSⱼ=Aⱼ(gAⱼ−ΣₖAₖgAₖ)。这利用 softmax Jacobian，避免显式构造完整矩阵。被 mask 的位置视为固定不可访问位置。

4. **穿过点积投影**：∂L/∂Q=gS·K/√d；∂L/∂K=gSᵀ·Q/√d。若 Q=XWq，再用线性层反传求 Wq 和 X；多头按相应维度组织。

PyTorch：reshape/transpose/contiguous、head 维度、bool/additive mask、数值稳定性。

```python
import torch, math
q = torch.randn(2,4,8, requires_grad=True)
k = torch.randn(2,4,8, requires_grad=True)
v = torch.randn(2,4,6, requires_grad=True)
scores = q @ k.transpose(-2,-1) / math.sqrt(8)
future = torch.triu(torch.ones(4,4,dtype=torch.bool),1)
a = scores.masked_fill(future, float("-inf")).softmax(-1)
out = a @ v
out.square().mean().backward()
```

验证：把上面的四步解析反向与 autograd 对照。注意 API 的 bool mask 语义可能不同，换接口时查该接口文档，不能凭变量名猜。

## 09 Patch 数为什么让 attention 平方增长？

约定：正方形图像、整除的非重叠 patch；密集全局注意力。

1. **计算 token 数**：N=(H/P)(W/P)，含 class token 时 T=N+1。

2. **计算投影维度**：每个 RGB patch 有 3P² 个输入值，线性投影参数为 (3P²)d+d，不按 patch 个数另建一份。

3. **区分参数与激活**：QKᵀ 的每个 head 有 T² 个分数。patch 边长减半时 N 约变为四倍，分数矩阵约十六倍。

4. **解释近似条件**：class token 和其他中间激活让比例不是精确十六倍；窗口 attention 的复杂度关系又不同。

PyTorch：Conv2d patch embedding、flatten/transpose、位置参数、预训练权重形状。

```python
import torch
from torch import nn
embed = nn.Conv2d(3,64,kernel_size=16,stride=16)
x = torch.randn(2,3,224,224)
tokens = embed(x).flatten(2).transpose(1,2)
print(tokens.shape)  # [2,196,64]
# transpose 后不要假定张量连续；不确定时用 reshape。
```

验证：运行 tensor_shapes.py，将 P=16 改成 8，预测 token 与注意力矩阵大小；不需要训练完整 ViT。

## 10 BCE logits 梯度与 GAN 两次更新

约定：D 输出 logit a，p=sigmoid(a)；本节说明常见 non-saturating G loss。

1. **稳定 BCE 形式**：对 y∈{0,1}，L=softplus(a)−ya。它等价于 −ylogσ(a)−(1−y)log(1−σ(a))。

2. **对 logit 求导**：softplus 导数是 σ(a)，所以 ∂L/∂a=σ(a)−y。D 对真实样本取 y=1、假样本取 y=0。

3. **更新 D 时断开 G**：fake=G(z).detach()，D_loss 两项反传只需更新 D。避免无意为 G 累加梯度。

4. **更新 G 时保留输入路径**：G_loss=BCEWithLogits(D(G(z)),1)。即使冻结 D 的参数，也不能 no_grad 包住 D 的整个前向，因为 G 仍要经 D 对输入的导数接收梯度。

PyTorch：detach vs requires_grad_(False)；两个 optimizer；BCEWithLogitsLoss。

```python
# G、D 和 opt_g 已定义；此片段为 G step
for p in D.parameters(): p.requires_grad_(False)
opt_g.zero_grad(set_to_none=True)
fake = G(z)
score = D(fake)
loss_g = torch.nn.functional.binary_cross_entropy_with_logits(
    score, torch.ones_like(score))
loss_g.backward(); opt_g.step()
for p in D.parameters(): p.requires_grad_(True)
```

验证：临时检查 G 参数 .grad：正确 G step 应有有限梯度。把 fake detach 后解释为什么路径断开，再恢复。

## 11 DDPM 直接加噪公式如何得到？

约定：前向噪声各步独立、标准高斯；αₜ=1−βₜ。只推导前向边际，不声称完整推导 ELBO。

1. **写一步转移**：xₜ=√αₜxₜ₋₁+√(1−αₜ)εₜ。

2. **展开两步**：代入 xₜ₋₁，信号系数为 √(αₜαₜ₋₁)，两项独立高斯噪声的方差是 αₜ(1−αₜ₋₁)+(1−αₜ)=1−αₜαₜ₋₁。

3. **递推到任意时间**：重复得到 ᾱₜ=∏ₛ₌₁ᵗαₛ，xₜ=√ᾱₜx₀+√(1−ᾱₜ)ε，其中合成后的 ε 仍是标准高斯。

4. **区分边际与条件逆过程**：可以直接采样任意 t 的边际带噪图来训练 εθ。反向采样还需要选定参数化和更新公式；知道此式不等于已推导整个 DDPM。

PyTorch：时间步索引、广播到 BCHW、randn_like、MSE 与条件输入。

```python
# x0: [B,C,H,W]，alpha_bar: [T]，model 预测噪声
t = torch.randint(len(alpha_bar), (x0.shape[0],), device=x0.device)
a = alpha_bar[t].view(-1,1,1,1)
eps = torch.randn_like(x0)
xt = a.sqrt()*x0 + (1-a).sqrt()*eps
pred = model(xt, t)
loss = torch.nn.functional.mse_loss(pred, eps)
```

验证：在固定 x₀ 与多个独立噪声样本上测经验均值/方差，与公式比较；潜空间版还需要理解编码器尺度，不直接套像素参数。

## 12 NMS 是一个有顺序的贪心算法

约定：同一类别、连续坐标、框面积为正；标准贪心 NMS。

1. **先按 score 降序排列**：最先处理最高分框，这一步体现算法的贪心选择。

2. **选一个框并计算剩余重叠**：保留当前最高分框 b，对所有尚未处理框算 IoU(b,·)。

3. **抑制超过阈值的候选**：本实验用 IoU>threshold 抑制。等于阈值时保留；说明这个比较符号才能精确重现边界结果。

4. **循环到没有剩余候选**：再选剩余最高分框，直到候选为空。同类拥挤目标可能被误抑制；Soft-NMS、集合预测采用不同机制。

PyTorch：坐标约定、向量化 IoU、候选过滤、设备与 dtype 一致。

```python
# boxes: [N,4]，连续 xyxy；a、b 均为单框
lt = torch.maximum(a[:2], b[:2])
rb = torch.minimum(a[2:], b[2:])
inter = (rb-lt).clamp_min(0).prod()
area_a = (a[2:]-a[:2]).clamp_min(0).prod()
area_b = (b[2:]-b[:2]).clamp_min(0).prod()
iou = inter / (area_a+area_b-inter).clamp_min(1e-8)
```

验证：在新增 NMS 实验里改变阈值，解释哪个框因为谁被抑制；然后用 4 个框手写循环对照。

## 13 从计数关系推出 Dice 与 IoU

约定：同一非空二值预测/真值 mask，不含额外平滑项。

1. **写两个定义**：IoU=TP/(TP+FP+FN)，Dice=2TP/(2TP+FP+FN)。

2. **引入共同分母**：设 U=TP+FP+FN，则 Dice=2TP/(U+TP)。

3. **同除 U**：Dice=2(TP/U)/(1+TP/U)=2IoU/(1+IoU)。空并集时该推导不能直接用于 0/0。

4. **区别评价值与可微代理**：硬 mask 指标用于评价；训练中的 soft Dice 可能加平滑项并使用概率，其梯度与这里的计数公式不是同一个对象。

PyTorch：argmax 只在评价路径；标签 long；ignore_index；插值恢复空间尺寸。

```python
# logits:[B,C,H,W]，target:[B,H,W] long
loss = torch.nn.functional.cross_entropy(logits, target, ignore_index=255)
with torch.no_grad():
    pred = logits.argmax(dim=1)
    valid = target != 255
    # 按类统计 TP/FP/FN；明确空类的处理约定。
```

验证：构造一个手算的 3×3 mask，核对 FP/FN 方向；再构造全空类，明确约定而不是让 NaN 混入平均。

## 14 Gram 矩阵的形状与输入优化

约定：特征 F:[C,S]，S=HW；目标 A:[C,C] 对称；这里只给未归一化的 L=||FFᵀ−A||²_F。

1. **Gram 的每个元素**：Gᵢⱼ=ΣₛFᵢₛFⱼₛ，统计通道 i,j 的空间共激活，G:[C,C]。

2. **对矩阵平方差求导**：令 E=FFᵀ−A，dL=2 tr(EᵀdG)。

3. **代入乘积微分**：dG=dF·Fᵀ+F·dFᵀ；A 对称时 E 也对称，收集 dF 得 ∂L/∂F=4EF。

4. **传播到图像并说明尺度**：再沿特征提取网络传回输入 x。若 Gram 或 loss 除以 C、S 等常数，梯度要带上对应因子；不同教程约定不能直接混用权重。

PyTorch：固定网络参数、输入 requires_grad、优化输入 tensor；detach 后的展示副本。

```python
# feature_net 已加载；固定其参数但保留输入计算图
for p in feature_net.parameters(): p.requires_grad_(False)
x = content_image.clone().requires_grad_(True)
optimizer = torch.optim.Adam([x], lr=0.01)
# 每步计算特征/损失 → backward → step
# 可视化时使用 x.detach()，而不是在损失路径 detach。
```

验证：用一个 2×3 的 F，对 4(FFᵀ−A)F 做 autograd 对照，再考虑归一化后的形式。

## 15 把“收敛更快”拆成可检验的指标

约定：比较同一任务、相同 loss 定义与归约；评价阈值在看结果之前确定。

1. **先定义横轴**：epoch 是遍历数据次数；step 是参数更新次数；wall time 是真实时间。batch 变大后，同 epoch 的 step 数会变少。

2. **定义达到阈值的时间**：选择验证指标阈值，例如 accuracy≥0.90。记录首次达到的 step/time，并同时检查是否稳定。没有达到就记未达到。

3. **控制初始条件**：固定划分、初始参数和 batch 顺序比较主要变量；再多种子重复。记录初始化/排序变化与数据划分变化的区别。

4. **限制结论的范围**：更快降低训练 loss 不必然提高泛化；不同优化器需要公平调参预算。保存配置、日志和环境，避免只挑某次最好结果。

PyTorch：checkpoint 中的 model/optimizer/scheduler/scaler 状态；随机状态与采样器进度；计时与 profiler。

```python
# 用于恢复训练的 checkpoint 至少需要模型和优化器状态
torch.save({"model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch}, "checkpoint.pt")
ckpt = torch.load("checkpoint.pt", weights_only=True, map_location="cpu")
model.load_state_dict(ckpt["model"])
optimizer.load_state_dict(ckpt["optimizer"])
# 完整续训还需按训练设置恢复随机、scheduler、scaler、采样进度等。
```

验证：本次 optimizer_study.py 已提供固定 CPU 下一 batch 的恢复一致性检查。自己添加 Dropout 后，考虑还缺哪些随机状态。
