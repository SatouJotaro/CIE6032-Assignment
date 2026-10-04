"""Build a small upload bundle and an executable Colab notebook, without weights."""
import ast
import hashlib
import json
import zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
cells=[]
def md(s): cells.append({'cell_type':'markdown','metadata':{},'source':s.splitlines(True)})
def code(s):
    ast.parse(s)
    cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':s.splitlines(True)})

md('''# CIE6032：Qwen 0.5B LoRA（免费 Colab 优先）
先选择“运行时 → 更改运行时类型 → GPU”，再逐格运行。不要一次运行全部单元格。
本 notebook 在本机做了语法与打包检查，尚未在你的 Colab 账号执行。免费 GPU 不保证分配或持续可用。
先运行 10 步短测，再运行 1000 条，最后才考虑 5000 条。不会自动购买算力。
上传包仅含公开数据的派生训练文件、代码和许可证，不包含原始课程文件、账号密钥或约 1GB 模型。''')
code('''from google.colab import files
from pathlib import Path
import zipfile, hashlib, json, os
uploaded = files.upload()  # 选择本地 intent_router_colab.zip
assert 'intent_router_colab.zip' in uploaded
root = Path('/content/intent_router')
root.mkdir(exist_ok=True)
with zipfile.ZipFile('intent_router_colab.zip') as z:
    for entry in z.infolist():
        target = (root / entry.filename).resolve()
        assert target.is_relative_to(root.resolve()), 'Unsafe archive path'
    z.extractall(root)
os.chdir(root)
for name, sha in json.loads(Path('bundle_manifest.json').read_text()).items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha
print('上传包校验通过')
''')
md('安装固定框架版本，保留 Colab 自带的 CUDA torch/vision/audio。若依赖解析失败，不要强行忽略依赖。安装在子进程中，后续训练也使用新进程。')
code('''import subprocess, sys, importlib.metadata as metadata
constraints = []
for name in ['torch', 'torchvision', 'torchaudio']:
    try: constraints.append(name + '==' + metadata.version(name))
    except metadata.PackageNotFoundError: pass
Path('colab_constraints.txt').write_text('\\n'.join(constraints))
subprocess.run([sys.executable, '-m', 'pip', 'install', '-c', 'colab_constraints.txt', 'llamafactory==0.9.4'], check=True)
print('框架安装完成')
''')
code('''# 下载与本机相同 revision 的模型；不需要付费 API。
subprocess.run([sys.executable, 'download_model.py'], check=True)
subprocess.run([sys.executable, '-c', "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name()); print('BF16:',torch.cuda.is_bf16_supported())"], check=True)
''')
md('挂载 Drive 保存 checkpoint，避免运行时回收后全部丢失。授权只在 Colab 页面进行。每次运行新目录，checkpoint 每 25 步保存一次，保留最近两个；最后一次未保存的进度仍可能丢失。')
code('''from google.colab import drive
from datetime import datetime
drive.mount('/content/drive')
result_root = Path('/content/drive/MyDrive/CIE6032-intent-router')
result_root.mkdir(exist_ok=True)
''')
md('先运行这一格：10 个 optimizer step 的 GPU 短测。完成后查看 gpu_measurement.json 的显存和时间；短测 adapter 不用于正式效果对比。')
code('''smoke_out = result_root / ('smoke-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
subprocess.run([sys.executable, 'colab_train.py', '--smoke', '--output', str(smoke_out)], check=True)
print((smoke_out / 'gpu_measurement.json').read_text())
''')
md('短测正常后，手动运行下一格开始 1000 条正式训练。5000 条是另一轮：稍后将 SIZE 改成 5000，再运行此格。两组使用不同输出目录。当前评估是 teacher-forced dev loss；任务准确率需之后用统一生成评测，不能用 loss 替代。')
code('''SIZE = 1000
train_out = result_root / (f'train-{SIZE}-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
subprocess.run([sys.executable, 'colab_train.py', '--size', str(SIZE), '--output', str(train_out)], check=True)
print((train_out / 'gpu_measurement.json').read_text())
''')
md('如断线：重新连接、上传、安装、下载模型、挂载 Drive，跳过上述新训练格。把恢复格中的两个路径填成原输出目录和其最新完整 checkpoint 目录；参数 SIZE 必须与原运行一致。')
code('''# 恢复示例：取消注释并填写真实路径后执行
# old_out = result_root / 'train-1000-实际时间'
# checkpoint = old_out / 'checkpoint-100'
# subprocess.run([sys.executable, 'colab_train.py', '--size', '1000', '--output', str(old_out), '--resume', str(checkpoint)], check=True)
''')
md('训练结束后可把该次目录打包下载；Drive 中仍保留结果。之后将 adapter 和日志带回本机做同一批样本的生成评测。')
code('''import shutil
last = Path(Path('colab_last_output.txt').read_text())
archive = shutil.make_archive('/content/' + last.name, 'zip', last)
files.download(archive)
''')
notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'colab':{'name':'CIE6032_Qwen_LoRA.ipynb'},'accelerator':'GPU'},'nbformat':4,'nbformat_minor':5}
for i,cell in enumerate(cells): cell['id']=f'cell-{i:02d}'
(ROOT/'CIE6032_Qwen_LoRA.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=2),encoding='utf-8')
names=['COLAB.md','colab_train.py','download_model.py','model_lock.json','configs/lf_train_1000.json','configs/lf_train_5000.json','data/sft_v1/dataset_info.json','data/sft_v1/train_1000.json','data/sft_v1/train_5000.json','data/sft_v1/dev.json','data/sft_v1/manifest.json','data/raw/LICENSE']
manifest={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
with zipfile.ZipFile(ROOT/'intent_router_colab.zip','w',zipfile.ZIP_DEFLATED) as z:
    for name in names: z.write(ROOT/name,name)
    z.writestr('bundle_manifest.json',json.dumps(manifest,indent=2))
print('Notebook syntax checked; bundle bytes:',(ROOT/'intent_router_colab.zip').stat().st_size)

# VS Code sends cell code to the remote kernel; embed only our small public-data
# bundle, never credentials, model weights or unrelated workspace files.
import base64
import copy
vs=copy.deepcopy(notebook)
vs['metadata']['colab']['name']='CIE6032_Qwen_LoRA_VSCode.ipynb'
vs['cells'][0]['source']=['# VS Code + Colab：Qwen LoRA\n','在 VS Code 打开此文件，右上角选择 Colab GPU 内核。先运行环境检查，再逐格执行。无需手动上传 ZIP。\n','本 Notebook 内嵌公开训练数据与脚本；运行解包格会将其发送到 Colab。模型在云端下载。尚未验证实际远程连接。\n']
probe='''import platform, sys, subprocess, shutil
import torch
print('OS:', platform.system(), 'Python:', sys.executable)
print('CUDA:', torch.cuda.is_available())
print('PyTorch:', torch.__version__, 'CUDA build:', torch.version.cuda)
if not torch.cuda.is_available():
    smi = shutil.which('nvidia-smi')
    result = subprocess.run([smi, '-L'], capture_output=True, text=True) if smi else None
    if result and result.returncode == 0 and 'GPU' in result.stdout:
        print(result.stdout)
        raise RuntimeError('已检测到 NVIDIA GPU，但 PyTorch 无法使用。请保留以上版本信息，检查 CUDA 环境；不要反复新建服务器。')
    raise RuntimeError('当前运行时未检测到可用 NVIDIA GPU。这不代表连接失败。在内核选择器中选择 Colab → New Colab Server，并选择 GPU（可用时选 T4），不要继续选择原 CPU 服务器。')
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
'''
payload=base64.b64encode((ROOT/'intent_router_colab.zip').read_bytes()).decode('ascii')
expected=hashlib.sha256((ROOT/'intent_router_colab.zip').read_bytes()).hexdigest()
unpack=f'''from pathlib import Path
import base64, io, zipfile, hashlib, json, os
payload = {payload!r}
raw = base64.b64decode(payload)
assert hashlib.sha256(raw).hexdigest() == {expected!r}
root = Path('/content/intent_router')
root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(raw)) as z:
    for entry in z.infolist():
        assert (root / entry.filename).resolve().is_relative_to(root.resolve())
    z.extractall(root)
os.chdir(root)
for name, sha in json.loads(Path('bundle_manifest.json').read_text()).items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha
print('已解包并验证训练数据与脚本：', root)
'''
vs['cells'][1]['source']=unpack.splitlines(True)
vs['cells'].insert(1,{'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':probe.splitlines(True)})
vs['cells'][-2]['source']=['训练结束后把最近一次输出打包保存在 Drive，用 Drive 网页下载或扩展的远程文件浏览器取回。这里不用网页专用 files.download()。\n']
vs['cells'][-1]['source']='''import shutil
last = Path(Path('colab_last_output.txt').read_text())
archive = shutil.make_archive(str(result_root / last.name), 'zip', last)
print('结果保存在 Google Drive：', archive)
'''.splitlines(True)
for i,cell in enumerate(vs['cells']):
    cell['id']=f'vscode-{i:02d}'
    if cell['cell_type']=='code': ast.parse(''.join(cell['source']))
(ROOT/'CIE6032_Qwen_LoRA_VSCode.ipynb').write_text(json.dumps(vs,ensure_ascii=False,indent=2),encoding='utf-8')
print('VS Code notebook built with embedded, hash-checked bundle')
