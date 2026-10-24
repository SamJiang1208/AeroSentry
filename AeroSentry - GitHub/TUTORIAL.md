# 完整教程 · 训练与使用

AeroSentry 天衡 · HackSHSID 2026 Track 3 · Sam Jiang 9(10)

这份教程覆盖三件事：**怎么用**、**怎么接模型**、**怎么训练**。每一步都标了预期输出，出问题时按第 5 节的表查。

---

## 1. 两种用法

| 方式 | 需要什么 | 适合 |
|---|---|---|
| **A. 直接打开** | 一个浏览器 | 演示、评审、离线查看 |
| **B. 接上模型** | 浏览器 + Ollama | 想看到「对话」而不只是「报告」 |

方式 A 已经完整可用：推理、评级、行动、三份草案全部由内置引擎算出来，不需要任何后端。
方式 B 只是把结构化结论换成自然语言表达，结论本身不变。

---

## 2. 快速开始（约 30 秒）

1. 双击 `index.html`，浏览器打开。
2. 左侧点任意事件（EVT001–EVT005），中间立刻出现风险等级、判断依据、行动清单。
3. 顶部四个标签：**判断依据链 / 事实·假设·未知 / 行动与升级 / 与模型对话**。
4. 右侧切换 **运营 / 机场 / 旅客** 三份草案，可复制。
5. 右栏「人工反馈」选 **拒绝** → 填原因 → **提交反馈并重算**，看结论怎么变。

> 不需要联网。页面里没有任何外部请求，断网也能跑。

---

## 3. 接上 Qwen 模型（可选）

### 3.1 安装 Ollama

Windows 直接去 https://ollama.com/download 下安装包，装完任务栏会有图标。

验证：

```bat
ollama --version
```

### 3.2 准备模型

**情况一：已经有合并好的模型**（`models\Qwen3.5-9B-AeroSentry`）

```bat
ollama create aerosentry -f model\Modelfile.aerosentry
```

**情况二：只有基座，没训练过**

直接拉基座也能跑，只是它没学过本项目的输出规范：

```bat
ollama pull qwen3.5:9b
ollama create aerosentry -f model\Modelfile.aerosentry
```

> `Modelfile.aerosentry` 里写的是 `FROM ./models/aerosentry-q4.gguf`。
> 如果你用 `ollama pull` 的模型，把第一行改成 `FROM qwen3.5:9b` 再执行。

### 3.3 启动并验证

```bat
ollama serve
```

另开一个窗口：

```bat
curl http://127.0.0.1:11434/api/tags
```

能看到 `aerosentry` 就对了。

### 3.4 在页面里连接

打开 `index.html` → 顶栏 **连接模型** → 粘贴：

```
http://127.0.0.1:11434/v1/chat/completions
```

模型名填 `aerosentry`。连上后顶栏徽章变成 `aerosentry · 已连接`，对话标签页的回答会走模型；
端点关掉后自动回落到引擎，**结论一模一样**。

---

## 4. 训练完整流程

### 4.1 硬件

| 显存 | 能否训 9B | 用哪份配置 | 预计 |
|---|---|---|---|
| 8 GB（如 RTX 5070 Laptop） | **不能**（见 5.1） | — | — |
| 10–16 GB | 能 | `qwen3_5_9b_qlora_sft.yaml` | 60–120 分钟 |
| 24 GB+ | 能 | `qwen3_5_9b_lora_sft.yaml` | 30–60 分钟 |
| 8 GB，改训 4B | 能 | 改 `model_name_or_path` 即可 | 60–90 分钟 |

### 4.2 步骤

**① 环境**（只跑一次）

```bat
cd source\training
train.bat
```

它会依次做：建 `.venv` → 装 CUDA 12.8 版 PyTorch → 装 LLaMA-Factory 0.9.5 →
下载 Qwen3.5-9B 到 `models\Qwen3.5-9B`（约 19 GB）→ 检测显存选配置 → 开始训练。

> 下载慢的话，先把 `model\Qwen3.5-9B` 整个拷到 `models\Qwen3.5-9B`，脚本会跳过下载。

**② 数据**（已生成，不用重跑）

```bat
python build_dataset.py
```

预期输出：

```
samples: 320  train: 301  val: 19
by kind: {'assess': 202, 'qa': 90, 'safety': 12, 'dialog': 6, 'definition': 6, 'feedback': 4}
```

数据全部由引擎生成，所以模型学到的引用规范与安全边界和引擎完全一致。

**③ 训练**

`train.bat` 会自动选配置并写 `configs\_resolved.yaml`。想手动指定：

```bat
llamafactory-cli train configs\qwen3_5_9b_qlora_8gb.yaml
```

正常日志长这样：

```
***** Running training *****
  Num examples = 301
  Num Epochs = 3
  Total train batch size = 16
  Total optimization steps = 57
{'loss': 2.13, 'learning_rate': 9.8e-05, 'epoch': 0.05}
```

**57 步是正常的**（301 条 × 3 轮 ÷ 累积 16）。每 20 步存一次，产出在
`models\qwen3.5-9b-aerosentry-lora\`。

**④ 合并**

```bat
merge_and_export.bat
```

产出 `models\Qwen3.5-9B-AeroSentry\`（完整 HF 模型，约 19 GB）。

**⑤ 转 GGUF（给 Ollama / llama.cpp 用，可选）**

```bat
python llama.cpp\convert_hf_to_gguf.py models\Qwen3.5-9B-AeroSentry --outfile models\aerosentry-f16.gguf --outtype f16
llama-quantize models\aerosentry-f16.gguf models\aerosentry-q4.gguf Q4_K_M
```

> Qwen3.5 自带 MTP 模块，llama.cpp 不认，转换时必须用 legacy 格式——
> `merge_lora.yaml` 里的 `export_legacy_format: true` 就是干这个的。

**⑥ 评测**

```bat
run_eval.bat
```

它会用 15 项官方测试问一遍模型，并对比 5 个事件的评级与引擎是否一致，
结果写到 `reports\model_eval.json`。

---

## 5. 常见问题

### 5.1 `CUDA out of memory` 或直接闪退（8 GB 显卡训 9B）

**不是配置问题，是物理限制。** 9B 做 4-bit 量化时，加载过程要同时容纳一个 5 GB 的
bf16 分片和已量化的权重，峰值约 10 GB。8 GB 卡会在权重加载到 1% 时直接段错误，
日志里连报错都没有。

三条出路：

- 换 **Qwen3.5-4B**：把配置里的 `model_name_or_path` 改成 `./models/Qwen3.5-4B`，其余不变；
- 租一张 **24 GB** 卡（AutoDL / 恒源云，几块钱一小时），配置会自动走 10–22 GB 档；
- 用云端的免费 GPU（Colab / Kaggle），把 `source/training/` 传上去跑。

### 5.2 `no kernel image is available for execution on the device`

RTX 50 系（Blackwell, sm_120）跑了 cu121 版 torch。重装：

```bat
training\fix_env.bat
```

它会卸掉旧 torch、装 cu128 版、跑一次真实 GPU 矩阵乘法自检。

### 5.3 `WARNING: llamafactory 0.9.5 does not provide the extra 'torch'`

PyPI 包没有这些 extras，只是警告。正确命令是：

```bat
pip install llamafactory==0.9.5
```

### 5.4 `ValueError: False is not a valid IntervalStrategy`

命令行覆盖 `eval_strategy=no` 被 YAML 解析成布尔值。写成 `eval_strategy=steps`，
或者直接不传这个参数。

### 5.5 训练完了但 `models\qwen3.5-9b-aerosentry-lora` 是空的

看 `Total optimization steps`：如果只有个位数，说明 `save_steps` 比总步数大，
一次都没存。本项目三份配置已改成 20–25 步，不会出现这个问题；自己改配置时注意。

### 5.6 页面打开是空白 / 只有框架没有数据

- 打开的是 `ui\index.html`（源码版）→ 它需要服务器，改用根目录的 `index.html`；
- 根目录 `index.html` 打不开 → 文件被截断，重新解压。

### 5.7 连接模型后对话没变化

浏览器控制台看 `fetch` 是否被 CORS 拦。Ollama 默认允许本地来源；
如果用 vLLM，启动时加 `--allowed-origins "*"`。

---

## 6. 项目结构

```
index.html              单文件系统（HTML + CSS + 引擎 + 数据）
TUTORIAL.md             本教程
README.md               项目说明
model/
  Qwen3.5-9B/           基座权重（19 GB，离线可用）
  Modelfile.aerosentry    Ollama 模型定义
  download_model.bat    重新下载基座
reports/                D01–D09 提交材料、PDF、测试结果
docs/                   封面、截图、证据图
source/
  assistant/            Python 推理引擎
  ui/                   前端源码 + 浏览器引擎
  training/             微调管线（train.bat / 配置 / 数据）
  tests/                15 项行为测试、模型评测
  tools/                单文件构建、封面与证据图生成
```

---

## 7. 结果

| 项 | 结果 |
|---|---|
| 官方行为测试 | **15 / 15** |
| 五个官方事件 | EVT001 中 · EVT002 低 · EVT003 无法判断 · EVT004 低 · EVT005 低 |
| 引擎一致性（Python vs 浏览器） | 5 / 5 完全一致 |
| 异常阈值 | 5 pp ≈ 1.7σ，仅 6.2% 的机场-月触发 |

复算：

```bat
cd source
python tests\run_tests.py
```

---

有问题先看第 5 节。还解决不了，把 `training\_install.log` 或控制台截图发我。
