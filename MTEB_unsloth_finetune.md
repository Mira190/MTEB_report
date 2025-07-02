# 使用 Unsloth 对 Qwen3 进行高效微调与评估

## **关键词：** Qwen3、LoRA / QLoRA、Dynamic 2.0 量化、Unsloth、vLLM、W\&B Weave、可解释思考模式

## 目录

1.  何时以及为何选择 Qwen3
2.  Qwen3 的动态思考模式解析
3.  环境准备与依赖安装
    3.1 Python / CUDA 版本建议
    3.2 创建隔离环境
    3.3 安装 PyTorch
    3.4 安装 Unsloth + Dynamic 2.0 量化
    3.5 vLLM 或 Ollama（可选推理验证）
    3.6 Weights & Biases / Weave 监控评估
4.  数据集准备与格式化
5.  最新微调方案与技术深度解析
    5.1 LoRA 与 QLoRA 的对比及选择
    5.2 Flash-Attention 2 + Triton 加速的原理与实践
    5.3 单卡 / 多卡混合精度配置
6.  端到端示例代码（含保存-加载-推理）
    6.1 详细脚本分解与代码注释
    6.2 训练过程日志与可视化解读
7.  模型评估：W\&B Weave 与 EvalScope 全流程实战
    7.1 评估指标的选择与意义
    7.2 使用 Weave 与 EvalScope 进行多模型对比与结果分析
    7.3 Weave EvaluationLogger 与 EvalScope 的灵活应用
8.  实战案例：Qwen3-14B 在 AIME 2024 数学基准上的卓越表现
9.  常见问题与调优技巧
10. 结论与未来方向
11. 进一步阅读

---

## 1\. 何时以及为何选择 Qwen3

Qwen3 系列模型（通义千问 3）是阿里巴巴云推出的新一代开源大型语言模型，在 2025 年持续展现出强大的竞争力。其核心优势在于：

- **多尺寸覆盖，极致灵活部署：** Qwen3 模型家族庞大，提供了从 **0.6B（可在边缘设备运行）到 235B（包含 Mixture-of-Experts, MoE 架构）** 的多种尺寸。这使得它能够轻松适配各种部署场景，无论是资源受限的智能手机、IoT 设备，还是对性能要求极高的云端服务器和数据中心。这种灵活性是企业和开发者在选择基础模型时的重要考量。
- **思考 / 非思考双模式，兼顾解释性与低延迟：** Qwen3 独有的动态思考模式允许模型显式输出其内部的思维链，即 `<think>...</think>` 标签中包含的详细推理过程。这对于需要 **可解释性 (Interpretability)** 的应用场景（如教育、法律、复杂问题诊断）至关重要。同时，在需要 **低延迟 (Low-latency)** 的生产环境（如实时聊天机器人）中，可以便捷地关闭思考模式，以获得更快的响应速度，实现性能与解释性的按需权衡。
- **卓越的多语种能力：** Qwen3 在设计之初就考虑了全球化应用的需求，其训练数据广泛覆盖了 **119 种语言及方言**。这确保了模型在处理全球范围内的多语言任务时，无论是主流语种还是资源较少的方言，都能提供高保真的翻译和精确的指令遵循能力。这对于开发国际化产品至关重要。
- **突破性 128K 上下文长度：** 对于处理超长文本的应用场景，Qwen3 的大型模型和 MoE 变体能够一次性处理高达 **128K 个 token** 的上下文长度。这远超许多主流 LLM 的限制，使其在多文档摘要、法律文本分析、大型代码库理解等任务中表现出无与伦比的优势。
- **友好的 Apache 2.0 许可证：** Qwen3 系列模型以 **Apache 2.0 许可证** 发布所有权重，这意味着它对商业使用高度友好。企业和开发者可以自由地使用、修改、分发和二次开发 Qwen3 模型，无需担心版权和许可问题，极大地促进了其在开源社区的普及和创新。

---

## 2\. Qwen3 的动态思考模式解析

Qwen3 的动态思考模式是其核心创新之一，提供了对模型行为更深层次的洞察和控制。它通过在模型生成最终答案之前插入一个结构化的思考块来实现：

- `enable_thinking = True`（默认设置）：当模型认为需要进行复杂推理时，它会在最终答案之前生成一个 `<think>...</think>` 标签，其中包含了模型详细的 **链式推理 (Chain-of-Thought)** 过程、中间计算步骤或逻辑推导。这使得模型的决策路径透明化，极大地提升了模型的可解释性和可调试性。
- `enable_thinking = False`：在这种模式下，模型会省略推理内容。尽管 `<think></think>` 占位符可能仍然存在（取决于具体的 token 化实现），但其中的内容将为空，模型会直接给出简洁的答案。这适用于对速度要求高、对解释性要求低的应用。
- 在提示 (Prompt) 尾部添加 `/no_think`：这是一种更强大的控制方式。无论 `enable_thinking` 标志设置为 `True` 还是 `False`，如果在用户提示的末尾加上 `/no_think`，模型都会被 **强制关闭思考模式**。这意味着它不会生成任何推理内容，从而确保了输出的简洁性。

**该机制的价值：**
这种设计让开发者在不同场景下灵活权衡模型的 **速度 (Speed)** 和 **解释性 (Interpretability)**。在开发和调试阶段，开启思考模式可以帮助我们理解模型犯错的原因，优化提示策略或微调数据。在生产环境中，则可以根据用户需求和应用场景动态切换模式，例如在数学求解器中启用思考模式以展示解题步骤，而在智能客服中禁用思考模式以提高响应速度。

---

## 3\. 环境准备与依赖安装

为了确保 Qwen3 微调过程的顺畅和高效，正确的环境配置至关重要。

### 3.1 Python / CUDA 版本建议

| 组件            | 建议版本                                      | 备注                                                                             |
| :-------------- | :-------------------------------------------- | :------------------------------------------------------------------------------- |
| **Python**      | `3.10` – `3.12`                               | 推荐 `3.10` 或 `3.11` 以获得更广泛的库兼容性。                                   |
| **CUDA**        | `12.1+` 或 `12.2+`（与 PyTorch 官网命令匹配） | 务必确认您的 GPU 驱动支持此 CUDA 版本。                                          |
| **cuDNN**       | 对应 CUDA 版本，通常随 CUDA 工具包安装        | 优化深度学习计算。                                                               |
| **GCC**         | `GCC ≥ 9.4`                                   | 在编译 Triton 内核时（如 Unsloth 内部使用）更稳定。建议安装 `gcc-9` 或更高版本。 |
| **NVIDIA 驱动** | 最新稳定版本                                  | 确保与 CUDA 版本兼容，并充分发挥 GPU 性能。                                      |

### 3.2 创建隔离环境

使用 Conda 或 Virtualenv 创建一个独立的 Python 环境，以避免库版本冲突。强烈推荐 Conda。

```bash
conda create -n qwen3_ft python=3.10  # 创建名为 qwen3_ft 的 Python 3.10 环境
conda activate qwen3_ft                 # 激活环境
```

### 3.3 安装 PyTorch

根据您的显卡型号和所需的 CUDA 版本，从 PyTorch 官网获取准确的安装命令。以下示例针对 CUDA 12.1：

```bash
# ⚠️ 重要：请根据您的显卡和 CUDA 版本调整命令！
# 访问 PyTorch 官网：https://pytorch.org/get-started/locally/
# 选择您的配置（如 Linux, Pip, CUDA 12.1），复制生成的命令。
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**提示：** 如果您使用的是较新的 NVIDIA GPU（如 RTX 30/40 系列或 H100），并且希望利用最新的 `bfloat16` 混合精度，请确保您的 PyTorch 版本支持。

### 3.4 安装 Unsloth + Dynamic 2.0

Unsloth 在 2025 年推出了革命性的 **Dynamic 2.0 量化内核**，它针对 Qwen3 等主流 LLM 的官方权重做了高度优化的 4-bit 动态量化。这项技术带来了显著的性能提升和资源节省：

- **几乎零精度损失：** 通过精巧的量化算法，Dynamic 2.0 在保持模型性能的同时，实现了极低的精度损失。
- **速度提升 1.8-2.3 倍：** 针对 GPU 架构深度优化，加速了矩阵乘法和 Attention 机制，显著缩短了微调时间。
- **显存节省 60-70%：** 4-bit 量化大幅减少了模型在 GPU 显存中的占用，使得单张消费级 GPU（如 8GB 或 12GB 显存）也能轻松微调大型 Qwen3 模型。

**安装命令：**

```bash
pip install "unsloth[4bit]"  # 这会自动安装必要的 triton-kernel 等依赖
```

**对于显存受限的用户（例如仅有 8GB 显存的显卡）：**
如果您仅有 8GB 显存，除了使用 Dynamic 2.0，还可以考虑：

- **升级 `llama-cpp-python`：**
  ```bash
  pip install --upgrade llama-cpp-python
  ```
- **使用 GGUF 权重：** Unsloth 的 Dynamic 2.0 已原生支持 Qwen3 的 GGUF 权重格式，这进一步优化了 CPU/GPU 混合推理时的内存使用和加载速度。您可以在 Hugging Face 上找到社区转换的 GGUF 格式 Qwen3 模型。

### 3.5 vLLM 或 Ollama（可选推理验证）

微调后的模型需要进行推理验证，以评估其在实际应用中的表现。

- **vLLM：高性能高吞吐推理引擎**

  ```bash
  pip install vllm
  ```

  **优势：** vLLM 从 0.4.0 版本开始，内置了对 Qwen3 KV-cache 机制和思考标签 (`<think>`) 解析的优化支持，无需手动打补丁（patch）。它特别适合需要高并发、低延迟的生产环境部署。
  **启动示例：**

  ```bash
  vllm serve Qwen/Qwen3-8B --port 8000 --enable-reasoning --reasoning-parser deepseek_r1
  ```

  这会启动一个兼容 OpenAI API 的推理服务，可以通过 `curl` 或 `openai` Python 客户端调用。

- **Ollama：本地轻量级模型运行工具**

  ```bash
  # macOS 用户
  brew install ollama
  # Linux 用户请参考 Ollama 官网下载并安装二进制文件
  # https://ollama.com/download/linux
  ```

  **优势：** Ollama 是在本地运行大型语言模型的极简工具，安装和使用都非常便捷。它适合个人开发者进行快速的本地测试和原型验证。
  **使用示例：**

  ```bash
  ollama run qwen3:8b  # 如果本地没有会自动下载并启动交互模式
  ```

  更多关于 Qwen3 在 vLLM/Ollama 上部署的细节，请参考 Qwen3 官方文档或社区教程。

### 3.6 Weights & Biases / Weave 监控评估

**Weights & Biases (W\&B)** 是一个强大的机器学习实验跟踪和可视化平台，而 **W\&B Weave** 是其专为 LLM 评估设计的模块。它们对于长时间训练、团队协作和量化评估至关重要。

```bash
pip install wandb weave
wandb login <your_token> # 按照提示输入您的 W&B API Token
```

**W\&B 作用：**

- **实时监控：** 记录训练过程中的损失 (loss)、学习率 (learning rate) 变化、梯度范数等关键指标。
- **可视化报告：** 自动生成训练曲线图表，直观展示模型性能趋势。
- **协作与复现：** 方便团队成员远程查看、分享实验记录，并确保实验的可复现性。
- **资源利用率：** 监控 GPU、CPU、内存等硬件利用率，帮助发现性能瓶颈。

**Weave 作用：**

- **结构化评估：** 提供一套工具来系统化地比较不同模型（如微调前后、不同微调参数）的性能。
- **自动化指标计算：** 支持集成常见的评估指标（如 BLEU、ROUGE、模型判断准确率）。
- **交互式分析：** 尤其擅长生成并排对比报告，方便人工检查模型输出差异，甚至用于捕捉微调引入的潜在错误（如“幻觉”）。

---

## 4\. 数据集准备与格式化

微调的效果在很大程度上取决于训练数据的质量和格式。对于 Qwen3，通常建议将数据转换为其 **ChatML 格式**，以便模型能够学习到正确的对话模式和指令遵循能力。

**建议数据集字段：**
典型的指令微调数据集包含以下字段：

- `instruction` (指令)：用户希望模型执行的任务描述。
- `input` (输入)：任务相关的额外上下文信息（可选）。
- `output` (输出)：模型应该生成的期望响应。

**格式化原则：**

- **统一转为 Qwen3 ChatML 格式：** Qwen3 遵循类似于 OpenAI ChatML 的对话格式，其中包含 `role` (角色，如 `user`, `assistant`) 和 `content` (内容)。
- **训练阶段默认 `enable_thinking=False`：** 在准备训练数据时，通常建议在 `tokenizer.apply_chat_template` 中设置 `enable_thinking=False`。这有助于防止模型在微调过程中学到大量的 `<think>...</think>` 标记，从而避免 **过拟合 (Overfitting)** 到这些内部思考过程，确保模型在生成最终答案时更为直接和高效。如果您的目标是让模型生成思考过程，那么才需要在训练数据中包含它们。
- **示例格式化函数：** 在后续的“端到端示例代码”中，将提供一个具体的 `format_example` 函数，演示如何将常见的三字段数据集转换为 Qwen3 兼容的 ChatML 文本格式，供微调器使用。

**数据质量与数量：**

- **高质量：** 确保 `instruction`、`input` 和 `output` 之间的逻辑一致性，避免数据中的错误、偏见或不一致的风格。
- **多样性：** 数据应覆盖您希望模型处理的各种任务和情境，避免模型在未见过的情境中表现不佳。
- **数量：** 对于 LoRA/QLoRA 微调，几百到几千条高质量数据通常就能取得不错的效果。对于更复杂的任务或要求更高的性能，数据量可以适当增加。

---

## 5\. 最新微调方案与技术深度解析

2025 年的 LLM 微调技术在效率和性能上都有了显著提升，Unsloth 将这些前沿技术无缝集成，极大地降低了微调的门槛。

### 5.1 LoRA 与 QLoRA 的对比及选择

| 技术           | 说明                                                                                                                                                                                                     | 优势                                                                                                                                         | 适用场景                                                                                                                    |
| :------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------- |
| **LoRA**       | **低秩适配 (Low-Rank Adaptation)**。冻结原始预训练模型的大部分权重，仅向模型中注入少量可训练的“适配器层”(Adapter Layers)。这些适配器层的参数量很小（通常是原始模型的 0.01% - 0.1%）。                    | 显著减少可训练参数量，大幅降低 GPU 显存占用和训练时间。可在一个 GPU 上训练大型模型，且易于合并回原模型或共享。                               | **数据量适中 (≤ 80k 样本)** 的场景，希望在有限资源下对模型进行高效的领域适应或风格定制，且对性能有一定要求。                |
| **QLoRA**      | **量化 LoRA (Quantized LoRA)**。在 LoRA 的基础上，将预训练模型权重进一步量化到 4-bit（或更低精度），然后在这 4-bit 量化权重上进行 LoRA 微调。                                                            | **显存占用最低**，甚至可以在单张 8GB 或 12GB 消费级 GPU 上微调 Qwen3-7B/14B 等大型模型。通过 **Unsloth Dynamic 2.0**，实现了极低的精度损失。 | **消费级 GPU** 用户（如只有 8GB 或 12GB 显存），希望在极低硬件成本下进行 LLM 微调。对性能要求不是极致苛刻，但仍需较好效果。 |
| **DPO / SPIN** | **直接偏好优化 (Direct Preference Optimization)** / **自播放迭代训练 (Self-Play Fine-tuning)**。DPO 直接在偏好数据对（好答案 vs. 坏答案）上训练策略，无需奖励模型。SPIN 则通过模型自我对弈生成偏好数据。 | **强行提升模型对齐**，使其输出更符合人类偏好或特定行为。在 RLHF 之后或作为其替代方案。                                                       | 需要模型输出 **高度符合特定价值观、风格或行为模式** 的场景，例如内容审核、安全提示或遵循复杂指令。需要有高质量的偏好数据。  |

### 5.2 Flash-Attention 2 + Triton 加速

**Flash-Attention 2** 是一种高度优化的 Attention 机制实现，由 Triton 语言编写。它的核心优势在于：

- **线性显存：** 传统 Attention 的显存消耗通常是序列长度的平方 ($O(L^2)$)，而 Flash-Attention 2 通过优化计算图和内存访问模式，将显存消耗降低到几乎线性 ($O(L)$)，从而支持处理更长的上下文序列。
- **计算加速：** 减少了 GPU 内存读写次数，提升了计算吞吐量，使训练和推理速度更快。

**Unsloth 的集成：**
Unsloth 在其内部内核中 **已经把 Flash-Attention 2 完全内嵌**。这意味着您在安装 Unsloth 后，无需再手动安装和配置 Flash-Attention 2。当您在 `FastLanguageModel.get_peft_model` 中设置 `use_gradient_checkpointing="unsloth"` 时，Unsloth 会自动启用其优化的内核，其中就包括 Flash-Attention 2 的加速能力，从而为您提供开箱即用的高性能微调体验。

### 5.3 单卡 / 多卡混合精度配置

**混合精度训练 (Mixed Precision Training)** 是一种结合了 `float16` (或 `bfloat16`) 和 `float32` 精度进行训练的技术。

- **`float16` / `bfloat16` (低精度)：** 占用更少的显存，且在最新的 GPU 上计算速度更快。
- **`float32` (高精度)：** 保持数值稳定性，避免在训练过程中出现梯度消失或爆炸。

**配置方法：**
在 `transformers.TrainingArguments` 中进行配置：

```python
import torch

# ... 其他训练参数 ...
args=TrainingArguments(
    # ...
    fp16=not torch.cuda.is_bf16_supported(), # 如果不支持 bfloat16，则使用 float16
    bf16=torch.cuda.is_bf16_supported(),     # 如果支持 bfloat16，则使用 bfloat16
    # ...
)
```

**建议：**

- **优先使用 `bfloat16`：** 如果您的 GPU (如 NVIDIA A100、H100、RTX 30/40 系列) 和 PyTorch 版本支持 `bfloat16`，强烈建议使用它。`bfloat16` 在数值范围上与 `float32` 相同，但在精度上更低，能更好地保持训练稳定性，避免 `float16` 偶尔出现的溢出问题。
- **多卡 DDP (Distributed Data Parallel)：** 对于更大的 Qwen3 模型或更大的数据集，可以采用多卡 DDP 进行分布式训练。Unsloth 与 Hugging Face `Trainer` 的 DDP 支持是兼容的，只需在 `TrainingArguments` 中设置 `num_gpus` 或通过 `torch.distributed.launch` 启动脚本。

---

## 6\. 端到端示例代码（含保存-加载-推理）

以下脚本已根据 2025 年最新 Unsloth API 更新，支持 Dynamic 2.0 4-bit 量化、bfloat16 混合精度回退、以及自动与 Weave 日志集成。

### 6.1 详细脚本分解与代码注释

```python
# =========================================================
# 1️⃣ 环境设置与随机种子固定：确保实验可复现性
# =========================================================
import os
import torch
import random
import numpy as np

# 设置随机种子，保证每次运行结果一致
SEED = 3407
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# 对于 CUDA 操作，也需要设置额外的随机种子
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True # 确保 cuDNN 操作确定性
    torch.backends.cudnn.benchmark = False    # 关闭 cuDNN 自动寻找最佳算法，以确保确定性

# =========================================================
# 2️⃣ 载入基础模型：使用 Unsloth FastLanguageModel
# =========================================================
from unsloth import FastLanguageModel

# 定义模型路径和加载参数
# "unsloth/Qwen3-8B-Chat" 是 Unsloth 优化过的 Qwen3 模型，通常包含量化支持
MODEL_NAME = "unsloth/Qwen3-8B-Chat"
MAX_SEQ_LENGTH = 2048 # 最大序列长度，根据您的GPU显存和数据特点调整
LOAD_IN_4BIT = True   # 启用 4-bit 量化，使用 Unsloth Dynamic 2.0 技术

# 从预训练模型加载，自动进行量化和优化
print(f"正在加载基础模型: {MODEL_NAME} (4-bit 量化: {LOAD_IN_4BIT})...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=LOAD_IN_4BIT,
    # dtype=None # Unsloth 会自动选择合适的dtype (bfloat16 if supported, else float16)
)
print("模型加载完成。")

# =========================================================
# 3️⃣ 注入 LoRA 适配器：进行参数高效微调
# =========================================================
# r: LoRA 的秩，决定适配器层的参数量。更大的 r 可以捕获更多信息，但也需要更多显存和计算。
# lora_alpha: LoRA 缩放因子，通常与 r 相同。
# lora_dropout: LoRA 层 dropout 率，防止过拟合。
# target_modules: 指定要注入 LoRA 的模块。Qwen3 通常包括这些Attention和MLP层的投影。
# use_gradient_checkpointing="unsloth": 启用 Unsloth 优化的梯度检查点，进一步节省显存。
print("正在注入 LoRA 适配器...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                       # 可以在显存更大的卡上尝试 32 或 64
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", # Attention 层的投影矩阵
        "gate_proj", "up_proj", "down_proj"     # MLP 层的投影矩阵
    ],
    use_gradient_checkpointing="unsloth", # 启用 Unsloth 的优化，自动利用 Flash Attention 2
    random_state=SEED, # 保持随机性一致
    use_rslora=False,  # 更多参数高效 LoRA 变体，此处不启用
    loftq_config=None, # LoFTQ 是一种量化感知 LoRA，此处不启用
)
print("LoRA 适配器注入完成。")

# =========================================================
# 4️⃣ 数据集准备与格式化：将原始数据转为模型可训练格式
# =========================================================
from datasets import load_dataset

# 加载数据集，这里使用 alpaca-cleaned 的前 30% 作为示例
print("正在加载数据集 (yahma/alpaca-cleaned)...")
dataset = load_dataset("yahma/alpaca-cleaned", split="train[:30%]")

# 定义数据格式化函数，将原始数据转换为 Qwen3 ChatML 格式
# 注意：add_generation_prompt=False 和 enable_thinking=False 在微调阶段很重要
def format_example(example):
    # 根据是否有 input 字段，构建用户消息
    user_message = example["instruction"]
    if example["input"].strip(): # 检查 input 是否为空或只有空格
        user_message += "\n\n" + example["input"]

    # 构建 Qwen3 ChatML 消息列表
    messages = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": example["output"]}
    ]

    # 应用聊天模板，tokenize=False 返回字符串，add_generation_prompt=False 确保只包含训练数据
    # enable_thinking=False 确保模型在微调时不会在答案中生成 <think> 标签
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False # 训练时通常不希望模型自己产生思考标签
    )
    return {"text": formatted_text}

# 使用 map 函数并行处理数据集
print("正在格式化数据集...")
dataset = dataset.map(format_example, num_proc=4) # num_proc 可以根据CPU核心数调整
print(f"数据集格式化完成，首条数据示例:\n{dataset[0]['text']}")

# =========================================================
# 5️⃣ 训练模型：使用 TRL 库的 SFTTrainer
# =========================================================
from trl import SFTTrainer
from transformers import TrainingArguments

# 训练参数配置
# per_device_train_batch_size: 每个 GPU 上的批次大小
# gradient_accumulation_steps: 梯度累积步数，模拟更大批次
# max_steps: 最大训练步数，控制训练时长
# learning_rate: 学习率
# warmup_ratio: 学习率预热比例
# logging_steps: 日志记录频率
# optim: 优化器，"adafactor" 在内存占用方面优于 AdamW
# lr_scheduler_type: 学习率调度器类型
# report_to="wandb": 自动将训练日志发送到 Weights & Biases
# output_dir: 模型和训练状态保存目录
print("正在配置训练器并开始训练...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text", # 指定数据集中的文本字段
    max_seq_length=MAX_SEQ_LENGTH,
    # dataset_num_proc=2, # 如果map时已并行，这里可以不设或设为1
    packing=False, # packing 可以优化长序列训练效率，但会改变每个batch的实际长度，这里为简单起见不启用
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8, # 实际批次大小 = 2 * 8 = 16
        max_steps=100, # 示例用，实际应根据数据集大小和收敛情况调整
        learning_rate=2e-4,
        warmup_ratio=0.03,
        logging_steps=1,
        fp16=not torch.cuda.is_bf16_supported(), # 如果不支持 bf16，则使用 fp16
        bf16=torch.cuda.is_bf16_supported(),     # 优先使用 bf16
        optim="adafactor",                       # Adafactor 优化器内存占用更少
        weight_decay=0.01,
        lr_scheduler_type="cosine",              # 学习率余弦退火
        report_to="wandb",                       # 报告到 W&B
        output_dir="qwen3_lora",                 # 模型保存路径
        seed=SEED,                               # 训练随机种子
        # save_steps=50, # 每隔多少步保存一次检查点
        # save_total_limit=3, # 最多保存多少个检查点
        # evaluation_strategy="steps", # 可以设置评估策略
        # eval_steps=50,
        # logging_dir="./logs", # 日志目录
    ),
)
trainer.train()
print("模型训练完成。")

# =========================================================
# 6️⃣ 推理与保存：微调后模型测试与部署
# =========================================================
from unsloth import prepare_inference

# 准备推理模式：Unsloth 会自动切换 eval 模式并进行推理优化
print("正在准备模型进行推理...")
prepare_inference(model) # 内部会调用 model.eval() 并进行其他优化

# 保存 LoRA 适配器和 tokenizer
SAVE_DIR = "qwen3_lora_fine_tuned"
print(f"正在保存 LoRA 适配器和分词器到 {SAVE_DIR}...")
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print("模型保存完成。")

# 进行推理测试
print("正在进行推理测试...")
user_query = "1, 1, 2, 3, 5, 8，请继续斐波那契序列"
messages = [{"role": "user", "content": user_query}]

# 应用 ChatML 模板，并指定 enable_thinking=False 进行简洁输出
prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True, # 添加生成提示，告诉模型开始生成
    enable_thinking=False,      # 推理时不启用思考模式
    tokenize=False              # 返回字符串而非 token ID 列表
)

# 将 prompt 转换为 token ID 并移动到 CUDA 设备
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")["input_ids"]

# 生成响应
outputs = model.generate(
    inputs,
    max_new_tokens=20, # 最大生成 token 数
    do_sample=True, # 启用采样生成
    temperature=0.7, # 采样温度
    top_p=0.8, # Top-p 采样
    top_k=20, # Top-k 采样
    min_p=0.0, # 最小概率阈值
    use_cache=False, # 是否使用 KV 缓存 (通常推理时开启，这里为简化示例设为False)
)

# 解码生成的 token 并打印结果
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n=========== 微调模型推理输出 ============")
print(generated_text)

# 清理内存，准备重新加载模型以验证保存
del model
del tokenizer
torch.cuda.empty_cache()

# =========================================================
# 7️⃣ 重新加载与推理：验证保存的模型是否可用
# =========================================================
print(f"\n正在从 {SAVE_DIR} 重新加载模型以验证...")
reloaded_model, reloaded_tokenizer = FastLanguageModel.from_pretrained(
    model_name=SAVE_DIR,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=LOAD_IN_4BIT,
)
prepare_inference(reloaded_model) # 再次准备推理模式

# 使用重新加载的模型进行推理
reloaded_inputs = reloaded_tokenizer(prompt, return_tensors="pt").to("cuda")["input_ids"]
reloaded_outputs = reloaded_model.generate(
    reloaded_inputs,
    max_new_tokens=20,
    do_sample=True,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    min_p=0.0,
    use_cache=False,
)
reloaded_generated_text = reloaded_tokenizer.decode(reloaded_outputs[0], skip_special_tokens=True)
print("\n=========== 重新加载模型推理输出 ===========")
print(reloaded_generated_text)
print("模型重新加载并推理验证完成。")
```

### 6.2 训练过程日志与可视化解读

通过 `report_to="wandb"` 参数，训练过程的损失 (loss)、学习率 (learning rate) 等关键指标将自动上传到您的 Weights & Biases (W\&B) 项目。在 W\&B 界面中，您可以实时查看：

- **训练损失曲线：** 观察模型是否在学习，以及是否存在过拟合或欠拟合的趋势。
- **学习率变化：** 追踪学习率调度器的效果。
- **硬件资源利用率：** 监控 GPU 显存、使用率、CPU 使用情况，帮助诊断性能瓶颈。
- **系统指标：** 记录温度、功耗等，确保硬件运行在健康状态。

这些可视化日志对于诊断训练问题、比较不同超参数配置的效果至关重要。

---

I've integrated **EvalScope** into the "模型评估" section of your document and ensured all code blocks are correctly displayed within the Markdown format.

---

## 7\. 模型评估：W\&B Weave 与 EvalScope 全流程实战

微调完成后的模型性能评估是至关重要的环节。W\&B Weave 提供了一套强大的工具，帮助您系统化、可视化地比较模型表现。同时，**EvalScope** 作为灵活的评估框架，可以与 Weave 协同工作，提供更深层次的评估能力。

### 7.1 评估指标的选择与意义

除了传统的语言生成指标，针对 LLM 微调，我们关注以下几类：

- **`loss_only` (交叉熵损失)：**

  - **意义：** 仅对 `assistant` 生成的 token 计算交叉熵损失。这能直接反映模型在微调数据上的“学习”效果，即预测目标答案的准确程度。损失越低，说明模型在给定输入后，生成期望答案的概率越高。
  - **应用：** 通常用于验证微调数据是否被模型有效吸收。

- **BLEU / ROUGE：**

  - **意义：** 传统的机器翻译和文本摘要评估指标，通过比较生成文本与参考文本的重叠度来衡量流畅度和内容覆盖。
  - **应用：** 在生成型任务（如摘要、翻译）中衡量生成质量。但对于开放式问答，可能不如人类评估或基于 LLM 的评估准确。

- **可解释思考一致性：**

  - **意义：** 针对 Qwen3 的独特思考模式，评估模型生成的 `<think>...</think>` 块中的推理过程是否逻辑正确、步骤清晰，并最终导向正确的答案。这可能需要人工标注或使用另一个强大的 LLM 作为“裁判”进行自动化评估。
  - **应用：** 在需要模型提供透明推理过程的教育、决策支持等领域至关重要。

- **人类偏好评估 (Human Preference)：**

  - **意义：** 最直接、最可靠的评估方式。让人类专家对模型的输出进行评分，判断其相关性、连贯性、有用性和安全性。
  - **应用：** 最终验证模型在实际使用中的用户体验。Weave 的竞技场模式支持此类评估。

### 7.2 使用 Weave 进行多模型对比与结果分析

Weave 的核心优势在于其**交互式仪表板和并排比较视图**。您可以：

1.  **定义评估数据集：** 建议使用与训练集不同但来自同一分布的 **验证集 (Validation Set)** 或 **测试集 (Test Set)**，或直接使用真实的生产对话样本。
2.  **集成模型：** 将您的微调模型（LoRA Qwen3）和原始基础模型（Base Qwen3）注册到 Weave 中。
3.  **运行评估：** Weave 会对每个模型在同一组评估样本上生成响应，并捕获响应时间、所用 prompt 等元数据。
4.  **可视化对比：** Weave 自动生成详细报告，您可以：
    - **Side-by-side 对比：** 直观地看到两个模型对同一 prompt 的不同输出，快速发现微调效果。
    - **误差热力图：** 高亮显示模型在哪些类型的样本上表现不佳。
    - **思考链长度统计：** 如果启用了思考模式，可以分析思考步骤的长度和复杂性。

**详见官方 Weave 示例仓库和本文代码段。** Weave 不仅能提供量化指标，更重要的是帮助您进行**定性分析**，发现模型改进的方向和潜在的 Bug。

### 7.3 Weave EvaluationLogger 的灵活应用

除了声明式 (`weave.Evaluation`) 的评估方式，Weave 也提供了更灵活的 **`EvaluationLogger`**。当您现有的评估流程已经很复杂，或者需要更细粒度的控制时，`EvaluationLogger` 就显得非常有用。

它允许您**命令式地**在代码中记录每一次预测的输入、输出和对应的分数，而不是预先定义一个完整的评估配置。

```python
from weave.flow.eval_imperative import EvaluationLogger
import weave; weave.init("your_evaluation_project") # 初始化 W&B Weave 项目

# 1. 初始化 EvaluationLogger
# 这里的 model 和 dataset 可以是任意字符串标识符
eval_logger = EvaluationLogger(model="qwen3_14b_custom", dataset="my_custom_eval_set")

# 2. 遍历你的评估数据
for i, row in enumerate(dataset): # 假设 dataset 是你的评估数据源
    print(f"处理样本 {i+1}...")

    # 3. 调用你的模型进行预测
    # 这里的 model.predict 可以是你的实际推理函数调用
    # 比如：output = your_qwen3_model.generate(...)
    # 为了示例，这里使用一个模拟的 predict 函数
    output = your_inference_function(row["text_input"])

    # 4. 记录每次预测的输入和输出
    # inputs 可以是字典形式，包含原始 prompt 或其他相关信息
    pred_logger = eval_logger.log_prediction(inputs={"text_input": row["text_input"], "true_label": row["label"]}, output=output)

    # 5. 计算并记录得分
    # 这里的 gpt4o_judge 可以是你自定义的评分函数
    score_result = gpt4o_judge(row["label"], output) # score_result 假设是一个字典，如 {"correctness": True, "reasoning": "..."}

    # 记录单个指标
    pred_logger.log_score("correctness", score_result.get("correctness", False))
    # 如果有多个指标，可以多次调用 log_score 或在 score_result 中定义更多
    pred_logger.log_score("reasoning_quality", score_result.get("reasoning_quality", None))

    # 6. 完成本次预测的记录
    pred_logger.finish()

# 7. 循环结束后，记录评估总结
eval_logger.log_summary()
```

这种命令式风格让您能够将 Weave 评估集成到现有的复杂评估脚本中，同时仍然享受到 Weave 的仪表板、比较工具和可视化功能。

### 7.4 EvalScope：LLM 评估的全面框架

**EvalScope** 是一个专注于 LLM 评估的开源框架，旨在提供一个灵活、可扩展的解决方案，以应对不同模型、任务和评估指标的需求。它与 W\&B Weave 这样的平台可以互补，Weave 擅长可视化和实验管理，而 EvalScope 则可能在底层评估逻辑和自定义任务集成方面提供更强的灵活性。

**EvalScope 的特点：**

- **统一的评估接口：** EvalScope 提供了一套统一的 API，允许用户轻松集成各种 LLM（包括本地部署模型和通过 API 访问的云端模型）进行评估。
- **丰富的评估任务支持：** 内置支持多种常见的 LLM 评估任务，如问答、摘要、翻译、代码生成等。
- **可插拔的评估指标：** 除了传统的指标（如 BLEU, ROUGE），EvalScope 也支持基于 LLM 的自动评估指标，甚至允许用户自定义评估函数。
- **数据集管理：** 提供了方便的数据集加载和管理功能，支持多种数据格式。
- **报告生成：** 自动化生成结构化的评估报告，帮助用户快速理解模型性能。

**如何集成 EvalScope（概念性示例）：**

假设您已经安装了 EvalScope (`pip install evalscope`)，以下是一个概念性的集成示例：

```python
# 您需要根据EvalScope的实际API和您的具体评估流程来调整此代码。
# from evalscope.models import register_model
# from evalscope.datasets import register_dataset
# from evalscope.metrics import register_metric
# from evalscope.evaluator import Evaluator
# from unsloth import FastLanguageModel, prepare_inference
# from datasets import load_dataset # 假设你的评估数据来自这里

# 1. 定义您的 Qwen3 模型接口 (EvalScope注册模型的方式可能不同，这只是一个概念)
# @register_model("qwen3_fine_tuned_evalscope")
# class Qwen3FineTunedModelForEvalScope:
#     def __init__(self, model_path: str, max_seq_length: int = 2048, load_in_4bit: bool = True):
#         self.model, self.tokenizer = FastLanguageModel.from_pretrained(
#             model_name=model_path,
#             max_seq_length=max_seq_length,
#             load_in_4bit=load_in_4bit,
#         )
#         prepare_inference(self.model)

#     def generate(self, prompt: str, **kwargs) -> str:
#         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)["input_ids"]
#         outputs = self.model.generate(inputs, **kwargs)
#         return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# 2. 定义您的评估数据集接口 (EvalScope注册数据集的方式可能不同)
# @register_dataset("my_qwen3_eval_dataset_evalscope")
# class MyQwen3EvalDatasetForEvalScope:
#     def __init__(self, dataset_name: str, split: str = "validation"):
#         self.data = load_dataset(dataset_name, split=split)
#         # 确保数据格式符合EvalScope的期望，例如列表字典，每个字典有'prompt'和'reference'
#         self.formatted_data = [{"prompt": item["instruction"] + "\n\n" + item["input"], "reference": item["output"]} for item in self.data]

#     def __len__(self):
#         return len(self.formatted_data)

#     def __getitem__(self, idx):
#         return self.formatted_data[idx]

# 3. 定义自定义评估指标 (EvalScope注册指标的方式可能不同)
# @register_metric("my_llm_judge_metric")
# def my_llm_judge_metric(predictions: list[str], references: list[str]) -> dict:
#     # 这是一个概念性的LLM作为裁判的指标
#     # 实际实现会涉及调用一个强大的LLM API (如GPT-4o) 来对每个预测和参考进行评分
#     scores = []
#     for pred, ref in zip(predictions, references):
#         # 假设这里调用一个外部LLM服务来判断正确性
#         # judge_result = call_gpt4o_as_judge(pred, ref)
#         # scores.append(1 if judge_result["is_correct"] else 0)
#         scores.append(1 if pred.strip().lower() == ref.strip().lower() else 0) # 简化示例
#     return {"llm_judge_accuracy": sum(scores) / len(scores)}

# 4. 运行评估 (EvalScope评估器使用方式可能不同)
# if __name__ == "__main__":
#     model_path = "qwen3_lora_fine_tuned" # 你保存的模型路径
#     dataset_name = "yahma/alpaca-cleaned" # 你用于评估的数据集
#     eval_output_dir = "./evalscope_qwen3_results"

#     # 实例化模型和数据集 (EvalScope可能有自己的加载机制)
#     # qwen3_model = Qwen3FineTunedModelForEvalScope(model_path)
#     # eval_dataset = MyQwen3EvalDatasetForEvalScope(dataset_name, split="test") # 假设有测试集

#     # 初始化评估器 (EvalScope的Evaluator类和参数会有差异)
#     # evaluator = Evaluator(
#     #     model=qwen3_model,
#     #     dataset=eval_dataset,
#     #     metrics=["bleu", "rouge", "my_llm_judge_metric"], # 使用内置和自定义指标
#     #     output_dir=eval_output_dir,
#     #     evaluation_config={
#     #         "generation_kwargs": {"max_new_tokens": 50, "temperature": 0.7, "do_sample": True},
#     #         # 更多配置，如批处理大小、并发请求等
#     #     }
#     # )
#     # evaluator.run()
#     # print(f"EvalScope 评估完成，报告已生成在 {eval_output_dir}")

```

**EvalScope 与 W\&B Weave 的协同：**
您可以利用 **EvalScope** 进行底层的模型推理和指标计算，然后将这些结果（例如，每个样本的输入、模型输出、参考答案以及计算出的各项指标）导入到 **W\&B Weave** 中进行更高级的可视化、并排比较和团队协作。这种组合可以实现一个既灵活又强大的 LLM 评估工作流。

---

## 8\. 实战案例：Qwen3-14B 在 AIME 2024 数学基准上的卓越表现

一个引人注目的实战案例是 Qwen3-14B 模型在 **AIME 2024 (美国数学邀请赛)** 数据集上的表现。该基准用于测试模型在复杂数学推理和问题解决方面的能力。

- **测试方法：** 开启 Qwen3 的思考模式 (`enable_thinking=True`)，让模型在回答数学问题时显式输出解题步骤。为了避免本地显存瓶颈，推理通过 **OpenRouter API**（一个聚合了多种模型 API 的平台）进行。**GPT-4o** 被用作一个强大的自动裁判 LLM，来判断 Qwen3 的答案与正确答案的一致性。Weave Evaluation 被用于可视化地比对 Qwen3 的解答、GPT-4o 的判断结果以及真实答案。
- **结果：** 在 15 道 AIME 2024 题目上，开启思考模式的 Qwen3-14B 模型取得了 **66.7% 的正确率**。这一成绩显著超越了同期测试的另一个蒸馏模型——Distill-Llama3 14B，后者仅取得了 20% 的正确率。
- **意义：** 这项测试表明，Qwen3 的动态思考模式在复杂推理任务中具有显著优势，能够帮助模型分解问题并生成更准确的答案。同时，也印证了 Qwen3 在其模型尺寸下所展现出的强大推理能力，尤其是在数学和逻辑领域。

这个案例不仅展示了 Qwen3 的强大性能，也凸显了 Weave 在自动化评估和结果分析方面的价值，它能帮助研究者快速识别模型表现的优劣，甚至发现评估流程中的潜在 Bug（例如，之前在 Weave 中发现的，评分器会误判空白输出为正确答案的问题，通过添加最小长度检查得到了解决）。

---

## 9\. 常见问题 / 调优技巧

在 Qwen3 微调过程中，您可能会遇到一些常见问题或希望进一步优化性能：

| 问题                                      | 解决方案                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| :---------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **CUDA OOM**                              | **显存不足 (Out of Memory)** 是微调大模型最常见的问题。您可以尝试：\<br/\>1. 确保 `load_in_4bit=True` 启用 4-bit 量化 (Unsloth Dynamic 2.0)。\<br/\>2. 启用 `gradient_checkpointing="unsloth"`，这会显著节省显存（以略微增加计算时间为代价）。\<br/\>3. 减小 `per_device_train_batch_size`（例如从 2 降到 1）。\<br/\>4. 增大 `gradient_accumulation_steps` 来弥补 `batch_size` 减小带来的梯度更新频率降低。\<br/\>5. 减小 LoRA 的 `r` 值（例如从 16 降到 8）。\<br/\>6. 缩短 `max_seq_length`。\<br/\>7. 关闭其他占用 GPU 资源的应用程序。  |
| **推理输出中带 `<think></think>` 空标签** | 这是 Qwen3 聊天模板的设计特点。若不希望在推理时看到空标签：\<br/\>1. 在 **训练数据格式化阶段** 确保 `enable_thinking=False`，这样模型在训练时就不会学到填充思考标签。\<br/\>2. 在 **推理时** 调用 `tokenizer.apply_chat_template` 时，显式设置 `enable_thinking=False`。\<br/\>3. **后处理：** 如果模型偶尔仍生成空标签，在接收到模型输出后，使用正则表达式 (`re.sub(r'<think></think>', '', output_text)`) 或字符串替换 (`output_text.replace("<think></think>", "")`) 将其去除。                                                           |
| **微调后 BLEU / ROUGE 分数下降**          | 这可能是由多种因素导致：\<br/\>1. **数据污染：** 检查您的微调数据是否混入了错误标签、低质量样本或与任务不相关的噪音。\<br/\>2. **过拟合：** 模型可能过拟合了微调数据，导致在未见过的数据上泛化能力下降。尝试减少 `max_steps` 或增加 `warmup_ratio`，并引入 `lora_dropout`。\<br/\>3. **评估指标局限性：** 对于开放式生成任务，BLEU/ROUGE 可能无法完全捕捉生成质量的提升。考虑结合人类评估或基于 LLM 的裁判评估。\<br/\>4. **训练步数不足：** 确保训练步数 (`max_steps`) 足够，但也不要过多。通常 LoRA 训练 3-5 个 epoch 或几百到几千步即可。 |
| **评估结果波动大**                        | 这是由于深度学习训练和生成过程中的随机性。\<br/\>1. **固定所有随机种子：** 确保在模型加载、数据处理、训练和推理的每个阶段都固定随机种子（如 `SEED = 3407`）。\<br/\>2. **CUDA 确定性：** 对于 PyTorch，设置 `torch.backends.cudnn.deterministic=True` 和 `torch.backends.cudnn.benchmark=False` 可以确保 CUDA 操作的确定性，但可能会略微降低性能。\<br/\>3. **重复运行：** 进行多次评估运行，取平均值或中位数以减少单次运行的随机性影响。                                                                                                    |
| **模型出现“幻觉”或不当言论**              | 这是 LLM 的常见问题，微调可能会加剧。\<br/\>1. **数据清洗：** 彻底审查和清洗微调数据，移除任何可能导致“幻觉”或不当言论的错误、不准确或偏见性内容。\<br/\>2. **RLHF/DPO/SPIN：** 考虑使用 **人类反馈强化学习 (RLHF)**、**直接偏好优化 (DPO)** 或 **自播放迭代训练 (SPIN)** 等技术，通过偏好数据对模型进行对齐，使其行为更符合预期。\<br/\>3. **安全提示：** 在推理时使用系统级安全提示或内容过滤器进行后处理。                                                                                                                                |
| **推理速度慢**                            | 除了 `prepare_inference(model)`，还可以：\<br/\>1. 使用 **vLLM** 或 **TGI** 等高性能推理引擎进行部署。\<br/\>2. 确保 `use_cache=True`（生成时默认开启），以复用 KV 缓存。\<br/\>3. 对于极低延迟要求，考虑 Qwen3 的 **非思考模式**。                                                                                                                                                                                                                                                                                                          |

```python
import torch
import random
import numpy as numpy
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# 设置随机种子，确保可复现性

SEED = 3407
random.seed(SEED)
numpy.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =========================================================

# 2️⃣ 载入模型

# =========================================================

# 选择 Unsloth 优化过的 Qwen3 模型，并启用 4-bit 量化

MODEL_NAME = "unsloth/Qwen3-8B-Chat"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True

print(f"正在加载基础模型: {MODEL_NAME} (4-bit 量化: {LOAD_IN_4BIT})...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=LOAD_IN_4BIT,
)
print("模型加载完成。")

# =========================================================

# 3️⃣ 注入 LoRA 适配器

# =========================================================

# 配置 LoRA 参数，r 决定秩，lora_alpha 是缩放因子

# target_modules 指定要注入 LoRA 的层，通常包括 Attention 和 MLP 层的投影

# use_gradient_checkpointing="unsloth" 启用 Unsloth 的显存优化

print("正在注入 LoRA 适配器...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16, # LoRA 的秩，可根据显存和性能需求调整
    lora_alpha=16, # LoRA 缩放因子
    lora_dropout=0.05, # LoRA 层的 dropout 率
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", # Attention 层
        "gate_proj", "up_proj", "down_proj" # MLP 层 (GeLU/SwiGLU 等激活函数的投影)
    ],
    use_gradient_checkpointing="unsloth", # 启用 Unsloth 优化的梯度检查点以节省显存
    random_state=SEED, # 保持随机性一致
)
print("LoRA 适配器注入完成。")

# =========================================================

# 4️⃣ 数据集准备

# =========================================================

# 加载 alpaca-cleaned 数据集的子集

print("正在加载数据集 (yahma/alpaca-cleaned 的前 30%)...")
dataset = load_dataset("yahma/alpaca-cleaned", split="train[:30%]")

# 定义数据格式化函数，将原始数据转换为 Qwen3 ChatML 格式

# 在微调阶段，通常不希望模型自己产生 <think> 标签，所以 enable_thinking=False

def format_example(example):
    user_message = example["instruction"]
    if example["input"].strip():
        user_message += "\n\n" + example["input"]

    messages = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": example["output"]}
    ]

    # 使用 tokenizer 的 apply_chat_template 将对话转换为模型所需的字符串格式
    # tokenize=False 返回字符串，add_generation_prompt=False 确保不额外添加生成提示
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False # 训练时通常不希望模型自己产生思考标签
    )
    return {"text": formatted_text}

print("正在格式化数据集...")

# 使用 map 函数并行处理数据集，num_proc 可根据 CPU 核心数调整

dataset = dataset.map(format_example, num_proc=4)
print(f"数据集格式化完成，首条数据示例:\n{dataset[0]['text']}")

# =========================================================

# 5️⃣ 训练模型

# =========================================================

# 配置训练参数，使用 TRL 库的 SFTTrainer

print("正在配置训练器并开始训练...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text", # 指定数据集中的文本字段名
    max_seq_length=MAX_SEQ_LENGTH, # 训练时序列最大长度
    # packing=True 可以提高长序列训练效率，但会改变每个 batch 的实际长度，如果数据集长度不一，可以尝试
    # packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2, # 每个 GPU 上的批次大小
        gradient_accumulation_steps=8, # 梯度累积步数，有效批次大小为 2 * 8 = 16
        max_steps=100, # 最大训练步数，这里为示例值，实际应根据收敛情况调整
        learning_rate=2e-4, # 学习率
        warmup_ratio=0.03, # 学习率预热比例
        logging_steps=1, # 每多少步记录一次日志
        fp16=not torch.cuda.is_bf16_supported(), # 如果不支持 bfloat16 则使用 float16
        bf16=torch.cuda.is_bf16_supported(), # 优先使用 bfloat16
        optim="adafactor", # 优化器，Adafactor 在显存方面表现良好
        weight_decay=0.01, # 权重衰减
        lr_scheduler_type="cosine", # 学习率调度器类型：余弦退火
        report_to="wandb", # 将训练日志报告到 Weights & Biases
        output_dir="qwen3_lora", # 模型和训练状态的输出目录
        seed=SEED, # 训练过程的随机种子
        # save_steps=50, # 可以设置保存检查点的频率
        # save_total_limit=3, # 最多保存的检查点数量
        # evaluation_strategy="steps", # 可以设置评估策略
        # eval_steps=50, # 评估步数
    ),
)
trainer.train()
print("模型训练完成。")

# =========================================================

# 6️⃣ 推理与保存：微调后模型的测试与部署准备

# =========================================================

from unsloth import prepare_inference

# 准备模型进行推理：Unsloth 会自动将模型切换到 eval 模式并进行推理优化

print("正在准备模型进行推理...")
prepare_inference(model)

# 定义保存路径并保存 LoRA 适配器和分词器

SAVE_DIR = "qwen3_lora_fine_tuned"
print(f"正在保存 LoRA 适配器和分词器到 {SAVE_DIR}...")
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print("模型保存完成。")

# 进行推理测试

print("正在进行推理测试...")
user_query = "1, 1, 2, 3, 5, 8，请继续斐波那契序列"
messages = [{"role": "user", "content": user_query}]

# 应用 Qwen3 的 ChatML 模板，并指定推理时禁用思考模式以获得简洁输出

prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True, # 添加生成提示，告知模型开始生成响应
    enable_thinking=False, # 推理时禁用思考模式
    tokenize=False # 返回字符串，而不是 token ID 列表
)

# 将 prompt 转换为 token ID 并移动到 GPU

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")["input_ids"]

# 模型生成响应

outputs = model.generate(
    inputs,
    max_new_tokens=20, # 最大生成 token 数
    do_sample=True, # 启用采样生成，使输出更具多样性
    temperature=0.7, # 采样温度
    top_p=0.8, # Top-p 采样
    top_k=20, # Top-k 采样
    min_p=0.0, # 最小概率阈值
    use_cache=False, # 是否使用 KV 缓存（通常推理时建议开启以加速，这里为示例简洁设为 False）
)

# 解码生成的 token 并打印结果

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n=========== 微调模型推理输出 ============")
print(generated_text)

# 清理 GPU 缓存，准备重新加载模型以验证保存

del model
del tokenizer
torch.cuda.empty_cache()

# =========================================================

# 7️⃣ 重新加载与推理：验证保存的模型是否可用

# =========================================================

print(f"\n 正在从 {SAVE_DIR} 重新加载模型以验证...")

# 从保存的路径重新加载模型和分词器

reloaded_model, reloaded_tokenizer = FastLanguageModel.from_pretrained(
    model_name=SAVE_DIR,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=LOAD_IN_4BIT,
)

# 再次准备推理模式

prepare_inference(reloaded_model)

# 使用重新加载的模型进行推理

reloaded_inputs = reloaded_tokenizer(prompt, return_tensors="pt").to("cuda")["input_ids"]
reloaded_outputs = reloaded_model.generate(
    reloaded_inputs,
    max_new_tokens=20,
    do_sample=True,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    min_p=0.0,
    use_cache=False,
)
reloaded_generated_text = reloaded_tokenizer.decode(reloaded_outputs[0], skip_special_tokens=True)
print("\n=========== 重新加载模型推理输出 ===========")
print(reloaded_generated_text)
print("模型重新加载并推理验证完成。")
```

---

## 10\. 结论与未来方向

2025 年，Qwen3 模型以其**开放许可证、独特的动态思考模式、卓越的多语言支持和超长的上下文处理能力**，在开源大型语言模型领域中脱颖而出，成为开发者和企业构建下一代 AI 应用的强大基石。

**Unsloth Dynamic 2.0** 的引入，结合 **QLoRA/LoRA** 等参数高效微调技术，彻底改变了消费级 GPU 上的大模型定制体验。它使得即使是拥有有限硬件资源的个人开发者，也能够高效地对 Qwen3 进行深度定制，将大型语言模型的能力普及到更广泛的应用场景。

**W\&B Weave** 则为模型评估和实验跟踪提供了一个直观、可协作且高度可视化的工作流。它不仅能帮助我们量化微调效果，更能通过并排对比、错误分析等功能，深入理解模型行为，加速迭代优化。

结合 **vLLM、Ollama** 等高性能推理框架，可以实现从实验到生产部署的快速跃迁，将微调后的 Qwen3 模型高效地服务于实际应用。

展望未来，随着 LLM 技术和 Unsloth 等工具的不断演进，我们可以期待：

- **多卡 DDP (Distributed Data Parallel) 微调的进一步优化：** 使更大规模的 Qwen3 模型微调在集群环境中更易于部署和管理。
- **混合专家 (MoE) 模型的更高效微调：** Qwen3 MoE 模型拥有巨大的潜力，未来微调技术将更好地利用其稀疏激活特性，实现更低的成本和更高的性能。
- **多模态 Vision 适配的深度集成：** 随着多模态 LLM 的兴起，Qwen3 与视觉、音频等模态的融合将成为新的热点，Unsloth 等工具也将适配这些新的微调范式。

---

## 11\. 进一步阅读

- **Unsloth 官方“Run & Fine-tune Qwen3”教程：** [docs.unsloth.ai](https://www.google.com/search?q=https://docs.unsloth.ai/quickstart/qwen/) (请注意，官方文档会持续更新，请查看最新版本)
- **Qwen3 技术报告：** [arxiv.org](https://arxiv.org/abs/2405.07194) (参考最新的 arXiv 预印本，通常包含模型架构和性能的详细信息)
- **Unsloth GitHub 仓库：** [github.com/unslothai/unsloth](https://github.com/unslothai/unsloth) (关注 Release 和 Issues 以获取最新性能和 API 更新日志)
- **vLLM 官方文档：** [docs.vllm.ai](https://docs.vllm.ai/)
- **Ollama 官方网站：** [ollama.com](https://ollama.com/)
- **Weights & Biases Weave 官方文档：** [wandb.ai/site/solutions/llm-fine-tuning](https://wandb.ai/site/solutions/llm-fine-tuning)

---
