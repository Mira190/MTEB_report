# 嵌入模型融合在法律信息检索中的应用：以 MTEB AILAStatutes 任务为例的创新策略研究

## 引言

在大型语言模型（LLM）和人工智能快速发展的背景下，嵌入模型已成为自然语言处理（NLP）领域不可或缺的组成部分，尤其在信息检索任务中发挥核心作用。嵌入模型通过将文本转换为高维向量表示，从而能够捕捉语义含义并促进高效的相似性搜索。然而，单一的预训练或微调嵌入模型往往难以在所有任务或领域中达到最优性能，特别是在处理法律文本这种高度专业化、术语复杂且语境微妙的领域时。

模型融合作为一种新兴且具有变革性的策略，旨在将两个或更多预训练模型的参数整合到一个统一的模型中，从而在不进行额外训练的情况下保留并结合原始模型的优势和能力。此方法不仅能创建多任务模型、支持持续学习，还能有效减少灾难性遗忘，同时保持与单一模型相同的推理成本，甚至实现更优异的性能。本研究重点探讨嵌入模型融合提升法律信息检索能力的方式，围绕 MergeKit 工具展开，提出两种创新策略，并在 MTEB AILAStatutes 任务上进行实验验证。

---

## 嵌入模型融合理论基础

模型融合的核心在于以各种方式组合模型参数，这些参数通常被视为高维空间中的“任务向量”，代表模型从通用基础到特定任务微调过程中参数的变化。

### SLERP（球面线性插值）及其优势

球面线性插值（SLERP）是一种在两个模型检查点之间进行参数插值的技术，是对简单线性插值（LERP）的扩展。与 LERP 在欧几里得空间中沿直线混合参数不同，SLERP 利用超球面几何，沿测地线生成平滑路径。

**公式**  
对于单位向量 \(q_0\) 和 \(q_1\)，SLERP 定义为：  
S(t,q0​,q1​)=sin(θ)sin((1−t)θ)​q0​+sin(θ)sin(tθ)​q1​

其中，$t \in $ 是插值因子，$\theta$ 是 q0​ 和 q1​ 之间作为 4 维向量（或更高维）的角度，由 cosθ=q0​⋅q1​ 确定

**优势**

- 在超球面上沿最短路径插值，通常避开高损失区域
- 更好地保留父模型特性，实现更准确、更稳定的融合

### 检查点（Checkpoint）概念

检查点指训练或微调过程中在特定阶段保存的模型参数状态。这些检查点承载了模型在不同任务上的知识。融合技术正是通过合并多个检查点参数来构建新模型。

### TIES（修剪、选择符号与合并）

TIES 方法的核心在于利用“任务向量” τt​=θtft​−θinit​，其中 θtft​ 是微调后的模型参数，θinit​ 是基础模型参数。这些任务向量捕获了每个任务在参数空间中所需的定向调整

1. 修剪 (Trimming)： 对每个任务向量 τt​ 进行修剪，只保留具有最高幅度的前 k 个最重要参数，将其余参数设为零，从而得到 τ^t​ 。这有助于减少冗余信息和噪声。
2. 选择符号 (Electing Sign)： 对于每个参数，根据所有修剪后的任务向量中正负号的总幅度，选择一个主导方向 γm​。这解决了不同模型对同一参数更新方向可能存在的冲突。
3. 不相交合并 (Disjoint Merging)： 只有那些与所选主导方向一致的参数 θm​ 才会被平均。最终的模型参数 θm​ 由 θm​=θinit​+λ⋅τm​ 给出，其中 λ 是一个缩放因子。

**优势**

- 在多任务、多架构场景下具有更高稳定性
- 关键参数选择优于随机选择，尤其在 k 较小时性能提升显著

### DARE（丢弃与重缩放）

DARE（Drop And REscale）通过减少 delta 参数的冗余，实现更高效的融合。

核心思想与数学公式：
DARE 算法主要包括三个步骤：

1. Delta 参数计算： 计算微调模型权重 WSFT_t​ 与预训练模型权重 WPRE​ 之间的差异，得到 delta 参数 Δt​ 。
2. 随机丢弃与重缩放： 随机将这些 delta 参数中的一部分 p 设为零。然后，将剩余的非零 delta 参数按 1/(1−p) 的因子进行重缩放，以弥补被丢弃的值 。
3. 加回预训练权重： 最后，将重缩放后的 delta 参数 Δ^t​ 加回到原始预训练权重 WPRE​ 中，得到 DARE 适应后的权重 WDARE_t​。

优势：
DARE 的主要优势在于其能够减少 delta 参数中的冗余，从而实现更高效的模型融合。DARE 可以与其他合并算法结合使用，例如
dare_ties 结合了 DARE 和 TIES 的符号共识算法，而 dare_linear 则不使用符号共识 。这种方法有助于在合并模型时有效保留技能 。

---

## MergeKit：模型融合工具

MergeKit 是一款开源“核外”融合工具，支持在有限资源（如 8 GB 显存）环境中执行复杂合并操作，且融合后模型保持与单模型相同的推理成本。

### 主要功能与配置元素

- **merge_method**：选择合并方法（linear、slerp、ties、dare_ties 等）
- **models / slices**：定义全模型或按层切片合并
- **base_model**：任务向量计算的基础模型
- **parameters**：设置权重、density、lambda 等
- **dtype**：指定数据类型（如 bfloat16）
- **tokenizer / tokenizer_source**：控制词表合并策略

#### Token Embeddings 处理

1. Token 存在于基础模型：使用基础嵌入
2. 仅一个模型拥有：使用该模型嵌入
3. 多模型拥有：取平均值

用户可通过 `tokenizer` 字段细粒度控制特定 token 的来源，确保兼容不同模型提示格式。

---

## MTEB AILAStatutes 任务分析

MTEB（Massive Text Embedding Benchmark）是综合基准测试，评估嵌入模型在多种任务上的表现。AILAStatutes 专注法律文本检索。

### 任务描述与数据集特点

- **任务类型**：文档检索
- **语言**：英语
- **记录总数**：349 条
- **测试样本**：132 条，涉及 82 个文档、50 个查询
- **文档长度**：平均 1,975 字符，最长 26,039 字符
- **查询长度**：平均 3,038 字符，最长 5,936 字符

处理长文本与法律术语细微语义是关键挑战。

### 评估指标与 SOTA 模型

- **指标**：召回率（Recall）、平均精度（MAP）
- **参考模型**：voyage-lite-02-instruct、e5-mistral-7b-instruct、bge-large-en-v1.5 等
- **领域模型**：LEGAL-BERT、CaseLawBERT、ColBERT 等

---

## 创新融合策略与实验设计

本研究提出两种利用 MergeKit 的创新融合策略，在 AILAStatutes 任务上提升检索性能。

### 策略一：分层 SLERP 融合 (Layer-Specific SLERP Fusion, LSSF)

**概念**  
对不同层或层组应用差异化 SLERP 插值因子，精细混合早期词法/句法特征与后期语义/上下文能力。

**实验步骤**

1. 选取通用模型：BAAI/bge-large-en-v1.5
2. 选取法律模型：nlpaueb/legal-bert-base-uncased
3. 在 MergeKit 中配置：
   - `merge_method: slerp`
   - `base_model: BAAI/bge-large-en-v1.5`
   - 前 16 层使用 `t=0.7` （self_attn），后 16 层使用 `t=0.3` （mlp）
   - `dtype: bfloat16`
   - `tokenizer: union`

**示例 YAML**

```yaml
merge_method: slerp
base_model: BAAI/bge-large-en-v1.5
slices:
  - sources:
      - model: BAAI/bge-large-en-v1.5
        layer_range: [0, 15]
      - model: nlpaueb/legal-bert-base-uncased
        layer_range: [0, 15]
    parameters:
      t: 0.7
      filter: self_attn
  - sources:
      - model: BAAI/bge-large-en-v1.5
        layer_range: [16, 31]
      - model: nlpaueb/legal-bert-base-uncased
        layer_range: [16, 31]
    parameters:
      t: 0.3
      filter: mlp
parameters:
  t: 0.5
dtype: bfloat16
tokenizer:
  source: union
```

### 策略二：多模型集成融合 (Multi-Model Ensemble Fusion, MMEF)

**概念**
融合三个或更多模型（通用、法律 BERT、长上下文），利用 TIES/DARE 处理参数冲突。

**实验步骤**

1. 通用模型：BAAI/bge-large-en-v1.5
2. 法律模型：nlpaueb/legal-bert-base-uncased
3. 长上下文模型：Salesforce/SFR-Embedding-2_R 或 Lawformer
4. MergeKit 配置：

   - `merge_method: ties` 或 `dare_ties`
   - `base_model: BAAI/bge-large-en-v1.5`
   - 为各模型分配 `weight`，设置 `density`、`lambda`
   - `dtype: bfloat16`
   - `tokenizer: union`

**示例 YAML**

```yaml
merge_method: ties
base_model: BAAI/bge-large-en-v1.5
models:
  - model: BAAI/bge-large-en-v1.5
    parameters: { weight: 0.4 }
  - model: nlpaueb/legal-bert-base-uncased
    parameters: { weight: 0.35 }
  - model: Salesforce/SFR-Embedding-2_R
    parameters: { weight: 0.25 }
parameters:
  density: 0.6
  lambda: 1.0
dtype: bfloat16
tokenizer:
  source: union
```

---

## 实验评估与结果分析

### 评估流程

1. 安装依赖：`mteb`、`mergekit`
2. 生成融合模型：

   ```bash
   git clone https://github.com/arcee-ai/mergekit.git
   cd mergekit
   pip install -e .
   mergekit-yaml config.yaml ./merged_model \
     --copy-tokenizer --allow-crimes
   ```

3. 加载与评估：

   ```python
   import mteb
   from sentence_transformers import SentenceTransformer

   model = SentenceTransformer("./merged_model")
   evaluator = mteb.MTEB(mteb.get_tasks())
   results = evaluator.run(model)
   print(results)
   ```

### 预期结果与局限性

- **LSSF**：提升对法律术语与复杂语义的平衡捕捉
- **MMEF**：增强长文档与复杂查询处理能力
- **局限**：

  - 参数调优耗时
  - 模型架构兼容性要求高
  - 融合增益存在上限

---

## 结论与未来展望

本研究提出了分层 SLERP 融合（LSSF）与多模型集成融合（MMEF）两种策略，并设计了完整实验流程，以期在 MTEB AILAStatutes 任务中获得显著性能提升。未来可从以下方向继续探索：

- **自动化参数优化**：利用进化算法或元学习技术减少手动调优
- **领域数据扩展**：整合法律子领域更多高质量语料
- **新融合算法**：探索跨架构或多模态融合方法
- **多模态检索**：结合文本、图像、音频等多模态法律信息

通过持续创新，嵌入模型融合技术有望在法律人工智能领域开启新可能，为构建更智能、高效的法律信息系统奠定基础。
