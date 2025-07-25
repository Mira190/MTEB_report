# 调研报告：MTEB AILAStatutes 法律检索任务模型微调方案

## 0. 引言

本报告旨在针对 MTEB 榜单中的 AILAStatutes 法律检索子任务，提供系统化的调研与实施方案。内容涵盖项目目标与任务定义、技术路径与模型选择、资源规划与时间安排、评估与提交流程等。方案特别关注硬件资源的适配与部署策略，以期在限定时间内高效完成模型微调并成功提交至 Hugging Face Hub，从而在 MTEB 排行榜上获得展示,需注意模型权重、训练与评估脚本、数据处理流程及技术报告均须**开源发布**。

## 1. 项目目标与任务选择

### 1.1 任务背景与核心目标

- **任务背景**：AILAStatutes 检索任务要求在给定法律查询（Query）的情况下，从法规条文集合中检索最相关条文。该任务具有典型法律文本语义匹配需求，评价指标常为 nDCG@k、MAP@k 等。
- **核心目标**：在 8 周冲刺期内完成 AILAStatutes 检索模型的微调，并将结果上传至 Hugging Face Hub，使其在 MTEB 排行榜中自动评估并展示。
- **阶段性目标**：
  1. **阶段一**：完成数据获取与预处理流水线，验证可用基线模型与环境配置；
  2. **阶段二**：基于初步模型进行小规模微调实验，选择最优技术路径；
  3. **阶段三**：利用高性能算力完成大规模训练与性能优化；
  4. **阶段四**：在本地完成全面评估，生成并整合元数据，提交模型并撰写技术报告。
- **成功判据**：
  - MTEB 官方评估流程顺利运行并生成榜单条目；
  - 模型仓库公开且文档完整，可复现性得到保证；
  - 微调脚本、评估脚本、模型卡片等交付物齐备且符合规范。

### 1.2 任务范围与优先级

- **专注方向**：仅限检索（Retrieval）类别下的法律领域子任务 AILAStatutes。
- **优先级说明**：首期不强求最高名次，而是确保方案落地、结果可复现；如在中期获得稳定基线后，可再针对指标提升进行额外优化或探索更大模型。

## 2. 技术方案概述

### 2.1 基础模型选择与对比

本方案将以下几类模型视为候选基座，依据执行效率、硬件可行性和领域适配度进行对比与微调。

1. **Qwen 3-Embedding**

   - 说明：阿里巴巴开源的文本嵌入模型，在多语言及多任务评测中表现优异，兼具通用语义理解能力。
   - 优势：在 MTEB 各类任务上已有领先记录；嵌入生成速度与质量兼备；许可证（Apache-2.0）满足公开要求。
   - 资源需求：0.6B 版本可在 32 GB 显存 GPU 上完成微调（FP16/4-bit QLoRA）；更大版本在 A100/H100 上微调时显存足够。
   - 适配性：原生 Embedding 架构，直接兼容双编码检索流程，无需额外架构改造；可作为首选基座进行对比微调。
   - 风险与对策：法律领域语料覆盖有限，需通过少量对比学习微调补齐领域特定能力；关注数据稀缺导致过拟合风险。

2. **通用 Sentence Transformers 模型**

   - 典型示例：`all-MiniLM-L6-v2`、`multi-qa-mpnet-base-dot-v1` 等。
   - 优势：模型体积较小，推理与微调成本低；易于快速验证；社区与示例丰富。
   - 资源需求：在 RTX 5090 等消费级 GPU 上即可完成全参或 LoRA 微调。
   - 适配性：通用语义嵌入能力较强，但法律领域用例需微调以提升针对性。

3. **法律领域特定模型**

   - 典型示例：Legal-BERT（或类似在法律语料上预训练的 Encoder 模型）。
   - 优势：在法律术语及上下文理解方面具备先验；可在检索任务中更快适配领域特征。
   - 资源需求：与通用 BERT 相当；可在 RTX 5090 上进行 LoRA 微调。
   - 适配性：需将其转换为双编码嵌入生成方式（如取 [CLS] 池化或其它池化策略）；也可作为下游 reranker 或 cross-encoder 使用，但计算成本较高。

4. **小型开源 LLM**
   - 典型示例：Gemma-2B、Mistral-7B 等，通过 QLoRA 进行微调。
   - 优势：除嵌入外，可探索生成式检索（如生成相关法条摘要或解释），但生成部分对纯检索指标帮助有限。
   - 资源需求：4-bit QLoRA 在 RTX 5090 或 A100 上可行；但相较纯嵌入模型，训练与推理成本更高。
   - 适配性：主要用于探索性实验，不推荐作为首选基座；若后期需二阶段 rerank 或生成解释，可结合嵌入模型使用。

### 2.2 微调策略

- **对比学习（Contrastive Learning）**

  - 核心：根据查询-相关文档对构建正负样本，使用 MultipleNegativesRankingLoss、Triplet Loss 或 InfoNCE 等损失，引导模型在嵌入空间中拉近正例、推远负例。
  - 负样本构造：包括随机负样本和基于 BM25 或当前嵌入模型检索结果的硬负样本挖掘；针对小数据集，动态在线挖掘有助提升判别能力。
  - 注意事项：需平衡正负样本数量，避免过度拟合；可结合多任务或软标签策略，以维持通用嵌入能力。

- **参数高效微调（PEFT / QLoRA）**

  - 目的：大规模模型（如 Qwen 3-Embedding 更大版本或 LLM）在常规微调下显存消耗高。PEFT 方法仅微调少量低秩权重，结合量化（4-bit、8-bit）可显著降低显存和计算成本，同时保持较高性能。
  - 实践：对 Qwen 3-Embedding、Sentence Transformers、LLM 均可应用 LoRA/QLoRA；在 RTX 5090 上验证小版本，对 A100/H100 进行更大版本微调。
  - 超参数：LoRA rank、alpha、dropout、量化位宽需根据显存和性能目标调节；训练时应监控 GPU 使用情况与效果提升幅度，灵活调整微调深度。

- **指令微调（Instruction Fine-tuning）**

  - 说明：将检索任务以自然语言指令形式输入大语言模型，使其在生成式模式下“生成”相关条文标识或内容。
  - 适用性：对纯检索指标帮助有限，可能用于生成解释或辅助检索，但单纯生成不等同于深度向量匹配，且成本较高。
  - 建议：可作为辅助实验，但不作为主线策略；主要依赖对比学习与嵌入模型微调。

- **二阶段流程（Retrieval + Rerank / 生成说明）**
  - 第一阶段：基于双编码嵌入模型进行快速检索，生成候选列表（top-k）。
  - 第二阶段（可选）：使用 cross-encoder 或简易生成模型对候选列表进行 rerank 或提供解释。此阶段计算成本高，仅在主检索性能稳定后考虑。

### 2.3 数据准备与处理

- **数据集获取**

  - 来源：MTEB 官方或 Hugging Face Datasets 中的 AILAStatutes 数据；若官方未直接提供，应从 Zenodo 或相关论文附录下载原始标注文件（queries.jsonl、statutes.jsonl、relevance.jsonl）。
  - 许可：务必核实数据使用许可，确保符合公开发布要求。

- **数据清洗与预处理**

  - 文本规范化：统一编码（UTF-8）；去除页眉页脚、特殊符号、HTML 标签或其他非文本内容；处理长文档可按条款或段落切分。
  - 分词与编码：根据所选模型使用对应分词器进行编码；注意处理最大长度限制（如截断或分段策略），尤其是长法律条文。
  - 样本构建：
    - **正样本对**：利用 relevance 标注，将查询与相关法条文本配对。
    - **负样本生成**：可采用随机选择不相关法条，也可基于 BM25 或当前模型检索结果选取难负；对于数据稀缺场景，在线硬负挖掘更为重要。
    - **训练样本格式**：对比学习框架通常需要 `(query, pos_doc, neg_doc)` 形式；Sentence Transformers 使用 InputExample(texts=[query, pos_doc], label=1.0) 并在批处理中自动引入其他样本作为负例。

- **数据流水线可复现性**
  - 将数据处理脚本纳入版本管理，使用配置文件或命令行参数控制预处理流程（如路径、清洗选项、负样本策略等），以便他人复现或调整。
  - 保留原始数据副本及中间处理产物，或记录处理日志/摘要，确保可追溯。

### 2.4 评估与提交流程

- **本地评估**

  - 使用 MTEB Python 库进行评估，确保评估流程与排行榜一致。
  - 在本地评估前，应先在小规模子集或部分任务上进行快速验证，以估算资源消耗与时间。
  - 完整评估时，指定输出文件夹，保存每个子任务的 JSON 结果文件。

- **生成元数据**

  - 通过 `mteb create_meta` 脚本自动生成 YAML 头部及结果表格/图表。
  - 将生成片段与已有 README 合并，确保模型卡片顶部包含 MTEB 结果元数据字段（language、license、model_name、mteb_results 等）。

- **提交至 Hugging Face Hub**
  - 确保仓库公开且包含权重、配置、分词器、推理/示例代码、更新后的 README。
  - 推送更新后，等待 MTEB 排行榜刷新；如未出现，检查 YAML 格式及路径正确性。

## 3. 项目要求与约束

- **公开性与可复现性**：模型权重、训练与评估脚本、数据处理流程及技术报告均须**开源发布**。
- **合规性**：遵守 AILA 数据许可、预训练模型许可及 MTEB 排行榜规则。
- **资源限制**：在初期优先使用可控硬件（如 RTX 5090）；中后期需申请或租用 A100/H100 等高性能资源；结合 PEFT 方法降低显存需求。
- **项目风险**：
  - 数据稀缺导致过拟合：需采用丰富负样本策略或多任务/软标签微调；
  - 过度微调通用模型可能削弱泛化能力：需在验证集上反复测试并保留基线能力；
  - 生成式方法成本高且收益有限：优先聚焦双编码检索；
  - 硬件调度与突发故障：需建立检查点机制和备份策略。

## 4. 资源规划

### 4.1 人员安排与角色分工

- **核心团队规模**：3–4 人，外加 1 名技术指导。
- **成员挑选标准**：具备 LLM/RAG 基础理解，熟练 Python 与深度学习框架，熟悉 Linux/GPU 操作，自驱与快速学习能力；加分项包括 Hugging Face 生态经验、法律 NLP 背景、竞赛或榜单参与经验、英文阅读能力。
- **角色与职责**：
  - **项目负责人（Project Lead）**（1 人）：总体规划、进度监督、资源协调（GPU 申请）、风险管理、对外沟通与最终提交审核。
  - **数据工程师/预处理专家（Data Engineer）**（1 人）：数据获取、解析、清洗、样本构建与负样本生成、流水线可复现性保障。
  - **机器学习工程师（ML Engineer）**（1–2 人）：基础模型选型与加载、微调方法实现（对比学习、PEFT）、训练调试与评估、性能优化。
  - **基础设施与工具支持（Infra & Tools）**（可由 ML/Data 工程师兼任）：环境搭建与维护、MTEB 评估工具集成、Hugging Face Hub 提交流程、版本控制管理、CI/CD（如适用）。
  - **技术指导（Technical Advisor）**（1 人，外部）：提供大模型微调、RAG、法律 NLP 等领域的高层次咨询、经验分享与代码审查。

### 4.2 硬件资源与算力部署

- **RTX 5090（32 GB VRAM）**
  - 适用于初期探索、小规模微调（Sentence Transformers、Qwen 3-Embedding 小版本、7B LLM 4-bit QLoRA 原型）。
  - 用于环境搭建、快速实验验证、负样本策略测试等。
- **NVIDIA A100（40GB/80GB VRAM）**
  - 适用于中期大规模训练、全参数或 PEFT 微调更大模型、长轮次收敛实验。
  - 建议在小版本原型结果稳定后申请用于主力模型训练。
- **NVIDIA H100（80GB VRAM）**
  - 用于后期极限性能优化、超大模型微调、复杂知识蒸馏或大规模多任务联合训练。
  - 若团队有对应资源，可在冲刺阶段使用以提升最终指标。
- **国内云 GPU**
  - 作为备选或补充，需关注数据传输成本、网络稳定性与存储方案。
- **辅助资源**：高速存储（SSD/NVMe）、充足主机内存（≥64 GB）、网络带宽与备份机制。

## 5. 项目时间线（8 周冲刺）

### 5.1 阶段划分

- **阶段一：调研与数据准备（第 1–2 周）**

  - 深入理解 AILAStatutes 数据集结构、格式与许可；
  - 完成数据下载、文本清洗与预处理脚本；
  - 确定候选基础模型列表（Sentence Transformers、Legal-BERT、Qwen 3-Embedding、小型 LLM）；
  - 在 RTX 5090 上搭建环境，验证基础加载与推理流程。

- **阶段二：模型选型与初步微调（第 3–4.5 周，约 2.5 周）**

  - 利用 RTX 5090 对 Qwen 3-Embedding（0.6B 版本）、通用 SBERT、小型 LLM(7B) 进行 LoRA/QLoRA 原型微调；
  - 构建对比学习训练样本，测试随机与硬负样本策略；
  - 分析开发集评估结果，选择最有潜力的模型与微调方案；
  - 如基线效果较差，可适度探索数据扩充或域预训练思路。

- **阶段三：大规模训练与优化（第 4.5–7 周，约 2.5 周）**

  - 根据阶段二结果，在 A100/H100 上对选定模型（如 Qwen 3-Embedding 更大版本或 Legal-BERT）进行更长轮次或更大批次微调；
  - 探索更复杂的负样本挖掘、数据增强或多任务联合微调；
  - 监控训练曲线与评估指标，持续优化超参数；
  - 如需二阶段 rerank，可使用 cross-encoder 在候选集上微调；但应优先保证首阶段嵌入检索质量。

- **阶段四：评估、提交与文档撰写（第 7–8 周）**
  - 在本地完成对最终模型的全面 MTEB 评估，保存所有子任务结果；
  - 运行 `mteb create_meta` 生成元数据片段，整合至 Hugging Face 仓库 README；
  - 确保模型仓库公开，推送更新并验证排行榜刷新情况；
  - 撰写技术报告与附录，公开数据处理细节、模型微调策略、实验结果与分析。

### 5.2 每周工作节奏

- 每日简短站会（10–15 分钟），跟进进度与问题；
- 周中汇报实验结果、讨论瓶颈与调整方案；
- 每周末进行回顾，总结已完成工作、存在风险与下阶段计划；
- 对关键实验（如大规模训练）提前预估资源需求并预留时间，以应对可能重试或故障。
