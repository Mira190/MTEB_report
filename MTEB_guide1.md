# MTEB Retrieval 任务两阶段优化调研

## 1. 任务概述与优化策略

### 1.1 任务描述

本文档针对 MTEB (Massive Text Embedding Benchmark) 中的 **Retrieval（检索）** 任务，提出了一套完整的两阶段优化微调方案：

- **任务类型**：文本检索 (Text Retrieval)
- **目标语言**：英语 (English, `eng`)
- **涉及领域**：法律 (Legal)、书面语 (Written)
- **评估指标**：NDCG@10 (Normalized Discounted Cumulative Gain at 10)
- **模态**：文本 (Text)

### 1.2 两阶段优化策略

**阶段一：对比学习微调（语义空间优化）**

- **目标**：让模型学习到高质量的语义嵌入，使相似文本在向量空间中距离相近
- **方法**：使用 InfoNCE Loss 或 MultipleNegativesRankingLoss，结合精心构建的正负样本
- **产物**：领域内表现良好的文本嵌入模型

**阶段二：强化学习优化（排名优化）**

- **目标**：在嵌入模型基础上，通过 RL 进一步优化检索结果的排序
- **方法**：使用策略梯度算法（如 PPO）训练重排序器
- **产物**：能够生成更优检索排序的优化模型

### 1.3 策略优势

- **降低复杂性**：避免直接用 RL 训练大型 Embedding 模型的困难
- **提高数据效率**：对比学习使用大量无监督数据，RL 使用少量精细奖励数据
- **兼容性强**：预训练的 Embedding 可作为 RL 模块的特征提取器

---

## 2. 阶段一：对比学习微调

### 2.1 背景技术

对比学习是当前提升嵌入模型性能的有效方法，其核心思想是**拉近正样本对（语义相似）的距离，推远负样本对（语义不相似）的距离**。

#### 最新研究成果

- **金融领域**：Dolphin 等人（2024）研究表明，对比学习在金融资产嵌入上显著提升了 F1-score
- **小型语言模型**：Trapoom Ukarapol 等人（2024）针对小型 LM 的对比微调，在 STS 基准上平均性能提升 56.33%
- **内部测试**：微调后的嵌入模型在知识库实体链接准确率提升约 35%，语义检索 Recall@5 指标提升约 42%

### 2.2 数据集准备

#### 2.2.1 原始语料收集

针对"法律"和"书面语"领域，收集大规模文本语料库：

- 法律判例、法律法规、法律新闻、法律论文
- 官方文档、报告、新闻报道、学术论文等书面文本
- 使用 PDF 解析工具（如`unstructured`库）从现有文档中提取文本

#### 2.2.2 正负样本构建

**正样本对构建**：

- **手工标注**：由法律专家挑选语义高度相关的文本对
- **自动聚类**：使用现有嵌入模型对原始语料进行聚类，同簇内文本视作潜在正样本对
- **问答系统**：用户问题(Query)与对应正确答案(Answer)形成正样本对

**负样本对构建（重点）**：

1. **随机采样**：从不同主题簇中随机抽取文本对
2. **"难负样本"挖掘**：
   - **基于相似度挖掘**：使用预训练嵌入模型计算相似度，选择相似度较高但语义不相关的样本
   - **具体流程**：
     - 计算查询与语料库的相似度，获取 Top-K 候选
     - 排除正样本，应用筛选条件（`absolute_margin`、`relative_margin`等）
     - 使用`sentence_transformer`库中的`mine_hard_negatives`函数
   - **LLM 辅助生成**：使用 GPT-4o 或 Qwen-Max 等 LLM API 生成语义相似但不相关的负样本

#### 2.2.3 数据格式

```json
{
  "query": "绿色对老年人有哪些健康益处？",
  "response": "绿色环境可以帮助老年人放松紧张的中枢神经，改善和调节身体功能...",
  "rejected_response": [
    "绿茶的主要功效是预防癌症和心血管疾病...",
    "绿茶水洗脸可以防止肌肤衰老..."
  ]
}
```

### 2.3 模型与工具选择

#### 2.3.1 基础嵌入模型

- **Sentence-Transformers 系列**：`sentence-transformers/all-MiniLM-L6-v2`或`BAAI/bge-base-en-v1.5`
- **Qwen3 Embedding 系列**（推荐）：`Qwen3-Embedding-0.6B`或`1.8B`版本

#### 2.3.2 微调库

- **Sentence-Transformers**：功能强大且易于使用
- **MS-Swift**（推荐用于 Qwen）：支持 LoRA 等高效微调方法

#### 2.3.3 损失函数

**InfoNCE Loss**：

- 适用于大规模无标签或弱监督数据
- 在每个 Batch 中，将正样本对视作类内同类，其他对作为负样本

**MultipleNegativesRankingLoss**（推荐）：

- 与三元组（Anchor, Positive, Negative）格式数据配合使用
- 旨在最大化正样本对相似度，最小化负样本对相似度

**Supervised Contrastive Loss**：

- 适用于有标签的"难例样本"
- 将难例样本的类内样本拉近，类间样本推远

### 2.4 训练流程

#### 2.4.1 环境配置

```bash
pip install sentence-transformers torch torchvision faiss-cpu datasets
# 使用ms-swift微调Qwen3
pip install ms-swift
```

#### 2.4.2 数据加载与处理

```python
from sentence_transformers import SentenceTransformer, InputExample
from torch.utils.data import DataLoader
from datasets import load_dataset

# 加载预训练模型
model = SentenceTransformer('BAAI/bge-base-en-v1.5')

# 转换数据格式
examples = []
for item in dataset['train']:
    examples.append(InputExample(texts=[item['query'], item['positive']], label=1.0))
    if 'negative' in item and item['negative']:
        if isinstance(item['negative'], list):
            for neg_text in item['negative']:
                examples.append(InputExample(texts=[item['query'], neg_text], label=0.0))

train_loader = DataLoader(examples, shuffle=True, batch_size=32)
```

#### 2.4.3 损失函数定义

```python
from sentence_transformers.losses import MultipleNegativesRankingLoss

# 创建损失函数
train_examples_mnr = []
for item in dataset['train']:
    train_examples_mnr.append(InputExample(texts=[item['anchor'], item['positive']]))

train_dataloader_mnr = DataLoader(train_examples_mnr, shuffle=True, batch_size=32)
mnr_loss = MultipleNegativesRankingLoss(model=model)
```

#### 2.4.4 训练配置

```python
from sentence_transformers import SentenceTransformerTrainingArguments, SentenceTransformerTrainer

args = SentenceTransformerTrainingArguments(
    output_dir="bge-finetuned",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    per_device_eval_batch_size=16,
    warmup_ratio=0.1,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    optim="adamw_torch_fused",
    tf32=True,
    bf16=True,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_dim_128_cosine_ndcg@10",
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    loss=mnr_loss,
    evaluator=evaluator,
)

trainer.train()
trainer.save_model()
```

#### 2.4.5 Qwen3 微调示例

```bash
swift sft \
  --model /path/to/Qwen3-Embedding-0.6B \
  --task_type embedding \
  --model_type qwen3_emb \
  --train_type lora \
  --dataset /path/to/qwen3_emb.json \
  --split_dataset_ratio 0.05 \
  --eval_strategy steps \
  --output_dir output \
  --eval_steps 100 \
  --num_train_epochs 1 \
  --save_steps 100 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 6e-6 \
  --loss_type infonce \
  --label_names labels \
  --dataloader_drop_last true
```

---

## 3. 阶段二：强化学习优化

### 3.1 RL 目标：优化重排序（Reranking）

**核心思想**：训练一个独立的重排序模型，或在 Embedding 模型之上添加轻量级排序层，通过 RL 优化排序决策。

**优势**：

- **环境简化**：RL 的"状态"和"动作"空间大大缩小
- **计算资源需求低**：重排序模型通常比 Embedding 模型小
- **效果直接**：重排序能直接影响 NDCG@10 等指标

### 3.2 RL 关键要素设计

#### 3.2.1 状态 (State)

- 用户查询的嵌入向量
- 召回的 Top-K 文档的嵌入向量列表
- 查询与每个文档的初始相似度分数

#### 3.2.2 动作 (Action)

- **方案 1**：选择下一个要放置在排名列表中的文档
- **方案 2**：对 Top-K 文档进行两两比较并交换位置
- **方案 3**（推荐）：直接输出 Top-K 文档的排序分数

#### 3.2.3 奖励 (Reward)

**开箱即用方案：LLM 作为奖励模型 (RLAIF)**

**操作流程**：

1. **准备数据**：针对查询生成不同的检索结果列表
2. **LLM 评估 Prompt**：

   ```
   你是一个专业的法律文档评估助手。
   用户查询是："[用户查询]"
   以下是一个检索到的文档列表，请评估其对用户查询的相关性、信息丰富度和排序合理性，并给出总分（0-100分）。

   文档列表：
   1. [文档1内容摘要]
   2. [文档2内容摘要]
   ...
   10. [文档10内容摘要]

   评分：[总分]
   解释：[简要解释打分理由]
   ```

3. **收集奖励数据**：批量调用 LLM API，收集(查询, 检索列表, 奖励分数)数据对

**备选方案**：

- **用户点击数据**：利用用户交互的隐式反馈
- **离线指标**：直接将 NDCG@10 作为回合结束时的奖励

### 3.3 强化学习算法与实现

#### 3.3.1 推荐算法：PPO (Proximal Policy Optimization)

- 最流行、最稳健的策略梯度算法
- 在许多 RLHF 任务中取得成功

#### 3.3.2 推荐库：Hugging Face `trl`

**使用步骤**：

1. **加载重排序模型**：在微调后的 Embedding 模型顶部添加排序层
2. **集成奖励模型**：将奖励信号与`trl.PPOTrainer`结合
3. **自动化处理**：`trl`负责处理复杂的 PPO 算法细节

### 3.4 RL 训练流程

**1. 数据准备**：

- **查询-候选文档对**：为每个查询生成 Top-K 候选文档列表
- **奖励数据**：通过 LLM 评估生成(查询, 候选列表, 奖励分数)

**2. 模型初始化**：

- 加载第一阶段微调好的 Embedding 模型
- 构建或加载 Reranker 模型

**3. PPOTrainer 配置**：

- 定义 Reranker 模型作为`policy_model`
- 定义奖励函数
- 配置 PPO 参数（学习率、Epochs、Batch Size 等）

**4. 训练执行**：

- PPOTrainer 在模拟环境中进行"试错"
- 根据当前策略对候选文档进行排序
- 获取奖励并更新策略模型参数

**5. 评估验证**：

- 在独立测试集上使用优化后的 Reranker
- 计算 NDCG@10 与基线比较

---

## 4. 效果评估

### 4.1 评估指标

- **NDCG@K**：衡量检索结果排序质量的重要指标
- **Recall@K**：查询的正样本是否出现在 Top-K 结果中
- **MRR (Mean Reciprocal Rank)**：平均倒数排名
- **CosineSim**：查询与返回结果的平均余弦相似度

### 4.2 评估实现

```python
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.util import cos_sim

# 构建评估器
evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name="retrieval_eval",
    score_functions={"cosine": cos_sim},
    main_score_function="cosine",
    show_progress_bar=True,
    corpus_chunk_size=50000,
    mrr_at_k=[10],
    ndcg_at_k=[10],
    recall_at_k=[10]
)

# 评估模型
results = evaluator(model)
print(f"NDCG@10: {results['ndcg_at_10']}")
print(f"MRR@10: {results['mrr_at_10']}")
print(f"Recall@10: {results['recall_at_10']}")
```

---

## 5. 整体技术栈建议

### 5.1 Embedding 微调

- **基础模型**：`Qwen3-Embedding-0.6B`或`1.8B`
- **微调工具**：`ms-swift`（推荐 for Qwen）或`sentence-transformers`
- **数据构建**：结合人工标注和 LLM 辅助生成难负样本

### 5.2 Reranker RL 优化

- **基础 Reranker 模型**：在微调后的 Embedding 模型上加 MLP 层，或使用`Qwen3-Reranker`
- **RL 框架**：`trl` (Hugging Face)
- **奖励模型**：利用 Qwen-Max/Qwen-Base API 进行 RLAIF 奖励生成
- **RL 算法**：PPO（由`trl`库封装）

### 5.3 向量检索

- **索引构建**：使用 Faiss 进行快速近似最近邻搜索
- **检索流程**：查询嵌入 → 向量索引查找 → Top-K 结果 → 重排序优化

---

## 6. 实施建议与注意事项

### 6.1 团队实操的建议

1. **从简单开始**：

   - 先完成阶段一的对比学习微调
   - 验证基础效果后再进入 RL 阶段

2. **利用现有工具**：

   - 充分利用`ms-swift`、`trl`等成熟框架
   - 避免从头实现复杂算法

3. **数据质量优先**：

   - 投入更多精力在难负样本的构建上
   - 利用 LLM API 辅助数据生成和标注

4. **逐步优化**：
   - 先使用简单的奖励设计（如 LLM 评分）
   - 后续可根据业务需求细化奖励函数

### 6.2 关键成功因素

1. **高质量难负样本**：这是对比学习成功的关键
2. **合理的奖励设计**：直接影响 RL 的优化方向
3. **充分的评估**：确保每个阶段的改进都有数据支撑
4. **计算资源规划**：合理分配微调和 RL 训练的资源

---

## 7. 总结与展望

通过"对比学习微调 + 强化学习优化"的两阶段方案，可以显著提升文本嵌入模型在法律及书面语领域的检索精度。这种方法既充分利用了对比学习在语义空间优化上的优势，又通过 RL 直接针对目标指标进行优化，是当前最前沿且实用的技术路线。

对于刚入门 llm 团队，关键在于选择合适的工具和框架，重点投入数据质量的提升，并采用渐进式的优化策略。随着技术的不断发展，这一方案将在更多专业领域的信息检索任务中发挥重要作用。
