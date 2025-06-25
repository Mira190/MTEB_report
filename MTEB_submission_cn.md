以下为符合学术文档风格的 MTEB 模型提交流程详细指南，已采用正式语体、结构清晰，并附带必要引用。请根据实际项目调整细节。

---

# MTEB 模型提交流程：详细指南

本指南基于 MTEB 官方文档与社区实践，分为三个阶段：模型准备、在本地运行 MTEB 评估、将评估结果提交至排行榜。文中引用 Hugging Face 博客与 MTEB 官方资源，旨在确保流程严谨且可复现。

---

## 1. 引言

本文档旨在提供系统化的操作流程和技术指导，使研究者或工程团队能够将自定义文本嵌入模型提交至 MTEB（Massive Text Embedding Benchmark）排行榜。流程涵盖模型公开、评估执行、元数据生成与提交等环节，符合 MTEB 及 Hugging Face 平台要求。参考文献包括 MTEB 官方论文和 Hugging Face MTEB 指南。

---

## 2. 模型准备阶段

### 2.1 开源与可复现性要求

1. **模型架构与权重公开**

   - 必须保证模型定义及其训练得到的权重对外公开可获取。推荐方式是在 Hugging Face Hub 创建公开（public）仓库并上传相关文件。
   - 架构描述（如模型配置文件 `config.json`）应完整记录，包括隐藏层维度、注意力头数、词汇表大小等，以确保他人可复制模型结构并重现结果。
   - 权重文件（如 `pytorch_model.bin` 或 `model.safetensors`）需与配置对应，并标明所用训练数据版本或预训练来源。

2. **推理接口与代码公开**

   - 应提供推理脚本或库接口，使用户能够加载模型并生成文本嵌入。若兼容 `sentence-transformers` 或 `transformers` 框架，应在 README 中给出示例代码（例如 `encode(List[str]) → np.ndarray`）。
   - 若实现自定义封装，需明确说明输入/输出格式、批处理方式及可能的预处理（如分词、文本截断）。

3. **训练代码（推荐）**

   - 虽然 MTEB 提交并不强制要求提供训练脚本，但建议将训练流程（包括数据处理、超参数配置、训练命令等）纳入开源仓库，以提升工作可信度与可复现性。
   - 在 README 或附录中，应说明训练环境（如依赖库版本、显存需求、分布式配置等）及运行步骤。

4. **许可证声明**

   - 模型及相关代码须采用开源许可证（如 Apache 2.0、MIT 等），并在仓库根目录明确 LICENSE 文件。
   - 确保所用预训练模型或数据许可兼容所选开源协议，避免法律风险。

### 2.2 上传至 Hugging Face Hub

1. **账号与仓库创建**

   - 注册并登录 Hugging Face 账号（[https://huggingface.co）。](https://huggingface.co）。)
   - 点击“New model”创建新仓库，选择描述性名称并设置为 Public，确保平台能够访问。

2. **必要文件上传**

   - **模型权重**：`pytorch_model.bin` 或 `model.safetensors`。
   - **配置文件**：`config.json`（包含模型超参数设置）。
   - **分词器资源**：如 `tokenizer.json`、`vocab.txt`、`tokenizer_config.json`、`special_tokens_map.json`。
   - **Sentence-Transformers 专用文件**（若适用）：`modules.json`、`config_sentence_transformers.json` 等。
   - **附加资源**（如有）：模型使用说明、示例脚本、环境依赖列表（`requirements.txt`）。

3. **模型卡（README.md）撰写**

   - **模型概述**：简要介绍模型设计思路、预训练或微调数据来源。
   - **使用示例**：提供加载与推理示例，展示如何生成嵌入。
   - **性能摘要**（可选）：若已有初步评估结果，可简述在部分基准上的表现。
   - **依赖与环境**：列出关键依赖版本，例如 PyTorch、Transformers、Sentence-Transformers 版本。
   - **许可声明**：明确所使用许可证类型。
   - **引用方式**：提供相关论文或项目引用格式。

---

## 3. 本地运行 MTEB 评估

### 3.1 安装与环境准备

- 安装 MTEB Python 包：

  ```bash
  pip install mteb
  ```

- 确保 Python 环境具有必要深度学习框架（如 PyTorch、TensorFlow）及所需硬件驱动（GPU、CUDA 等）。
- 建议使用虚拟环境或 Conda 管理依赖，以避免冲突并便于重现。

### 3.2 加载与封装模型

1. **兼容 Sentence-Transformers**

   - 若模型符合 Sentence-Transformers 接口，可按以下示例加载：

     ```python
     from sentence_transformers import SentenceTransformer
     model = SentenceTransformer("username/model-repo")
     ```

   - `encode()` 方法应支持接收字符串列表并返回形如 `(n_samples, hidden_size)` 的嵌入张量或 NumPy 数组。

2. **自定义封装**

   - 若模型不直接兼容 Sentence-Transformers，可实现自定义包装类，使其提供 `encode(texts: List[str]) -> np.ndarray`。
   - 在包装中需包含必要的预处理（如分词、截断）和批处理逻辑，并将深度模型前向输出映射为嵌入向量。

3. **示例**

   ```python
   class CustomEmbedder:
       def __init__(self, model, tokenizer, device):
           self.model = model.to(device)
           self.tokenizer = tokenizer
           self.device = device
       def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
           embeddings = []
           for i in range(0, len(texts), batch_size):
               batch = texts[i:i+batch_size]
               inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
               with torch.no_grad():
                   outputs = self.model(**inputs).last_hidden_state
                   # 此处可按需池化，如取 [CLS] 或平均池化
                   pooled = outputs[:, 0, :].cpu().numpy()
               embeddings.append(pooled)
           return np.vstack(embeddings)
   ```

### 3.3 选择评估任务

- MTEB 支持多种任务类别（Classification、Semantic Textual Similarity、Retrieval、Clustering、Reranking 等），可根据需求选择全部或子集。
- **完整评估示例**：

  ```python
  from mteb import MTEB
  evaluation = MTEB(tasks=None)  # 运行所有可用任务
  ```

- **部分任务示例**：

  ```python
  evaluation = MTEB(tasks=["Banking77Classification", "STSBenchmark", "QuoraRetrieval"])
  ```

### 3.4 运行评估并保存结果

- 指定输出目录；建议与模型名称相关联，例如 `results/your_model_name`：

  ```python
  output_folder = "results/your_model_name"
  results = evaluation.run(model, output_folder=output_folder)
  ```

- 运行过程包括：

  - 自动下载并加载各子任务数据集
  - 使用模型 `encode()` 生成嵌入
  - 执行每项任务的评估逻辑（如分类任务训练轻量分类器、STS 计算余弦相似度、检索任务计算 nDCG\@k 等）
  - 将各任务详细结果以 JSON 格式存储于指定目录

- **资源消耗**：评估可能消耗较多时间与计算资源，取决于模型大小与任务数量；建议提前确认硬件（GPU/CPU/内存）可用性，并考虑任务并行或分阶段运行。

---

## 4. 生成并整合 MTEB 元数据

### 4.1 生成元数据文件

- MTEB 提供命令行工具 `create_meta`，可自动根据评估结果生成适用于模型卡的元数据片段（YAML 头部及结果表格）：

  ```bash
  mteb create_meta \
    --results_folder results/your_model_name \
    --output_path model_card.md \
    --from_existing path/to/your/README.md  # 可选，用于合并已有文档
  ```

- `model_card.md` 包含：

  - YAML 头部（例如 `language`, `license`, `model_name`, `mteb_results` 等字段）
  - 各任务评估结果表格或可视化图表（如性能雷达图）

- 若已有 README，可通过 `--from_existing` 合并，保留原有介绍并将 MTEB 结果置于开头或合适位置。

### 4.2 更新 Hugging Face 仓库 README

1. 打开生成的 `model_card.md`，将其内容复制或合并到仓库根目录的 `README.md` 顶部或适当位置。
2. 确保 YAML 头部字段齐全且格式正确：

   - `language`：模型支持语言，如 `"en"` 或 `"multilingual"`
   - `license`：与仓库 LICENSE 文件一致
   - `model_name`：Hugging Face 仓库名称
   - `mteb_results`：指向本地或仓库中 JSON 结果文件位置

3. 提交更改：

   - 若使用 Git，本地修改后执行 `git add README.md`、`git commit -m "Add MTEB evaluation results"`、`git push`。
   - 或通过 Hugging Face 网页端直接编辑并保存。

### 4.3 验证与故障排除

- 提交后 MTEB 排行榜通常每日刷新一次。若模型未在排行榜中出现，应检查：

  1. 仓库是否为 Public。
  2. README 中 YAML 头部位置是否在文件顶部或符合 MTEB 预期解析位置。
  3. `mteb_results` 路径是否正确指向已上传结果文件。
  4. JSON 结果文件格式是否与 MTEB 要求一致（通常无需修改，若使用官方评估包生成即符合规范）。
  5. 等待刷新周期（通常基于 UTC 时间每日运行）。

- 如遇错误，可参阅 MTEB GitHub Issue 或社区讨论，确认最新要求；也可本地重新运行 `mteb create_meta` 并检查输出文件格式。

---

## 5. 附录：最佳实践与注意事项

1. **版本管理与依赖声明**

   - 使用 Git 管理模型仓库和评估脚本，确保每次更改可追溯。
   - 在 `requirements.txt` 或 `environment.yml` 中列明关键库及版本（如 `mteb>=1.0.0`, `sentence-transformers>=2.2.0` 等）。

2. **硬件与时间预估**

   - 轻量嵌入模型评估可使用单 GPU；若模型规模较大（数十亿参数或使用复杂后处理），建议使用多 GPU 或高性能实例。
   - 评估全量 MTEB 基准可能需数小时至数天，视硬件与并行策略而定；可先对少量任务进行试验以估算时间。

3. **结果可视化**

   - `mteb create_meta` 自动生成基础表格及雷达图；如需更定制化可在本地进一步绘制或整合到论文/报告中。
   - 在 README 中可简要说明模型在不同任务上的优势与不足，以便读者理解模型特性。

4. **多语言支持声明**

   - 若模型支持多语言，应在 README 中明确列出支持语言或说明“multilingual”，并在评估时选择相应语言子集。
   - 若只针对特定语言（如英语法律文本），应注明限制，便于用户设置期望。

5. **训练流程开源**

   - 虽非提交硬性要求，但若将训练脚本、数据预处理代码纳入公开仓库，可增强学术贡献影响。
   - 建议在附录或单独文档中说明大规模训练/微调过程，包括超参数选择、训练时间、硬件配置等。

6. **许可证与合规性**

   - 确认所有使用的预训练模型、评估数据许可兼容；避免未经授权的数据使用。
   - 在 README 或专门文档中注明使用的外部资源及其许可信息。

7. **持续更新与迭代**

   - 随着新的 MTEB 版本或基准数据更新，应及时重新评估并更新 README。
   - 可在仓库中保留 `results/` 目录下不同版本的评估结果，以展示模型迭代成果。

---

## 6. 参考文献

1. Ciancone M., Kerboua I., Schaeffer M., Sequeira G., Siblini W. “MTEB: Massive Text Embedding Benchmark”, arXiv, 2022.
2. Hugging Face 博客 “MTEB Leaderboard Best Practices”，Hugging Face, 2023.
3. MTEB 官方 GitHub 仓库及文档，[https://github.com/embeddings-benchmark/mteb](https://github.com/embeddings-benchmark/mteb) 。
4. Sentence-Transformers 文档，[https://www.sbert.net/](https://www.sbert.net/) 。
