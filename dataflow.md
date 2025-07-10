# 利用 OpenDCAI DataFlow TextPipeline 搭建端到端流程

OpenDCAI 的 **DataFlow** 框架提供了强大的流水线功能，可以把数据处理的各个环节串联起来，实现从「原始法条数据」到「训练所需的 法条-查询 对」的全自动生产流程。

DataFlow 把每一步封装成模块化的 **算子（Operator）**，支持规则方法、深度模型、LLM API 等多种手段在同一个管道里组合使用。

这里，我们将使用 **TextPipeline** 来为「法条 → 案例式查询」场景定制一个完整可复用的流水线。

---

## ✅ 流程步骤概览

### 1️⃣ 环境准备

在终端中安装：

```bash
pip install open-dataflow
```

- 如果需要调用本地推理模型（例如 vLLM），可以安装带扩展的版本：

```bash
pip install open-dataflow[vllm]
```

- 安装完可用以下命令检查版本：

```bash
dataflow -v
```

---

### 2️⃣ 加载输入数据

**DataFlow** 提供了基础的读取算子，支持多种格式：

- **JSON**：

  - 使用 `JsonReader` 指定路径、字段映射（如 `id`, `content`）。
  - 每条 JSON 记录会成为管道里的一个元素。

- **TXT**：

  - 预处理成每条法条一行的 JSON，再用 JSON Reader 读取。
  - 或使用 Text Reader 逐行读取，后续算子再拆分成法条。

- **PDF**：

  - 推荐先离线转为结构化文本（按条拆分）。
  - DataFlow 也提供 Knowledge Base Cleaning Pipeline 的 PDF Reader 支持，可直接提取条文段落。

---

### 3️⃣ 数据预处理（可选）

- 轻度清洗法条文本：

  - 去掉多余空白
  - 统一全角/半角标点
  - 修正 OCR 错误

- DataFlow 有通用文本清洗算子，可通过正则配置替换规则。

---

### 4️⃣ 查询生成（核心步骤）

为每条法条内容生成「案例式查询」。可以选择以下策略之一或组合：

#### ✅ 模板生成算子

- 使用事先设计好的规则模板。
- 可实现为自定义的 PythonOperator：

  - 例如检测“是指”生成定义型问句
  - 生成类似「什么是 X？」的问题

#### ✅ LLM 调用算子

- 使用 `LLMTextGenerator` 调用大语言模型（例如 GPT-3.5）。
- 指定：

  - 模型名称（如 `openai/gpt-3.5-turbo`）
  - prompt 模板
  - 最大 token 限制
  - 停止符（如 `\n` 防止超长生成）

- 生成更自然、多样的用户提问。

#### ✅ 混合策略

- 可在 DataFlow 中串联多个生成模块：

  - 先用模板生成基础问题
  - 再调用 LLM 重写或扩展

- 也可以并行生成后合并，按优先级保留非空结果。

---

**关于多问句的处理：**

- 如果一次生成多个问句：

  - 可以让 LLM 用分号、换行分隔
  - 下游可以用 Split 算子拆分成多条独立记录
  - 实现一条法条 → 多条训练记录

---

### 5️⃣ 结果组装与输出

最后，需要把字段整理成训练集格式：

- **Formatter 算子**：

  - 选取并重命名需要的字段
  - 例如统一输出：`law_id`, `law_content`, `query`

- **Writer 算子**：

  - 写出为 **JSON Lines (JSONL)** 格式
  - 每行一个 JSON 记录

---

**✅ 输出文件示例：**

```json
{"law_id": "LAW_123", "law_content": "……", "query": "……"}
{"law_id": "LAW_124", "law_content": "……", "query": "……"}
```

- 方便后续直接流式读取
- 适合大规模训练

---

### 6️⃣ 运行管道

- 将完整流程保存在 **DataFlow YAML 配置文件**中。
- 运行命令：

```bash
dataflow run -c law_query_generation_pipeline.yaml
```

- DataFlow 会自动依次执行：

  - 加载 → 清洗 → 生成 → 格式化 → 输出

- 可在日志中查看各阶段处理数量，方便排查错误。

---

### ✅ YAML 配置示例

下面是一份示例（可根据自己需求修改路径和参数）：

```yaml
pipeline:
  name: law_query_generation_pipeline
  operators:
    - name: load_laws
      type: JsonReader
      params:
        input_path: "./data/laws.json"
        output_fields:
          - field: id
            rename: law_id
          - field: content
            rename: law_content

    - name: clean_text
      type: TextCleaner
      params:
        rules:
          - pattern: "\\s+"
            replacement: " "
          - pattern: "﻿"
            replacement: ""

    - name: generate_query
      type: LLMTextGenerator
      params:
        model: "gpt-3.5-turbo"
        prompt_template: |
          请阅读以下法律条文，并生成一个用户可能提出的关于该条文的案例式问题。
          法条内容："{law_content}"
          用户提问：
        max_tokens: 100
        temperature: 0.7
        stop: ["\n"]
        input_field: law_content
        output_field: query

    - name: rule_based_query
      type: PythonOperator
      params:
        code: |
          def process(record):
              content = record["law_content"]
              if "是指" in content:
                  term = content.split("是指")[0].strip()
                  record["query_rule"] = f"根据法律规定，什么是{term}？"
              return record

    - name: merge_query
      type: Merger
      params:
        sources: [query, query_rule]

    - name: format_output
      type: Formatter
      params:
        fields:
          - law_id
          - law_content
          - query

    - name: save_jsonl
      type: JsonWriter
      params:
        output_path: "./output/law_queries.jsonl"
        write_mode: "w"
```

---

### ✅ 提示词模板范例

一个可用于 LLM 的中文 prompt 示例：

```
你是一个智能助手，擅长将法律条文转化为用户的提问。
现在我会给你一段法律条文，请你阅读并理解，然后站在普通用户的角度，
基于条文内容提出一个相关的法律问题。这个问题应当以描述具体场景或疑问的口吻提出。

范例1:
【法条】任何人不得扰乱法庭秩序。
【提问】如果有人在法院故意喧哗扰乱秩序，会面临怎样的法律后果？

范例2:
【法条】遗嘱是指自然人依法处置个人财产并于死后生效的法律行为。
【提问】根据法律规定，什么是遗嘱？其法律特征是什么？

现在请根据下面的法条提出问题：
【法条】{law_content}
【提问】
```

---

### ✅ 预处理脚本示例

如果法条是纯文本 TXT，需要先拆分成 JSON：

```python
import re, json

data = []
current_law = {}

with open("laws.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if re.match(r'^第[一二三四五六七八九十百0-9]+条', line):
            if current_law:
                data.append(current_law)
            current_law = {"id": line.split("条")[0]+"条", "content": ""}
        else:
            if "content" in current_law:
                current_law["content"] += line

    if current_law:
        data.append(current_law)

with open("laws.json", "w", encoding="utf-8") as fout:
    for law in data:
        fout.write(json.dumps(law, ensure_ascii=False) + "\n")
```

---

### ✅ 组织成 JSONL 的建议格式

输出建议采用 **JSONL** 格式，每行一个 JSON：

```json
{"law_id": "LAW_001", "law_content": "第一条 根据《刑法》……", "query": "某人违反了刑法第一条的规定，会受到什么处理？"}
{"law_id": "LAW_002", "law_content": "第二条 ...", "query": "在什么情况下适用法律第二条的规定？"}
```

- 每条记录独立
- 便于大规模流式读取
- 适合用于训练检索模型

---

### ✅ 组织策略建议

- 一条法条可生成多个查询

  - 输出文件中为多行记录
  - 同一个 `law_id` 和 `law_content`，不同的 `query`

- 可在生成后随机打乱

  - 避免模型学习到序列偏差

- 可在输出时拆分训练集和验证集

  - 例如随机 10%留作验证

---

## 🎯 结语

通过以上步骤，你可以：

- 从 PDF、TXT、JSON 等多格式法条数据
- 自动生成丰富的 **案例式查询**
- 输出标准化 **JSONL** 格式
- 直接用于法条检索模型的训练

这套流程灵活、可扩展，可以轻松增删算子、调整提示、支持本地或远程大模型调用，是法律 NLP 数据准备的高效方案。

---

如需更详细的 DataFlow 文档，可访问：

- [OpenDCAI DataFlow 文档（中文）](https://opendcai.github.io/DataFlow-Doc/zh/guide/textpipeline/)
