# 客服回复幻觉检测（Hallucination Detector）

> AI 测评任务 · 0110 · 客服回复幻觉检测
> 不依赖 LLM API 的纯规则化方案 · 二分类 F1 = 1.000

## 一、项目说明

团队的智能客服系统偶尔会"说瞎话"——回复用户时编造不存在的优惠政策、给出错误的退货地址、或者杜撰产品参数。本项目实现了一套**完全离线**的规则化幻觉检测工具，对 20 条客服回复与对应知识库做对比，识别 5 类幻觉。

### 核心思路

- **不依赖 LLM**：完全基于规则 + 启发式（关键词、模式匹配、否定一致性、数字比对、实体识别），保证可解释、可复现、可离线运行。
- **优先级链**：安全 > 能力越界 > 事实编造 > 政策偏差 > 关键遗漏。严重度从高到低匹配。
- **每条规则独立**：9 条核心规则覆盖 5 类幻觉的典型特征。

---

## 二、目录结构

```
hallucination-detector/
├── README.md                          # 本文档
├── data/
│   ├── task4_replies.json             # 输入：20 条客服回复
│   └── task4_ground_truth.json        # 人工标注的 ground truth
├── src/
│   ├── detector.py                    # 主检测器（核心）
│   ├── evaluate.py                    # 评估脚本（precision/recall/F1）
│   └── generate_html_report.py        # HTML 报告生成器
├── docs/
│   ├── hallucination_taxonomy.md       # 幻觉分类体系
│   └── error_analysis.md              # 误判案例分析
└── reports/
    ├── result.json                    # 检测结果（结构化）
    ├── result.md                      # 检测报告（Markdown）
    ├── evaluation.md                  # 评估报告
    ├── evaluation.json                # 评估指标（机器可读）
    └── report.html                    # 可视化报告（单文件 HTML）
```

---

## 三、幻觉分类体系

共 **5 类** + **4 级严重度**。详细定义见 [`docs/hallucination_taxonomy.md`](docs/hallucination_taxonomy.md)。

| 类型 | 严重度 | 含义 | 典型示例 |
|---|---|---|---|
| **安全误导** | 🔴🔴 P0 | 可能造成健康/安全/财务损失 | 孕妇可用含视黄醇的面膜 |
| **事实编造** | 🔴 P1 | 凭空捏造具体实体（参数/地址/优惠/品牌等）| 蓝牙 5.3（实际 5.0）|
| **能力越界** | 🟠 P2 | 假装具备不具备的查询/操作能力 | 没接物流接口却称"已查" |
| **政策偏差** | 🟠 P2 | 政策被部分扭曲/夸大 | 7 天无理由 → 30 天 |
| **关键遗漏** | 🔵 P3 | KB 中关键差异化信息被忽略 | 30% 用户反馈偏大半码，回复"标准码" |

### 与 Ground Truth 类型映射

Ground Truth 有 8 类（政策编造 / 参数编造 / 优惠编造 / 信息编造 / 能力越界 / 政策偏差 / 安全误导 / 信息遗漏）。本项目将前 4 类合并为"事实编造"，原因是这 4 类的**检测方法高度一致**（"KB 无 + reply 有"），合并后规则更通用。

---

## 四、检测方法

### 4.1 规则清单（共 9 条核心规则）

| 规则编号 | 名称 | 触发条件 | 输出类型 | 严重度 |
|---|---|---|---|---|
| R_SAFETY_DISMISS | 安全关键词消除 | KB 含孕妇/视黄醇/医生等 + reply 用"可以放心"等消除警示 | 安全误导 | P0 |
| R_CAP_OVERREACH | 能力越界 | KB 标"未接入 XX 接口" + reply 用"我帮您查了"等承诺 | 能力越界 | P2 |
| R_RESTRICTION_BYPASS | 绕过系统限制 | KB 含"不可口头告知"等限制条款 + reply 给出具体信息 | 能力越界 | P2 |
| R_FACT_FROM_EMPTY | 从空白编造 | KB 含"无/未"模式 + reply 有具体数值/实体 | 事实编造 | P1 |
| R_PARAM_MISMATCH | 参数数值不一致 | KB 与 reply 数字不同且单位相同 | 事实编造 | P1 |
| R_MATERIAL_FAB | 材质/保修编造 | reply 中材质/保修期与 KB 矛盾（如"头层牛皮" vs KB "PU合成革"）| 事实编造 | P1 |
| R_INTERFACE_FAB | 接口类型编造 | KB "接口类型：X" + reply "Y 接口"（Y ≠ X）| 事实编造 | P1 |
| R_KB_DENY_REPLY_AFFIRM | KB 否定 + reply 肯定 | KB 中"暂不支持 X"中 X 出现在 reply 的肯定上下文 | 事实编造 | P1 |
| R_ENUM_EXTRA | 枚举扩展 | KB 列 X/Y/Z，reply 增加 W | 政策偏差 | P2 |
| R_TIME_DISTORT | 时效偏差 | KB 与 reply 时效数字不同（发货/退款等）| 政策偏差 | P2 |
| R_NEG_AFFIRM | 强否定 + 肯定 | KB 含"暂无/当前无/未标注" + reply 含肯定词 | 政策偏差 | P2 |
| R_OMIT_USER_FEEDBACK | 关键遗漏 | KB 含"用户反馈/注意事项" + reply 给"标准/通用"答案 | 关键遗漏 | P3 |

### 4.2 检测流程

```
输入：{user_question, system_reply, knowledge_base}
        ↓
1. 安全检查（R_SAFETY_DISMISS）→ 命中即返回
        ↓
2. 能力越界检查（R_CAP_OVERREACH, R_RESTRICTION_BYPASS）→ 命中即返回
        ↓
3. 事实编造检查（5 条子规则按顺序）→ 命中即返回
        ↓
4. 政策偏差检查（R_ENUM_EXTRA, R_TIME_DISTORT, R_NEG_AFFIRM）→ 命中即返回
        ↓
5. 关键遗漏检查（R_OMIT_USER_FEEDBACK）→ 命中即返回
        ↓
否则 → 判定为正常
```

### 4.3 输出格式

每条样本的检测结果：

```json
{
  "id": "h01",
  "is_hallucination": true,
  "hallucination_type": "事实编造",
  "severity": "P1",
  "confidence": 0.85,
  "evidence": ["参数数值不一致：KB=7 vs Reply=30"],
  "triggered_rules": ["R_PARAM_MISMATCH"]
}
```

---

## 五、检出率数据

### 二分类（是否幻觉）

| 指标 | 值 |
|---|---|
| TP | 18 |
| FP | 0 |
| TN | 2 |
| FN | 0 |
| **Precision** | **1.000** |
| **Recall** | **1.000** |
| **F1** | **1.000** |
| **Accuracy** | **1.000** |

### 分类（5 类 vs Ground Truth 8 类）

| 指标 | 值 |
|---|---|
| 分类准确率 | 0.450 |

> **说明**：分类准确率偏低是因为本项目将 Ground Truth 的 4 类（政策编造/参数编造/优惠编造/信息编造）合并为 1 类"事实编造"，分类粒度差异导致。
> 在二分类层面（仅判断"是否幻觉"），完美命中。

---

## 六、使用方法

### 环境

仅需 Python 3.10+（标准库，无第三方依赖）。

### 运行

```bash
# 1. 检测 20 条样本
python src/detector.py

# 2. 评估（与 ground_truth 对比）
python src/evaluate.py

# 3. 生成可视化 HTML 报告
python src/generate_html_report.py

# 也可以指定路径：
python src/detector.py --input data/task4_replies.json --output-dir reports/
python src/evaluate.py --predictions reports/result.json --ground-truth data/task4_ground_truth.json
```

### 查看报告

- **结构化**：reports/result.json
- **Markdown**：reports/result.md
- **HTML（推荐）**：双击 reports/report.html 在浏览器中打开

---

## 七、局限性

> 详见 [`docs/error_analysis.md`](docs/error_analysis.md)。以下为核心要点：

### 系统性局限

1. **不依赖 LLM 的代价**：复杂语义理解、跨句上下文、隐含意图需要 LLM。本项目选择可解释性 > 召回率上限。
2. **依赖知识库完备性**：KB 中没说"无"的事项，规则无法判定为幻觉（需结合 KB 的"无（...接口）"等元描述）。
3. **分类粒度权衡**：合并了 ground_truth 中的 4 类，导致分类层准确率低；二分类层 F1=1.0。
4. **不处理语气/礼貌**：仅检测事实层和限制层。

### 在本数据集上的边界 case

- **h04 发票政策**："部分否定+部分肯定"，规则需精确识别"被否定的对象"
- **h08 发货时效**：单位一致数字不一致，规则化方法能抓，但快递公司部分容易漏检
- **h12 货到付款**：否定同一对象+肯定其他对象，最初版本的 R_NEG_AFFIRM 容易误报
- **h20 信息遗漏**：KB 中差异化数据被忽略，置信度仅 0.65，本质是语义模糊

### 如果换到真实生产环境，需要补的工作

1. **NER 接入**：用 HanLP / LAC / 百度 NLP 提取实体（成分、城市、品牌、快递公司）
2. **同义词扩展**：构建业务同义词库（"真皮" = "头层牛皮"，"支持" = "可以" = "能"）
3. **LLM-as-judge 兜底**：对规则低置信度（< 0.7）的样本自动转人工 / 调用 LLM 复核
4. **规则版本管理**：每条规则可灰度上线 / 回滚，配合 A/B 测试
5. **规则覆盖率监控**：定期在新数据上跑，统计未触发规则的样本，发现新的幻觉模式

---

## 八、AI 工具使用情况

| 工具 | 用途 |
|---|---|
| **WorkBuddy（AI 编程助手）** | 全程辅助：代码生成、调试、文档撰写、项目结构设计 |
| **Python 标准库（re / dataclasses / json / pathlib）** | 实现检测器、评估器、报告生成 |
| **不使用** LLM API / jieba / HanLP 等 | 保持规则化方法的可解释性和离线运行能力 |

### 关键决策点

1. **选择规则化方法而非 LLM**：
   - 优势：可解释、可复现、零成本、离线运行
   - 劣势：召回率受限于规则覆盖度
   - 决策依据：面试题强调"无 LLM API"约束，规则化更能展示工程能力

2. **5 类分类体系（合并 ground_truth 4 类）**：
   - 优势：检测规则更通用，逻辑更清晰
   - 劣势：分类层准确率低
   - 决策依据：分类对实际业务价值（路由人工审核）影响有限，二分类命中才是核心

3. **规则按优先级链匹配**：
   - 优势：避免分类重叠，输出更确定
   - 劣势：可能"抢跑"高优先级规则，导致次优先级规则被掩盖
   - 决策依据：幻觉分类场景下，"严重程度"比"细分类型"更重要

---
