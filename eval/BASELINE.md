# 训练前基线测试记录（任务 3A）

在 40 道样题（`eval/sample_questions.jsonl`）上测试未训练的 Qwen2.5-3B-Instruct，作为后续 GRPO 训练的对比基准。

**结论**：检索一次（rag）比只看摘要提升约 75%（Answer F1 0.212 → 0.372）。但未训练的多轮 agent（0.224）**不如**检索一次。主要原因是模型不知道何时停止、几乎不精读段落、自己改写的搜索词不如原问题。这三点正是 RL 训练要改进的方向；训练后 agent 能否明显超过 rag（三次运行在 0.35–0.37 之间），是检验训练效果的核心指标。

## 1. 部署环境

| 项目 | 设定 |
| --- | --- |
| 模型 | Qwen/Qwen2.5-3B-Instruct |
| 硬件 | Google Colab T4（15360 MiB 显存） |
| 推理框架 | vLLM，OpenAI 兼容接口（`localhost:8000/v1`） |
| 精度 | fp16（T4 不支持 bf16，需加 `--dtype half`） |
| 启动参数 | `--max-model-len 8192 --gpu-memory-utilization 0.85` |
| 显存占用 | 12299 MiB（其中权重约 6 GB，其余为 vLLM 预分配的 KV cache） |
| 生成参数 | temperature = 0，单轮最多 400 token |
| 单次简短对话延迟 | 3.9 秒（首次请求，含预热） |

以下结果来自 2026-10-17 的复测，对应 `agent_loop.py` 改为消息列表接口后的代码（commit 445c73d）。

启动命令：

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct --dtype half --max-model-len 8192 \
    --gpu-memory-utilization 0.85 --port 8000
```

## 2. 三组基线

| 组 | 做法 |
| --- | --- |
| direct | 只给标题、摘要、问题，直接作答，不检索 |
| rag | 用原问题 BM25 检索一次，取 top-3 段落，交给模型作答 |
| agent | `backend/agent_loop.py` 多轮循环（search / read / answer），最多 5 轮 |

三组使用同一模型、同一批题、同一检索器。提示词见 `eval/system_prompt.txt`（agent）和 `scripts/run_baseline.py`（direct / rag）。

## 3. 结果

### Answer F1

| 题型 | 题数 | direct | rag | agent |
| --- | --- | --- | --- | --- |
| extractive | 15 | 0.085 | 0.266 | 0.157 |
| free_form | 10 | 0.019 | 0.188 | 0.161 |
| yes_no | 10 | 0.200 | 0.500 | 0.300 |
| unanswerable | 5 | 1.000 | 0.800 | 0.400 |
| **全部** | 40 | **0.212** | **0.372** | **0.224** |

### Evidence F1

| 题型 | direct | rag | agent（read 过的段落） | agent（看到过的所有段落） |
| --- | --- | --- | --- | --- |
| 全部 | 0.125 | 0.196 | 0.150 | 0.178 |

注：unanswerable 题的标准证据为空，不读任何段落即得满分，因此 direct 的 0.125 全部来自这 5 道题。比较检索质量时应只看前三类题。

### 效率与行为

| 指标 | direct | rag | agent |
| --- | --- | --- | --- |
| 平均耗时（秒/题，16 路并发） | 4.4 | 8.2 | 19.8 |
| 成功给出答案 | 40/40 | 40/40 | 28/40 |
| 平均轮数 | 1 | 1 | 3.30 |

agent 结束原因：正常作答 28，用满 5 轮未作答 11，连续格式错误 1。40 道中只有 2 道用过 read。

## 4. 分析

1. **检索有效**：rag 在每一类题上都明显优于 direct，extractive 从 0.09 升到 0.27，yes_no 从 0.20 升到 0.50。说明"在论文里查找"这条主线成立。
2. **未训练的 agent 不如检索一次**，与 Search-R1 论文中"仅靠提示词使用搜索工具效果有限"的结论一致。具体表现为：
   - **不会停**：11/40 用满 5 轮仍未作答，一直在搜索。
   - **不精读**：只有 2/40 用过 read，模型只依据 search 返回的截断片段作答，细节容易出错。
   - **搜索词不如原问题**：agent 看到过的段落，Evidence F1 为 0.178，低于直接用原问题检索的 0.196。
3. **对 unanswerable 题偏保守**：direct 只能看摘要，倾向于回答 Unanswerable，因此这类题得满分，但其他类题几乎为 0。agent 和 rag 看到原文后，会尝试作答，因此在 unanswerable 上反而失分。

## 5. 调试过程中的发现

| 问题 | 现象 | 处理 |
| --- | --- | --- |
| 提示词无示例 | 第一轮 40/40 都跳过 `<think>`，直接输出动作 | 提示词加入一轮完整示例后，格式正确率 0/40 → 40/40 |
| 工具结果的拼接方式 | 把 `<information>` 拼在模型回复中间让其续写，模型从第 2 轮起输出空串，29/40 因连续格式错误终止 | 改为标准多轮对话：工具结果作为新的 user 消息，格式错误降到 1/40 |
| 拆回字符串的设计 | 循环把对话拼成一整段字符串，适配器再拆回多轮消息，容易出错 | B 将 `agent_loop.py` 改为直接维护消息列表，`ModelClient` 接收 `list[dict]`；复测行为数据（作答 28、用满轮数 11、格式错误 1、read 2）与改前完全一致 |
| 结果有小幅波动 | temperature = 0 时重复运行，三次 Answer F1：rag 0.361 / 0.352 / 0.372，agent 0.247 / 0.253 / 0.224 | 并发批处理与 fp16 运算顺序导致；且样题只有 40 道，1 道题答对与否就影响总分约 0.025。正式对比训练前后时，建议各跑 3 次取平均 |

## 6. 对后续训练的建议

- ~~训练与推理的输入格式要一致~~：已完成，`agent_loop.py` 改为消息列表，GRPO rollout 直接复用即可。
- ~~系统提示词统一~~：已完成，统一使用 `eval/system_prompt.txt`。
- **评估要多跑几次**：40 道题的波动约 ±0.03，训练前后对比时各跑 3 次取平均，或扩大测试题数。
- **奖励设计可考虑**：Answer F1 作为主奖励；对用满轮数未作答的轨迹给 0 分，引导模型学会停止。
- **算力**：T4 足够做推理和基线测试，GRPO + LoRA 训练需要更大显存（L4 / A100 或学校集群）。

## 7. 复现

```bash
pip install vllm openai rank_bm25
# 启动 vLLM（见第 1 节）后：
python scripts/run_baseline.py                 # 三组都跑
python scripts/run_baseline.py --modes agent   # 只跑 agent
```

结果输出到 `eval/results/`：每组一个 `<mode>.jsonl`（逐题预测、得分、轨迹），汇总在 `summary.md`。
