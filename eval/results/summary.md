# 训练前基线测试结果

- 模型：Qwen/Qwen2.5-3B-Instruct（vLLM，temperature=0.0）
- 样题：eval/sample_questions.jsonl（40 道）

### direct

| 题型 | 题数 | Answer F1 | Evidence F1 |
| --- | --- | --- | --- |
| extractive | 15 | 0.085 | 0.000 |
| free_form | 10 | 0.019 | 0.000 |
| yes_no | 10 | 0.200 | 0.000 |
| unanswerable | 5 | 1.000 | 1.000 |
| **全部** | 40 | **0.212** | **0.125** |

- 有答案（成功解析出 answer）：40/40
- 平均耗时：4.4 秒/题

### rag

| 题型 | 题数 | Answer F1 | Evidence F1 |
| --- | --- | --- | --- |
| extractive | 15 | 0.266 | 0.222 |
| free_form | 10 | 0.188 | 0.170 |
| yes_no | 10 | 0.500 | 0.280 |
| unanswerable | 5 | 0.800 | 0.000 |
| **全部** | 40 | **0.372** | **0.196** |

- 有答案（成功解析出 answer）：40/40
- 平均耗时：8.2 秒/题

### agent

| 题型 | 题数 | Answer F1 | Evidence F1 |
| --- | --- | --- | --- |
| extractive | 15 | 0.157 | 0.000 |
| free_form | 10 | 0.161 | 0.100 |
| yes_no | 10 | 0.300 | 0.000 |
| unanswerable | 5 | 0.400 | 1.000 |
| **全部** | 40 | **0.224** | **0.150** |

- 有答案（成功解析出 answer）：28/40
- 平均耗时：19.8 秒/题
- 结束原因：{'answered': 28, 'max_rounds': 11, 'repeated_format_error': 1}
- 平均轮数：3.30
- 至少 read 过一次：2/40
- Evidence F1（口径二，看到过的所有段落）：0.178
