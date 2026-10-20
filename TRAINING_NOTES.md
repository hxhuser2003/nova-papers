# 训练笔记(GRPO / Agentic RL)

记录"4(训练部分)"相关的技术调研和设计决策,中文,随时间持续补充。

## 2026-10-20 调研:参考项目与TRL的GRPO多轮工具调用支持

### 参考来源

- [PeterGriffinJin/Search-R1 (GitHub)](https://github.com/PeterGriffinJin/Search-R1)
- [TRL: OpenEnv Integration for Training LLMs with Environments](https://huggingface.co/docs/trl/en/openenv)
- [TRL GRPOTrainer docs](https://github.com/huggingface/trl/blob/main/docs/source/grpo_trainer.md)

### 1. Search-R1(概念设计参考,不能直接抄代码)

- 用的训练框架是**veRL**,不是TRL,代码没法直接复用
- 但**reward设计思路**值得参考:纯规则式打分(`"reward_model": {"style": "rule", "ground_truth": solution}`),跟我们`reward.py`的思路一致
- **检索工具独立成服务**(不是写死在训练循环里),如果以后训练并发量大,这个架构模式值得借鉴

### 2. TRL官方的OpenEnv集成指南(可以直接用,甚至改一改代码就能跑)

TRL的`GRPOTrainer`现在原生支持"多轮工具调用"训练,文档里有完整代码示例。但这里有个关键的架构决策要做:

| | `environment_factory`(TRL原生方式) | `rollout_func`(自己写循环) |
| --- | --- | --- |
| 怎么调用工具 | 模型原生的function-calling(OpenAI那种结构化tool_calls) | 完全自己定义,可以沿用我们现有的`<search>/<read>/<answer>`标签格式 |
| 代码量 | 少——TRL自动处理生成→解析→执行→反馈整个循环 | 多——要自己接管整个rollout过程 |
| 跟现有代码兼容吗 | **不兼容**——要换成TRL的原生tool-calling格式,`reward.py`里检查`<think>`标签的格式分、A的基线测试结果、现有的`agent_loop.py`全部要跟着改 | **兼容**——可以直接复用`agent_loop.py`的`run_episode`/`parse_action`逻辑,原封不动接进去当rollout生成器 |

**决策:用`rollout_func`,不用`environment_factory`。**

理由:虽然`environment_factory`代码量更少、官方更推荐,但我们已经在tag格式(`<search>/<read>/<answer>`)上投入了很多(A的spec文档、基线测试、`reward.py`的格式检查、调试"别重复搜索"这些工作),换成TRL原生的function-calling格式等于把这些全部推倒重来。`rollout_func`虽然要多写点代码,但能让`agent_loop.py`直接复用,不破坏已有的一切。

### 其他实用细节

- **`max_completion_length`要覆盖整个多轮对话**(不是单轮),我们5轮的设计,这个参数要设得够大,不然轮数没到就被硬截断
- **单GPU(CARC上申请的V100/P100)用`vllm_mode="colocate"`**,vLLM和训练在同一进程里跑,不需要多GPU
- **Reward设计建议**:官方经验是"简单的二元奖励(对/错)比带部分分的复杂奖励效果更干净,GRPO看的是组内相对排名,不是绝对值大小"——这条对调`reward.py`的权重可能有参考价值
