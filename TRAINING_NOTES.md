# 训练笔记(GRPO / Agentic RL)

记录"4(训练部分)"相关的技术调研和设计决策,中文,随时间持续补充。

## 2026-10-20 调研:检索服务接口、并发、系统提示词风格(已核实)

来源:[search_r1/search/retrieval_server.py](https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/search/retrieval_server.py)、[scripts/data_process/nq_search.py](https://github.com/PeterGriffinJin/Search-R1/blob/main/scripts/data_process/nq_search.py)

### 1. 检索服务是批量接口,不是一次一个查询——现在不适用,但以后扩展要照这个样子设计

`retrieval_server.py`的`POST /retrieve`接口定义:
```python
class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False
```
一次性把当前这一步所有待处理的查询打包发过去,服务端批量处理完一起返回。这是因为GRPO训练时同一步可能有几十上百条轨迹同时要检索,逐条发HTTP请求的网络开销会累积得很可观。

**现在不适用**:我们`QasperCorpus.search()`是进程内直接调用,不走网络,没有这个开销。**但如果以后检索扩展成多篇论文共享的独立服务,接口就该照这个批量请求的样子设计**,不是"一条轨迹一次调用",先记在这里。

### 2. 他们自己的检索服务也没对共享资源加锁——印证"不修"的决定是对的

`retrieval_server.py`里的全局`retriever`对象,在`retrieve_endpoint`里直接用,FastAPI默认并发处理请求,没有任何锁保护,存在竞态条件。官方参考实现自己就是这么写的。

这跟我们`qasper_corpus.py`里"并发建索引可能重复构建"那个已知小瑕疵是同一类问题——之前判断"不影响正确性、暂时不用处理"是对的方向,连参考实现本身都没去修这类东西,不是工程上偷懒。

### 3. 系统提示词更短,也没告诉模型"还剩几轮"——确认跟我们现有设计已经对齐

Search-R1给模型的提示词原文:
> "Answer the given question. You must conduct reasoning inside `<think>` and `</think>` first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by `<search> query </search>` and it will return the top searched results between `<information>` and `</information>`. You can search as many times as your want."

没有提前告诉模型总共有几轮配额(连语法都有个小错误,原文如此)。确认我们自己`agent_loop.py`的`SYSTEM_PROMPT`也同样没有把"最多5轮"这个信息透露给模型——这点已经对齐,不用改。

**间接印证上一条"轮数耗尽强制追加一轮作答"这个改动是对的方向**:参考实现自己也没有通过"告诉模型还剩几轮"来解决"不知道何时停"的问题,而是靠"轮数用满后强制逼一个答案"这个兜底机制处理——不是临时想出来的偏方,是跟参考实现思路一致的做法。

## 2026-10-20(最高优先级):`reward.py`缺"答案对不对"这一项

对照Search-R1真实的奖励函数源码([verl/utils/reward_score/qa_em_format.py](https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/utils/reward_score/qa_em_format.py))逐条核实过,确认是真问题,不是误报。

### 问题

重新读了一遍[backend/reward.py:126](backend/reward.py#L126)的`composite_reward`:
```python
return (
    weights.format * format_reward(sample)
    + weights.action * action_reward(sample)
    + weights.faithfulness * faithfulness_reward(question, sample, evidence_passages, judge)
)
```
三项里**没有任何一项把模型答案拿去跟QASPER的标准答案(`answers`字段)做比对**。`faithfulness_reward`测的是"答案有没有依据检索到的证据文本"(跟检索段落的ROUGE-L重叠度),不是"答案是不是真的回答对了问题"。

**后果**:模型完全可以做到格式工整、检索词不重复、答案也确实是从检索到的某段原文摘出来的——但摘的是答案错误、答非所问的那段,现在这套reward照样给不低的分数,因为没有任何一项在检查"这跟标准答案对不对得上"。这是当前训练信号里缺失的核心一块。

### Search-R1的参照(决策树式打分,已核实源码属实)

| 情况 | 得分 |
| --- | --- |
| 格式对 + 答案完全匹配(EM) | 1.0 |
| 格式对 + 答案错,但检索到了正确证据 | 0.3 |
| 格式对 + 答案错,也没检索对 | 0.2 |
| 格式错 + 答案碰巧对 | 0.8 |
| 格式错 + 答案也错 | 0.1 |

核心逻辑:"答案是否匹配标准答案"是压倒性主信号,格式对错只是在这个基础上加减分。

### 建议的修复

往`composite_reward`里加一项`correctness_reward`,**直接复用`scripts/run_baseline.py`里已经写好的`answer_f1()`函数**([scripts/run_baseline.py:71](scripts/run_baseline.py#L71))——这个函数本来就是为算基线数字写的,现在正好能原样搬进训练奖励里,不用重新发明。EM换成Answer F1这个选择本身没问题(QASPER自由文本答案太多,EM太严格),不用跟Search-R1一样改回EM,只是"跟标准答案比对"这一项现在整个是空的,得补上。

顺手把`evidence_f1()`([scripts/run_baseline.py:77](scripts/run_baseline.py#L77))也作为第四项加进去,会比现在只有一个"grounding分数"更贴近Search-R1验证过的这套设计——检索质量在他们的打分里也是独立计分项。

**先记录,等按计划攒够参考点一起改,不单独改。**

### 两个次要参考点(优先级较低,现在不用改)

1. Search-R1的EM奖励函数要求`<answer>`标签至少出现2次匹配才提取答案、取最后一次匹配——这是兼容"模型先在`<think>`里打一遍草稿答案,结尾再正式给一次"这种行为。我们现在`parse_action`严格要求"恰好一个动作标签",如果以后真实模型也有类似"先预告一次答案"的习惯,这是个可以参考的宽松化思路,但现在没遇到这个行为,不用改。
2. 检索质量在Search-R1打分里是独立计分项(是否命中标准证据),这跟QASPER数据里已有的`evidence_para_ids`/Evidence F1是一回事,已经并入上面"加第四项"的建议里。

## 2026-10-20 调研:Search-R1代码细读(已对照真实代码逐条核实)

来源:[search_r1/llm_agent/generation.py](https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/llm_agent/generation.py)

### 最值得现在就考虑加进`agent_loop.py`的一条:轮数用完要强制作答,不能让轨迹直接作废

`generation.py`里主循环(最多`max_turns`轮)结束后,有一段**强制最后再生成一轮、但禁用search**的逻辑:
```python
if active_mask.sum():
    _, dones, valid_action, is_search = self.execute_predictions(
        responses_str, self.tokenizer.pad_token, active_mask, do_search=False
    )
```
意思是:正常循环跑满轮数还没给出`<answer>`的话,**不会直接判定这条轨迹失败**,而是强制再生成一轮、这轮把search工具禁掉,逼模型必须给出一个答案,哪怕质量一般。

**这正好对上我们`eval/BASELINE.md`第71行记的那条**:"不会停:11/40用满5轮仍未作答"。现在`agent_loop.py`的做法是轮数耗尽直接`terminated_reason = "max_rounds"`、`final_answer = None`,这条轨迹对训练毫无贡献(reward拿不到任何可打分的答案)。按Search-R1这个做法,相当于把40题里那11题"白跑的"轨迹,至少变成"有产出、能打分"的轨迹,训练数据利用率会明显提升。**先记在这里,按计划之后攒够参考点再一起改,不单独改。**

### 其他几条核实结果

1. **独立的rerank服务,确认不是瞎建议**——`search_r1/search/`目录下确实有独立的`retrieval_server.py`和`rerank_server.py`(还有一个合并版`retrieval_rerank_server.py`),说明"检索完再精排一次"在这个研究方向里是标准做法。

2. **截断动作标签的机制,我们是对的,但实现方式跟他们不一样**——他们是生成完之后再用字符串裁剪(`resp.split('</search>')[0] + '</search>'`);我们是用vLLM的`stop`参数**在生成过程中就截停**(`vllm_client.py`的`STOP_SEQUENCES`+`include_stop_str_in_output`)。效果类似(殊途同归),但不是同一种机制,不用改。

3. **工具返回内容的loss mask,已经看到具体怎么实现的**——`info_mask`是一个和token序列等长的张量,把检索返回的那部分token位置填成pad token id,跟真实token id版本分开存,训练时用这个mask把对应token的loss清零。等真正写训练脚本那一步,这是个可以直接参考的具体实现方式。

4. **批量并发这块,不用跟**——他们的batch并发是在张量层面手写的(GPU padding、`active_mask`追踪哪些轨迹还没完成),这是因为他们直接对接veRL做底层训练。我们现在用`ThreadPoolExecutor`调vLLM的HTTP接口做并发,把"怎么高效批量生成"完全交给vLLM服务端的continuous batching处理,更简单也更合理——这是依赖vLLM服务省掉的一层复杂度,不是落后于人,不需要为了"像他们一样"去手写张量级别的批处理逻辑。

### 文件结构对照(已核实存在)

| Search-R1 | 对应我们的 |
| --- | --- |
| `search_r1/llm_agent/generation.py` | `agent_loop.py` |
| `search_r1/search/retrieval_server.py`、`rerank_server.py` | `qasper_corpus.py` |
| `scripts/data_process/` | `prepare_qasper.py` |
| `train_grpo.sh`/`train_ppo.sh` | (还没做,对应"4训练部分") |

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
