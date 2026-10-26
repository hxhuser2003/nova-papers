# 训练笔记(GRPO / Agentic RL)

记录"4(训练部分)"相关的技术调研和设计决策,中文,随时间持续补充。

**主项目位置**：`/Users/hxh/Desktop/nova-papers` 是唯一维护入口，代码、计划和实验记录统一在这里更新。文中的 `~/nova-papers-grpo-7b` 等路径是 CARC 远程作业的独立运行目录，不是本机主项目；运行中的远程目录不搬移。

## 2026-10-23 当前进度（覆盖下方历史待办状态）

详细结果见 [docs/EXPERIMENT_RECORD.md](docs/EXPERIMENT_RECORD.md)。下方按日期保留的“尚未有checkpoint/HTTP未验证”等说法是历史状态，不代表当前进度。

- **真实token probe已通过**：13310264。BatchEncoding问题已修复。
- **3B LoRA两步smoke已完成参数更新与保存**：global_steps=2，504个张量改变，约58MB adapter。作业最终FAILED发生在显存CSV汇总，非训练未执行。不能据此宣称效果提升或完整训练完成。
- 参数更新累计3.3154秒，rollout/评分27.2451秒。PyTorch allocated峰值约21.34GiB；整卡采样失败，不能把allocator指标当整卡峰值。
- checkpoint在 `~/nova-papers-grpo-fixed/checkpoints/grpo-smoke-13310264/`；Mac备份 `~/Desktop/nova-papers-results/nova-grpo-backup-13310264-hLHFaH/` 已由用户确认SHA256通过。
- 显存监控及汇总修复2c5fdb0已导入CARC独立目录，尚待实机验证。旧数据汇总已重新执行。
- **实验1禁止重复搜索已完成**，13292586：Answer F1 0.192→0.118，不支持本样本上采用。
- **实验2截断提示read已完成**，13310260：Answer F1 0.192→0.163，Evidence F1 0.158→0.233，回答27→30/40；不是全面提升。
- **7B基线已完成**，13318062：direct/rag/agent Answer F1=0.175/0.349/0.241。7B Agent回答38/40，但Evidence F1=0.143低于3B对照0.158；不是全面超越，7B RL未验证。
- **实验3首次搜索原问题已提交**：13349421，目录 `~/nova-papers-exp3`，代码433935c。尚未拿到运行状态/结果。
- **实验4额外仅作答回合已提交**：13349424，目录 `~/nova-papers-exp4`，代码433935c。尚未拿到运行状态/结果。
- 实验3/4是独立评估入口，不改生产agent或GRPO，不混入read提示/禁止重复搜索。两项新增回合与查询行为均记录，实验4记录额外token成本。

### 2026-10-23 最新提交状态（用户终端确认）

- **7B GRPO smoke已提交：13349738**，代码361c65c，目录 `~/nova-papers-grpo-7b`，最后看到PENDING/Priority。先7B token probe，再2步LoRA训练（G=2、limit=8）；不是正式训练效果实验。
- 7B使用独立venv、输出和端口，无等待3B的依赖；资源允许时可并行。A40上的7B训练显存需求仍待实测。
- 实验3任务13349421、实验4任务13349424仍为PENDING/Priority。
- **此次队列没有新的3B训练任务**：实验3/4是推理评估；13349479是已有3B LoRA重载验证，不是训练。已完成的3B两步训练是13310264，勿混淆。
- 以下RL消融计划已同步到本主项目笔记；7B提交不代表十项消融已经启动。

### 当前下一步：已有LoRA重载与监控验证

新脚本 `scripts/carc_verify_lora.sh` / `scripts/verify_lora_reload.py`：只读复用原训练venv与checkpoint，不安装升级，不重新训练；PEFT加载后逐张量对比保存的适配器，检查有限logits及生成，同时核对GPU UUID、采样及汇总。adapter开启/关闭各跑固定短提示，验证适配器实际生效，不当作质量评估。

**版本限制**：备份adapter_config的revision=null，训练报告未记录基础权重精确revision。重载仅离线使用缓存，记录实际snapshot，无法追溯严格证明与训练时revision完全一致。缓存缺失则失败，不偷偷下载最新版。

此验证不启动vLLM，因此不能证明训练结束时cumem_allocator CUDA错误已修复，也不消除TRL 1.16.0与vLLM 0.32.0兼容警告。后续需独立验证兼容组合。重载验证已提交：13349479，目录 `~/nova-papers-lora-reload`；最新队列中未出现，需查sacct和日志确认结果，不能视为已通过。

---

## 2026-10-23 RL 消融实验计划（全部为 PLAN，未提交）

**目标**：十项候选均保留，分阶段判断哪些训练设计能改善 QASPER；不一次性全量交叉组合。下面是实验设计清单，不表示配置开关已经实现，也不授权自动提交全部 GPU 作业。

**方法**：固定一个参考配置，每次只改变一个因素，分别训练并在同一评估集比较。实现时按 executing-plans 逐项推进，先检查配置与测试，再提交。技术栈沿用 Qwen2.5、LoRA、TRL/GRPO、vLLM、QASPER；PPO 是独立扩展而非现有脚本切换参数。

### 启动条件及共同评估协议

- [ ] 查回 LoRA 重载/监控任务13349479、推理实验3/4任务13349421/13349424的结果；保留已备份的两步checkpoint。
- [ ] 解决或明确隔离 TRL/vLLM兼容警告及训练退出时CUDA allocator异常，验证完整作业退出、权重重载、真实显存采样；两步smoke不能作为正式训练效果。
- [ ] 建立可重复参考运行：3B-Instruct、GRPO、LoRA、mask开启、固定生产工具流程（不混入四项推理实验）、当前检索器和配置。记录精确模型/tokenizer revision、代码提交、环境、数据ID、奖励权重、训练和生成预算。先测吞吐再冻结正式步数/样本预算；预算确定前不启动消融。
- [ ] 训练/开发/最终测试按论文划分避免泄漏；已有40题仅作历史回归诊断，不当最终独立测试集。用开发集选参数和checkpoint，最终测试集不参与筛选。
- [ ] 主指标保持Answer F1，另报Evidence F1、题型分项、answered/max_rounds/格式错误、read和search次数、token数、耗时、分阶段及整卡显存。可补规范化EM，但不替换F1；不把格式合规率或训练reward升高等同于回答质量提升。
- [ ] 每组同时评估训练前checkpoint与训练后checkpoint；先单seed筛选，候选结论用至少3个seed复验并报告均值/波动、样本量与配对差异。失败运行单独记录，不算零分模型。

### 十项候选及优先级

| ID / 优先级 | 单变量对照设计 | 运行阶段与限制 |
| --- | --- | --- |
| R1 / P1 | 格式奖励：当前完整奖励 vs 仅format权重置0；其他权重保持不变 | 两组独立RL训练。不重新归一化其他权重；记录实际奖励分布。另做correctness-only参考，不能与去format混称同一消融 |
| R2 / P1 | 中间证据奖励：当前完整奖励 vs 仅evidence权重置0 | 两组独立RL训练。当前QASPER证据段落奖励不是Search-R1的答案子串检索奖励；faithfulness、action保持固定，不声称精确复现论文 |
| R3 / P1 | GRPO group size：G=2 vs 4；G=8仅资源允许后追加 | 分别训练，固定每次更新的独立问题数并明确调整batch/累积配置；报告不同采样量与成本。不将TRL的G=1当合法替代，论文size=1是REINFORCE口径 |
| R4 / P1 | 训练数据量：100 vs 1000个训练问题（数据足够时） | 固定seed的嵌套子集、同一更新/生成预算，报告重复采样和实际曝光量；若另测固定epoch，单列实验，不能混淆数据量与训练量 |
| R5 / P2 | 检索top-k：1 vs 3 vs 5 | 分别改变训练rollout检索数量。主比较固定推理top-k=3以隔离训练因素；训练/推理同k的系统比较另报。保持上下文上限，记录截断/溢出和token成本 |
| R6 / P2 | 检索token loss mask：开启 vs 关闭 | 分别训练；只改变外部内容的loss参与，不改attention/context。先验证外部token的logprob/importance-ratio语义；现有占位logprob不能直接参与优化。无法构造有效对照就记为未实施，不硬开开关 |
| R7 / P2 | 模型规模：3B vs 7B；14B为P3候选 | 分别RL训练；固定模型家族、LoRA目标和任务协议。7B先单卡容量/吞吐预检，记录因显存不得不改变的配置；14B不默认可在当前A40脚本运行 |
| R8 / P3 | Base vs Instruct：先3B | 分别RL训练，固定奖励/预算；模板差异必须记录，验证Base能执行动作协议。比较各自训练前后增量及最终分数 |
| R9 / P3 | PPO vs GRPO：先3B-Instruct | 需新增并验证PPO训练与critic，不能用当前GRPO脚本冒充。匹配数据/rollout预算，另报critic显存与耗时；资源不足可保留计划不做 |
| R10 / P2推理、P3训练 | BM25 vs dense检索器 | 先固定checkpoint仅换推理检索器；随后才考虑分别训练并做train/test检索器交叉评估。固定论文范围、分块、top-k，避免混入语料变化；检索侧先测gold evidence recall |

P1先做奖励诊断，再做G和数据量；P2按收益与GPU预算推进；P3是可选扩展。不是必须把十项全部跑完才可交付项目。默认参考配置在整批单变量实验中不滚动变化；采用胜出改动时新建参考版本，必要时补交互实验。

### 实施入口与验收记录

- [ ] `scripts/train_grpo.py` / `backend/reward.py`：显式传入并保存奖励消融、group size、seed、数据子集等配置；检查去除某项奖励不会误改其他项或格式错误处理。
- [ ] `backend/token_rollout.py` / 检索模块：先验证top-k及mask对照的token对齐和数学含义。新增行为用小型针对性测试覆盖，不以mock代替GPU结果。
- [ ] CARC脚本：复用已验证的CUDA/nvcc准备、独立环境和输出路径、GPU UUID监控；每项先做小型GPU验证，再开展正式预算。不得改正在运行作业的源码或覆盖checkpoint。
- [ ] 每次在 `docs/EXPERIMENT_RECORD.md` 记录配置、job ID、实际代码、训练前后结果、seed、资源成本、失败原因与结论；只有拿到真实日志后才将PLAN改为完成。

### 文献依据及边界

- [Search-R1原论文 §5、附录C–H](https://arxiv.org/html/2503.09516v5)：mask、优化算法、初始模型、top-k、group size、模型规模。
- [后续实证研究 §4–6、附录D/E](https://arxiv.org/html/2505.15117)：格式/检索奖励、检索器、规模、数据量、长答案F1。论文设置的收益不是QASPER收益保证。
- 当前四项推理实验与本节RL消融分开编号；官方最后一次不执行搜索的生成逻辑不等于已经证明我们的额外作答提示有效。

### 2026-10-23 外部对标:R10(检索器)的优先级可能需要提前,但不是确定结论

**来源**:WebSearch核实,非本地复现。[paperlens](https://github.com/EnesUguzz/paperlens)——定位跟我们几乎一样("Agentic RAG over QASPER scientific papers: local models"),用官方QASPER Answer-F1/Evidence-F1指标,本地模型规模也接近(ministral-3:14b、qwen3:8b、llama3.1:8b)。他们报的分数:qwen3:8b Answer F1 0.403,llama3.1:8b 0.326,250题验证集上最终配置0.456。对比我们现在7B agent基线0.241——差距接近一倍。

**关键是:他们没有做RL训练,纯靠工程做到这个分数**——混合检索(BM25+向量检索)+ cross-encoder重排序 + 反思式多轮推理,而我们现在`qasper_corpus.py`只有纯BM25单一检索。

**这提示什么,以及提示不了什么**:
- 提示R10("BM25 vs dense检索器",现在定的是P2推理优先级)**可能值得提前验证**,因为检索质量本身可能是比"要不要RL训练"更根本的瓶颈——如果连检索都找不到对的段落,后面reward设计得再好,RL也只是在一个检索不到位的基础上做局部优化,天花板可能很低。
- **不能当作"RL训练没用"的证据**:paperlens的架构(混合检索+重排序+反思)跟我们的架构(纯BM25+GRPO多轮search/read)差异太大,分数差距到底多少该归因于检索质量、多少该归因于他们的prompting/反思机制工程、多少只是不同40/250题样本的噪声,现在完全不知道,没有做任何对照实验。
- 不能跨项目直接比较评分可靠性——他们的具体评测流程(是否同样处理unanswerable、是否用同一套normalize_answer实现)没有逐行核实,数字量级可以参考,精确到小数点后三位的比较没有意义。

**建议**:不改动R10现有P2优先级定义本身,但在实际推进P1(R1-R4)验证完、准备进入P2阶段时,**优先做R10,不要按表格原有顺序排在R5/R6之后**——这是本条新增的具体建议,其余P2条目顺序不变。

**R10实施时的工程参考(来自WebFetch核实paperlens README,不是实验结果,是别人架构里的现成经验,直接抄这几点能省掉自己从头摸索的成本)**:

1. **混合检索**:bge-m3向量检索 + BM25(他们用bm25s,我们已经有`rank_bm25`),用**Reciprocal Rank Fusion**融合两者排名——这正是R10要测的那个方向,可以直接参考这个融合方式,不用自己设计新的。
2. **一个反直觉的发现**:他们自己做过"加不加reranker(bge-reranker-v2-m3)"的消融,**结果提升很小**——说明他们架构里真正起作用的是混合检索本身,重排序锦上添花效果有限。做R10时应该**先验证纯混合检索(BM25+dense+RRF)能不能带来提升,reranker作为依赖于前者的后续子实验,不要一开始就把两者堆在一起测**,否则分不清提升来自哪部分。
3. **YAML配置继承做消融**(`extends: ../baseline.yaml`,每次只改一两个参数)——这是他们管理大量单变量消融实验的工程模式,跟我们R1-R10"单变量对照"的纪律是同一个思路。**这是可选的工具改进,不是研究发现**,现在实验数量还不多(现有脚本一个个手写管得过来),不建议现在就重构成这套配置继承框架,等以后R1-R10真的全面铺开、脚本数量多到维护吃力时再考虑引入。

**现在不建议直接动手做前两项**:R10本身排在P2,P1(R1格式奖励消融、R2证据奖励消融、R3组大小、R4数据量)一个都还没开始做,而且混合检索是要新增一整套embedding+向量索引的基础设施(不是改几行代码),工作量不小。按计划顺序,应该先把P1做完,再轮到R10,到时候再按上面两点顺序(先混合检索、后reranker)展开,不要现在跳过去做。

---

## 2026-10-20 续接：token rollout 已实现，真实 GPU 验证仍待执行

**状态分开记录**：实现和本地测试不等于训练成功。当前没有真实 vLLM HTTP probe 报告、GPU 训练日志或 LoRA checkpoint。CARC 登录用户名为 `xianhanh`，主机为 `discovery.usc.edu`；`xiangren_1987` / `rashidin_1753` 是 Slurm 计费 account。

### 已接入的实现

- `backend/token_rollout.py`：append-only token 轨迹；直接保留模型生成的 IDs/logprobs，只编码新增加的工具/聊天边界。`env_mask=1` 为模型 token，工具、格式错误反馈和聊天边界为 0。沿用两次连续格式错误退出、answer 退出、最大轮数退出，不额外强制作答。
- 总 completion/context 预算包括外部 token；不截断证据后仍声称已经看到全文。放不下完整工具消息时终止，并不计入未看到的证据。末轮不再追加模型不会读取的工具结果，这是与推理轨迹记录的一个明确差异。
- `_rollout_func` 已替换 stub。数据集 prompt 中携带 question ID，避免不同论文同名问题歧义；真正输入模型的 token prompt 仍是原系统提示、论文标题摘要和问题。
- HTTP client 拒绝缺失 token IDs、非 `token_id:N` 格式、长度不匹配或无效 logprobs。支持整数 token prefix 和每轮预算。**HTTP 路径仍未连接真实服务器验证。**
- 修复 reward 的 `<answer>Yes</answer>` 标签解析；训练只对模型输出评分，不把 `<information>` 中的文本当成答案；重复检索动作按轨迹计分。
- 训练入口使用 LoRA，默认仅 2 个 optimizer steps / G=2，用于 smoke test，不是完整训练配置。保存前检查可训练参数确实改变；零方差导致完全未更新时会明确报错。

本地真实 Qwen2.5 tokenizer 检查已通过：三轮 search→read→answer，325 个 prompt tokens，112 个 completion tokens，其中 70 个外部 tokens；每轮生成前缀与最终训练序列逐 token 相等。模型输出仍由固定测试样例提供，这不是模型推理测试。

### 对旧笔记的两项纠正（直接检查 TRL v1.16.0 代码）

1. `RepeatSampler.mini_repeat_count = num_generations` 已在数据加载时重复样本。rollout 回调收到的是重复后的逐行输入，必须每行返回一条轨迹，不能再 `run_group(G)`。其接口 docstring 与采样执行路径描述存在歧义，以实际源码与集成验证为准。
2. 标准 `vllm serve` 的 HTTP 客户端不会自动跟随 optimizer 更新权重。训练保留 `vllm_mode="colocate"` 并通过 trainer 的 `vllm_generation.generate()` 获取 token/logprobs，让 TRL 负责同步权重；HTTP client 用于独立服务 probe。当前明确限制单进程，避免不同轨迹长度引发分布式 collective 死锁。

固定 `trl==1.16.0`；升级时重新验证实验性接口。源码参考：
- https://github.com/huggingface/trl/blob/v1.16.0/trl/trainer/grpo_trainer.py
- https://github.com/huggingface/trl/blob/v1.16.0/trl/generation/vllm_generation.py

### CARC 失败历史与下一步

用户提供日志确认：job 13263736 的 P100（sm_60）不受当时 torch 2.13.0+cu130 内核支持；vLLM 因 `no kernel image` 退出。此前 40/40 exception 的全零结果是服务启动失败，不是有效模型评估。13264048 的 A40 作业状态还需从用户终端检查。

从已应用本次改动的仓库根目录提交：

```bash
sbatch --account=xiangren_1987 scripts/carc_train_smoke.sh all
```

脚本只在分配的 A40 计算节点安装隔离 `.venv-grpo` 环境和运行 GPU 工作；先启动带 `--return-tokens-as-token-ids` 的服务，运行 `python -m scripts.check_vllm_rollout`，通过后关闭服务，再启动同进程 GRPO。失败不会继续运行后续步骤。依赖版本是待 GPU 实测的候选环境，不声称已通过 CUDA 验证。

结果位置：`eval/training/JOBID/`（环境、服务日志、真实 token probe）和 `checkpoints/grpo-smoke-JOBID/`（参数变化报告、phase metrics、通过参数更新检查后的 adapter）。

### 给 A 的性能口径与剩余工作

`phase_metrics.jsonl` 的两个互斥阶段：
- `rollout_and_scoring`：权重同步、模型生成、工具执行、reward、旧策略 logprob scoring；
- `parameter_update`：上述阶段之外的训练 forward/backward、optimizer 和相关 step 开销。

记录 CUDA 同步后的耗时和 **训练进程 PyTorch allocator** 的 allocated/reserved 峰值；这些不代表整个 GPU 峰值，也不包含另一个进程的所有 vLLM 内存。CARC 脚本另用 `nvidia-smi` 每 200ms 采样已分配 GPU 的整卡显存，按 phase 的 wall-clock 时间窗口汇总到 `phase-summary.json`；这是采样峰值，包含同卡其他进程，不能声称精确的本进程峰值。真实采样结果仍需作业运行后再交 A。当前串行 rollout 配合 TRL sleep(level=2)，可能每轮重新同步权重，正确但耗时；测过后再做性能优化。

真实 HTTP / 两步 GPU GRPO / checkpoint 和耗时显存结果都未获得，不能写成训练已完成。TRL 接口已逐项核对固定版本源码；真实 trainer optimizer 更新仍待 CARC 实测。本机没有完成额外的 tiny-model TRL 运行验证。

---

## 2026-10-20(进行中)token/logprobs缺口:第一步做完了,核心的多轮累积循环还没写

### 已完成:`backend/vllm_training_client.py`——能拿到真实token id + logprobs了

调用vLLM标准的`/v1/completions` + `logprobs`参数(没有用更新但接口还在频繁改动的`/inference/v1/generate`,查了vLLM仓库最近的PR记录,那个接口的返回格式这几个月还在被持续调整,不够稳)。需要vLLM服务端启动时加`--return-tokens-as-token-ids`参数,这样`logprobs.tokens`里每一项会是`"token_id:1234"`这种字符串,解析出真实整数token id,不用再额外过一遍tokenizer。

4个单元测试(mock HTTP响应)全过,**但这整个方案从没跟真实vLLM服务器跑过一次**——因为写这段的时候没有GPU/vLLM可连,request/response格式是查vLLM文档+源码PR记录拼出来的,逻辑上应该对,但没有实测验证过,等CARC上真有vLLM服务跑起来,第一件事应该是拿这个client测一次,确认字段名、解析逻辑是不是真的对得上。

### 还没做:把这个client接进一个"token级别的多轮循环"里

这是接下来最大的一块工作。现在`agent_loop.run_episode`是在**消息列表**(`messages: list[dict]`)层面做多轮循环的,但GRPO训练要的是**单条、完整的token序列**(`completion_ids`)+ 每个位置标记"这是模型自己生成的(算梯度)还是工具塞进去的(不算梯度)"的`env_mask`。这意味着需要新写一个**跟`run_episode`逻辑平行、但在token层面操作**的循环,具体要做的事:

1. 每一轮调`RolloutModelClient.generate()`拿到模型这一轮输出的token_ids + logprobs(这部分`env_mask`标1)
2. 用`parse_action()`解析这一轮输出(复用`agent_loop.py`现成的)
3. 执行工具,拿到`<information>...</information>`文本
4. **这部分文本需要自己用tokenizer编码**(`RolloutModelClient`不包含tokenizer,这是故意的,职责分离),这部分token的`env_mask`标0,`logprobs`随便填(比如0.0,反正不参与梯度)
5. 把两部分token_ids拼起来,继续下一轮,直到答案/轮数耗尽/连续格式错误(复用`agent_loop.py`的终止条件逻辑)
6. 最终把累积好的`completion_ids`/`logprobs`/`env_mask`按TRL的`rollout_func`要求的格式返回(具体字段要求见下面那条"`rollout_func`真实接口"笔记)

**这一步目前完全没写代码**,只是想清楚了要做什么。预计是`scripts/train_grpo.py`里`_rollout_func`那个`NotImplementedError`占位要被这整套逻辑替换掉。

## 2026-10-20 调研:三个同类项目,分别当什么参考(已逐条核实属实)

### 1. [jmhb0/PaperSearchQA](https://github.com/jmhb0/PaperSearchQA)——最像的"同行研究",数据处理思路可参考

EACL 2026论文配套代码,标题"Learning to Search and Reason over Scientific Papers with RLVR",跟我们立项思路几乎一样,也是用RLVR训练agent学会"在科学论文里搜索+推理"。区别:他们做的是生物医学论文摘要(1600万篇摘要语料库+6万条QA),我们是QASPER全文论文(1585篇,段落级证据标注)。

**顺手核实时多挖到一个有用的数字**(原讨论里没提):他们自己报的结果——Search-R1训练出来的agent,在Qwen2.5-7B上,PaperSearchQA这个数据集上51.0% vs RAG基线36.5%(**+14.5个百分点**),而且泛化到另一个数据集BioASQ上也有+15.1个百分点的提升。**这是一个"RL训练相比RAG基线,理论上能有多大提升空间"的参照系**——我们现在agent模式(0.224)还不如rag(0.372),这组数字说明训练好之后"反超"是有先例的,不是天方夜谭。

代码仓库README明确写"all artifacts compatible with the Search-R1 codebase"——跟我们现在照着Search-R1设计的`agent_loop.py`/`reward.py`,理论上兼容。数据生成流程(怎么从论文自动造问答对)值得参考,跟我们"数据整理"这块可能有互相印证的地方。

### 2. [rStar-RL/LoongRL](https://github.com/rStar-RL/LoongRL)——不是论文QA项目,但可以当"分数标尺"

ICLR 2026 Oral,做的是通用长上下文推理RL框架(GRPO critic-free训练),不是专门做论文问答的。但**查了它的README结果表,确认QASPER确实是它的基准测试集之一**:LoongRL-7B在QASPER上63.6分,LoongRL-14B是64.5分(具体指标口径大概率跟我们的Answer F1不是一回事,不能直接比)。

**怎么用这两个数字**:当一个"更大模型、训练更充分之后大概能到什么量级"的粗略参照,帮我们判断自己Qwen2.5-3B跑出来的0.253这类基线数字,离"训练好之后大概能到哪"还有多少空间——不是精确对标,是个数量级感觉。

### 3. [Future-House/paper-qa](https://github.com/Future-House/paper-qa)——产品形态参考,训练方法完全是另一条路

真·开源产品(star数、维护活跃度都高),做的是"对一堆论文PDF提问,拿到带引用的准确答案",而且已经支持"agentic RAG"(模型自己迭代式改查询、反复检索)。**但核实过README,它默认直接调用GPT-4级别的大模型做推理时工具调用(`gpt-4o-2024-11-20`),不训练小模型、不用RL**——方法论上跟我们(RL训练一个3B小模型)是两条完全不同的路。

**怎么用**:不适合当训练方法参考,但适合当"demo最终用户体验应该长什么样"的产品参考(比如它怎么展示引用、怎么设计agentic检索的交互)。

### 一句话总结怎么看这三个

PaperSearchQA抄数据处理思路(而且代码跟Search-R1兼容),LoongRL当分数标尺,paper-qa看产品形态——真正在"训练方法"这件事上能抄代码的,还是Search-R1那条线最直接,因为我们现在的`agent_loop.py`/`reward.py`设计本来就是照着它的思路来的。

## 2026-10-20 提案:"4(数据/验证部分)"(A)和"4(训练部分)"(B)的接口边界

写这个是为了在A真正开始做之前,先把两人各自产出的东西对齐,避免各写各的、到时候拼不上。**这是提案,需要A确认,不是已经定案**。

### B这边已经搭好的(`scripts/train_grpo.py`,骨架)

- `build_dataset_rows(corpus, split)`——给GRPOTrainer用的训练集,一行一道题:`{"prompt": 问题文本, "paper_id", "question_id", "answers": QASPER标准答案列表}`
- `_reward_func(completions, **kwargs)`——包装`composite_reward`,接收`answers`(必须)和`question`/`evidence_passages`/`evidence_para_ids`(可选,`rollout_func`会提供)
- `_rollout_func(prompts, trainer)`——**目前是`NotImplementedError`的stub**,因为真正调用`run_group`产出token_id+logprobs这件事,需要先解决"现在的`vllm_client.py`只返回文本、没有logprobs"这个技术缺口(见上面那条笔记),这部分**是B自己的工作,不需要A动手**

### 建议A产出的东西(她那部分"验证reward.py能否对真实AgentTrajectory算分")

不需要碰token_id/logprobs这些训练底层的东西——A的产出更偏"验证+分析"性质,建议格式:

```json
{
  "question_id": "...", "paper_id": "...",
  "rollouts": [
    {"raw_output": "...", "final_answer": "...", "evidence_para_ids": ["p_3"], "reward": 1.82},
    {"raw_output": "...", "final_answer": "...", "evidence_para_ids": [], "reward": 0.35}
  ],
  "advantages": [0.71, -0.71]
}
```
也就是:拿`run_group()`跑出来的一组`AgentTrajectory`,对每条构造`RolloutSample`(`raw_output`取轨迹最后一轮输出,`evidence_para_ids`取轨迹里`read`过的段落id),调`composite_reward`算分,再用`group_advantages`算组内优势——**这一步本身就是在验证reward.py能不能跑通真实轨迹**,产出的JSON可以直接用来人工检查"分数给得对不对"。

**边界说清楚**:A这边产出的是"验证用的评分数据",不是直接喂给`GRPOTrainer`的`prompt_ids/completion_ids/logprobs`——后者是B要解决的token级别问题,两边不会重复做同一件事。

---

## 2026-10-20 调研:`rollout_func`真实接口(直接查的TRL源码,不是文档推测)

来源:`trl/trainer/grpo_trainer.py`(pip装的trl包里的源文件)

### 真实签名

```python
RolloutFunc = Callable[[list[str], "GRPOTrainer"], dict[str, Any]]
```

接收:当前进程分到的那批`prompts`(list[str])+ trainer实例本身。

**必须返回的字段**(源码里有运行时校验,缺了会直接抛`ValueError`):
```python
required_keys = {"prompt_ids", "completion_ids", "logprobs"}
```
可选返回`"logprob_token_ids"`(跟logprobs同形状)。**其他任何额外字段,都会原样转发给reward函数**——这意味着我们可以把`evidence_para_ids`这类reward要用的信息,直接塞进rollout_func的返回dict里,不用另外设计传递通道。

**一个重要发现:`env_mask`字段**——官方注释:"Support custom env_mask from rollout_func (e.g., for environment feedback masking). Internally treated as tool_mask - marks model tokens (1) vs external tokens (0)"。这就是我们在Search-R1笔记里记的那个`info_mask`机制(工具返回内容不参与梯度更新)在TRL里的**官方对应实现**——不用自己从零写,用这个字段就行。

**约束**:"responsible for returning the correct number of completions per prompt"——每个prompt要生成几条rollout(对应GRPO的组大小`num_generations`),是`rollout_func`自己的责任,TRL不会帮忙重复采样。这意味着我们的`rollout_func`里应该直接调`run_group(...)`(它本来就是"同一题生成G条轨迹"的实现),不用自己再写一层采样循环。

### 发现一个必须先解决的真实技术缺口

GRPO训练要的是**token id + 每个token的logprob**,跟训练用的分词器精确对应——**不是解码后的文本字符串**。但我们现在的`backend/vllm_client.py`调的是vLLM的`/v1/chat/completions`,只拿到解码后的文本(`choice["message"]["content"]`),**完全没有拿token id和logprobs**。

这是搭训练脚本之前必须先解决的前置工作:需要扩展model client这一层(或者新写一个训练专用的client),让vLLM在返回里带上`logprobs`(vLLM的`/v1/completions`接口支持`logprobs`参数,能拿到每个token的概率;`/v1/chat/completions`这条路径对这个支持没那么直接,可能需要换回`/v1/completions`,或者查vLLM chat接口是否也支持)。这一步还没做,是下一步要具体解决的东西,不是搭个骨架就能绕过去的。

---

## 待办:训练跑起来之后要发给A的东西(2026-10-20 A提出)

A提了她打算在自己那部分(3A/5)顺手加一些推理性能分析,但需要B这边训练出了结果之后给她东西才能做。等训练真正产出东西了,记得发:

1. **LoRA checkpoint**——训练完成后给A一份,她要在Colab上对比"合并权重"vs"vLLM直接加载LoRA"两种部署方式的延迟和显存,结果供B决定Demo(6)最后怎么接入
2. **训练过程的性能数据**——显存占用 + 每步耗时,**rollout阶段和参数更新阶段要分开记**(不要混在一起算总耗时),A要拿去整理分析

A自己那边会做的(不需要B现在配合,只是记录背景):在Colab上测`run_group`在不同并发数/batch配置下的吞吐速度,测完把结果和推荐配置发给B,由B决定训练时要不要采用。

## 待办:job 13264048("别重复搜索词"实验)出结果之后,还要做2个新实验

这是agent模式"不如rag"那三条原因里,还没落地的另外两条(前提:都是单变量实验,不要跟"别重复搜索词"这条混在一起测,也不要互相混测,一次只改一个变量):

1. **几乎不用read精读**——方案:在`search()`返回的截断段落后面加提示,比如"(此段被截断,完整内容用`<read>`查看)",引导模型意识到还有更多内容可看。现在完全没实现,连提示词草稿都没写。
2. **自己改写的搜索词比原问题还差**——方案:第一轮`<search>`固定用原始问题本身去查(不让模型自己改写),只有第一次没查到有用信息,后续轮次才交给模型自己想关键词。现在完全没实现。

**顺序**:先看job 13264048的结果,确认"别重复搜索词"这条有没有用;然后这两条再单独各跑一次实验,不要跟它或者跟"轮数耗尽强制作答"(见下面)混在一起一次测完。

---

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

## 2026-10-20(最高优先级):`reward.py`缺"答案对不对"这一项 —— ✅ 已修复

**已确认不影响当前基线实验再动手的**:`scripts/run_baseline.py`不import `backend.reward`,基线测试(direct/rag/agent)自己单独算分,这个改动跟正在排队/跑着的实验完全无关,可以随时改。

**修复方式**:把`answer_f1()`/`evidence_f1()`从`scripts/run_baseline.py`挪到新建的`backend/qasper_metrics.py`,两边(`run_baseline.py`和`reward.py`)共用同一份实现,不是各写一份。`reward.py`里新增`correctness_reward`(对比QASPER标准答案)和`evidence_reward`(对比标准证据段落),`RewardWeights`加了`correctness=1.5`(权重最高,匹配Search-R1"答案对不对是压倒性主信号"的设计)和`evidence=0.5`。`RolloutSample`新增`evidence_para_ids`字段存轨迹里实际检索/读到的段落。`composite_reward`现在需要多传一个`answers`参数(QASPER的标准答案列表)。17个reward测试+全套53个测试通过。

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

## 2026-10-23 独立7B训练任务准备

用户要求保留已提交3B任务，另提交7B，不设置afterany依赖。新增 `scripts/carc_train_7b_smoke.sh`，A40单卡、48G系统内存、2小时；7B HTTP probe后做2步GRPO、G=2、limit=8。独立venv、端口、日志、checkpoint，不改3B入口。vLLM训练预算0.45为待验证起点，7B真实GPU训练容量尚未确认；失败保留完整日志，不自动缩短上下文以假装同条件成功。当前只是smoke，不是效果比较；正式3B/7B同预算pilot另行设置。脚本已提交：13349738；用户最后提供的状态为PENDING/Priority，真实训练结果待查。
