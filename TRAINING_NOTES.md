# 训练笔记(GRPO / Agentic RL)

记录"4(训练部分)"相关的技术调研和设计决策,中文,随时间持续补充。

**主项目位置**：`/Users/hxh/Desktop/nova-papers` 是唯一维护入口，代码、计划和实验记录统一在这里更新。文中的 `~/nova-papers-grpo-7b` 等路径是 CARC 远程作业的独立运行目录，不是本机主项目；运行中的远程目录不搬移。

## TODO(2026-10-27维护,每次有进展就更新这一节,不要翻到下面历史记录里找最新状态)

### 🔴 高优先级 / 正在等结果

- [x] **R1-R4真实rollout效果检查(train split)**:job `13499835`(r1-format)/`13499836`(r2-evidence)/`13499837`(r3-g4)/`13499838`(r4-limit1000)已完成,`13500255`(ref补测)/`13500256`(r1-format,**误把split传成默认train,跟13499835重复,没测到validation**)。结果(20题×G=5,temp=1.0,seed=599,100 rollouts,train split,2026-10-28实测):

  | | mean reward | Answer F1 | Evidence F1 | mean rounds | answered | format_error | max_rounds |
  |---|---|---|---|---|---|---|---|
  | ref | 0.604 | 0.197 | 0.088 | 3.06 | 59 | 31 | 10 |
  | r1-format(format=0) | 0.593 | 0.169 | 0.142 | 3.00 | 57 | 34 | 9 |
  | r2-evidence(evidence=0) | 0.628 | 0.202 | 0.095 | 3.14 | 69 | 25 | 6 |
  | r3-g4(group 2→4) | **0.715** | **0.237** | 0.133 | 2.91 | 66 | 29 | 5 |
  | r4-limit1000(100→1000题,步数不变) | 0.571 | 0.170 | 0.116 | 3.11 | 53 | 32 | **15** |

  **初步结论(train split,待validation split确认才能定案)**:r3-g4全面最好(reward/Answer F1/Evidence F1都领先,max_rounds最少)→优先级最高,建议后续训练默认改G=4。r4-limit1000全面最差,尤其max_rounds暴涨→"扩数据不扩步数"会稀释学习,不能照搬。r2-evidence反直觉不降反升(format_error全场最低,answered最多)→evidence reward可能在跟其他reward项抢梯度,值得验证是否该降权重而非直接保留。r1-format是混合信号(Answer F1降、format_error却也升、但Evidence F1大涨)→不是干净的"去掉约束就变差"的故事,不下结论。
- [ ] **上面是train split(模型训练时见过的题),r3-g4/r4-limit1000这种大效应量大概率是真信号,但r1-format/r2-evidence这种小/混合差异必须等validation split确认不是过拟合假象**。**第二次validation-split检查(`13534449`~`13534453`)也白跑了,原因是新的一类坑**:提交这批job时`~/nova-papers-rl-ablation`这个worktree本地的`scripts/carc_check_format_rate.sh`是一份很旧的本地文件(相对这个worktree当时的HEAD`f0bafc4`——比最早写这个脚本的`ef5d5bc`还早——是"新增"状态,从来没同步过任何更新),旧脚本根本不认`[split]`参数,传了也白传,又跑了一次train split(`grep "^Done. Compare" format-check_13534449.log`打出来的路径是`eval/rollout_scoring-ref/summary.md`,没有`-validation`后缀)。已用`git checkout origin/main`把这个worktree同步到`8bbf323`,需要第三次重新提交全部5个:
  ```bash
  cd ~/nova-papers-rl-ablation
  scancel 13534453   # 还在排队、会用旧逻辑跑出来的那个,取消重交
  sbatch scripts/carc_check_format_rate.sh checkpoints/ref ref validation
  sbatch scripts/carc_check_format_rate.sh checkpoints/r1-format r1-format validation
  sbatch scripts/carc_check_format_rate.sh checkpoints/r2-evidence r2-evidence validation
  sbatch scripts/carc_check_format_rate.sh checkpoints/r3-g4 r3-g4 validation
  sbatch scripts/carc_check_format_rate.sh checkpoints/r4-limit1000 r4-limit1000 validation
  ```
  **新教训**:`git status --porcelain`只检查"相对这个worktree自己的本地HEAD有没有未commit的改动",查不出"这个worktree的HEAD本身是不是落后于origin/main"。以后在任何worktree提交job前,不管看起来要不要,先`git checkout origin/main`一次,不能假设之前同步过就一直是最新的。
- [ ] validation split结果出来后,把job ID和summary.md数字填进下面表格,再据此在"R1-R4结果解读表"下最终结论。
- [ ] 拿到validation结果后,把最终结论写回`docs/EXPERIMENT_RECORD.md`,并据此决定reward权重要不要调整(比如r2-evidence若validation也不降,考虑调低evidence权重;r3-g4若validation也领先,后续训练默认改G=4)。
- [ ] 验证`carc_train_smoke.sh`的路径bug修复本身——下次提交用`-- --output-dir ...`覆盖默认路径的job时,确认收尾步骤不再误报FAILED。

### 🟡 方法论问题——已经修好脚本,第一次提交踩坑,第二次(2026-10-28)已重新交对

- [x] **ref/R1-R4的真实rollout检查默认用`--split train`**——确认本地`corpus/qasper`其实已经有validation(1005题)、test(1451题)两个干净的held-out split,训练完全没碰过。`carc_check_format_rate.sh`加了`[split]`可选参数(默认`train`向后兼容)。第一次提交(`13500255`/`13500256`)忘了实际传`validation`参数,白跑了,已撤销剩余3个、重新提交全部5个(`13534449`~`13534453`,见上方🔴清单),这次目录对、参数带上了。**教训:提交后要用`grep "^Done. Compare"`确认log里的输出路径带没带预期的后缀,不能只看sacct的COMPLETED状态——exit 0不代表测的是你以为的那个split。**
- [ ] **组合实验/R10用`--temperature 0`贪婪解码,换seed不会产生新信息**——`carc_eval_agent_ablation.sh`加了`[temperature] [seed]`可选参数(默认`0 42`向后兼容)。第一次复验`13500267`(seed=601)/`13500268`(seed=602)**也白跑了**,跟上面R1-R4 validation split同一类坑:提交时`~/nova-papers-exp-combined`这个worktree本地的脚本还是没有`[temperature][seed]`支持的旧版本,传的`1.0`/`601`/`602`被忽略,结果是`control`/`evidence_f1`等数字跟最早的`13467754`(temperature=0,seed=42)**精确到小数点后14位完全相同**——不可能是真的temperature=1.0跑出来的。意外收获:三次跑数字完全一致,反而确认了这套pipeline在temperature=0+固定seed下是完全可复现的(环境没有隐藏的不确定性),但回答不了"换随机种子结果稳不稳"这个本来想问的问题。`~/nova-papers-exp-combined`后来已经用`git checkout origin/main`同步到`eceb406`(带新脚本),重新提交:
  ```bash
  cd ~/nova-papers-exp-combined
  sbatch scripts/carc_eval_agent_ablation.sh combined 1.0 601
  sbatch scripts/carc_eval_agent_ablation.sh combined 1.0 602
  ```
  （结果出来后先用`cat comparison.json`确认两次跑出的数字跟`13467754`不再完全相同，再把job ID和comparison.json数字补进来，跟原始`13467754`和彼此比较方差）

### 组合实验要测多少题才能算统计显著(2026-10-27实测算出来的,不是估的)

用`13467754`真实的40题配对数据(`control`/`treatment`两份`agent.jsonl`,按`question_id`配对算Answer F1差值)算样本量:

```
n=40, mean_diff=0.0297, std_diff=0.2209
按95%置信度、80%把握的标准配对检验公式:required_n ≈ (1.96+0.84)² × std_diff² / mean_diff² ≈ 436题
```

**结论**:436题是这个效果量级(+0.03)要达到统计显著的最低门槛,40题远远不够,436在QASPER的held-out范围内能做到(test split有1451题,validation有1005题)。

**2026-10-27更新:脚本已经改完,本地测过,可以提交了**:
- [x] 新脚本`scripts/make_held_out_sample.py`:从QASPER **test split**(训练和checkpoint选择全程没碰过,干净的held-out)按`score_real_rollouts.usable()`同一套筛选规则(标注一致、无图表证据、可回答的题必须有证据段落)抽样,本地跑出来的结果是**480题**(`free_form 43 + unanswerable 79 + yes_no 108 + extractive 250`,前三类是该类型test split里全部可用的数量,extractive从487个可用里抽了250个凑够480),高于436题的统计显著性门槛。用法:`python -m scripts.make_held_out_sample --out eval/sample_questions_test500.jsonl`。
- [x] `scripts/carc_eval_agent_ablation.sh`加了`[samples-file]`(默认还是原来的40题文件,向后兼容)和`[workers]`(默认1,向后兼容)两个可选参数,去掉了硬编码的"必须是40题"断言,改成跟samples文件的实际题数动态比对。本地`bash -n`语法检查过。
- [x] **已在CARC上提交,排队中(2026-10-28)**:`job 13534856`。第一次提交(`13534505`)在worktree还是detached HEAD、`git pull`失败的情况下交的,脚本和样本文件都是旧的/不存在,已`scancel`;`git checkout origin/main`同步到位、`make_held_out_sample.py`跑出480题文件后重新提交,这次是有效的480题统计显著性确认实验,结果未知。命令记录:
  ```bash
  cd ~/nova-papers-exp-combined
  git checkout origin/main   # detached HEAD worktree不能直接git pull
  python -m scripts.make_held_out_sample --out eval/sample_questions_test500.jsonl
  sbatch scripts/carc_eval_agent_ablation.sh combined 0 42 eval/sample_questions_test500.jsonl 8
  ```
  （`temperature=0`、`seed=42`跟原始`13467754`保持一致,这次唯一变化的变量是样本量）

### 🟢 已验证有效,等上面两项解决后再推进

- [ ] **组合实验(first-search+final-answer)并入`agent_loop.py`生产默认行为**——已验证单次40题结果为正(Answer F1 0.192→0.222,Evidence F1 0.158→0.175),但故意暂缓:`score_real_rollouts.py`直接调`agent_loop.run_group`,现在改会让R1-R4的结果跟已记录的ref基线不可比。等R1-R4结果出来、且上面"🟡temperature方法论问题"有定论后再动代码。

### ⚪ P2消融(计划里的,还没开始)

- [x]/[ ] **R5(推理侧版本)**:代码已写完、本地测过、已commit+push(`8bbf323`),第一次在CARC上提交(`13535156`,2026-10-28)排队中,结果未知。做法:`QasperCorpus`/`HybridQasperCorpus`新增`search_limit`构造参数(默认3,向后兼容),`search()`不传`limit`时跟着实例的`search_limit`走(`agent_loop.py`不用改);`run_baseline.py`加了`--search-limit` CLI参数;新脚本`scripts/carc_eval_search_limit.sh`对`checkpoints/ref`跑`search-limit=1/3/5`三次agent模式评估,汇总成`comparison.json`。**这不是原计划里"训练侧改变rollout检索数量"的版本**(那个要对每个k值重新训练,成本高很多),是用已训练好的ref checkpoint在推理时切换top-k,只能回答"已训练模型对检索宽度敏不敏感",回答不了"训练时该用多大top-k"这个更贵的问题——如果推理侧结果有明显信号,再考虑要不要做训练侧版本。
- [ ] **R6**:检索token loss mask开关消融(外部内容要不要参与loss)。需要先验证`token_rollout.py`里外部token的logprob/importance-ratio语义是不是真的有意义,构造不出有效对照就按计划记为"未实施",不硬开。
- [ ] **R10 P3部分**:P2推理侧的BM25 vs 混合检索对比已经做完(`13467421`),但"真的拿混合检索器去训练"这部分(P3)还没做。
- [ ] Answer F1/Evidence F1完整评估(不只是格式错误率)——按分工交给A,她的`run_baseline.py`稍改指向训练后checkpoint即可测,B这边职责到"checkpoint能被serve起来"为止。

### ⚫ P3可选扩展(优先级最低,资源富余再做)

- [ ] **R7**:模型规模3B vs 7B完整RL消融(目前只做过7B的2步smoke test,不是正式消融)。14B默认不在当前A40脚本可跑范围内。
- [ ] **R8**:Base模型 vs Instruct模型对比(先3B)。
- [ ] **R9**:PPO vs GRPO对比(先3B-Instruct),需要新增并验证PPO训练+critic,不能拿现有GRPO脚本冒充。
- [ ] **队友提议的"给`<answer>`加置信度/引用来源子标签"**——2026-10-27讨论过,建议当成独立的新消融(比如编号R11)来测,不要现在就改动作语法混进正在跑的R1-R4,而且"置信度"这部分有reward hacking风险(模型可能学会永远喊高置信度),要留意。

### R1-R4结果解读表(train split已套用,validation split出来后复核)

| 实验 | 如果结果接近/好于ref | 如果结果明显差于ref(更接近未训练基线33/100格式错误率) | **train split实测落在哪一边** |
|---|---|---|---|
| R1(format=0) | 格式奖励多余,靠"答对"驱动就能自然学会守格式,可以简化reward去掉这项 | 格式奖励必要,去掉后格式错误率会往基线方向回弹,应保留甚至训练早期加大权重 | **偏向"必要"一边,但不干净**:reward和Answer F1都比ref差(Answer F1降14%),format_error也确实略微往基线方向回弹(31→34)——按这张表的标准,这组数据更接近"格式奖励必要,应保留"这一边,不是"格式奖励多余"。搅乱这个故事的是Evidence F1反而涨了61%,所以仍然不下定论,等validation确认是不是真信号。**旁证(2026-10-29,A独立测试,细节见`docs/EXPERIMENT_RECORD.md`"2026-10-29"节,口头转述、无job ID,严谨度低于本表其他行)**:A用改提示词措辞的方式做了对照,格式错误终止次数降了,但Answer F1没跟着涨——这说明的是"单靠提示词让格式合规,不能保证内容质量变好",跟R1测的不是同一件事(R1测的是"去掉训练时的格式reward会不会让整体变差"),不能简单说两者互相印证"格式没用"；合起来看反而更像是提示A的发现和R1的发现指向不同机制:格式达标本身不直接带来答案变好,但训练时的格式reward信号本身可能通过别的方式(不是单纯靠减少格式错误)帮助了整体学习——这个假说还需要更多证据,不是现在就能下的结论 |
| R2(evidence=0) | 证据奖励多余,"答对"本身会带动模型找对证据,可以简化 | 证据奖励必要,去掉后可能靠瞎蒙也能答对但引用段落错(幻觉风险更高),必须保留 | **接近/好于ref**:reward、Answer F1都略升,format_error全场最低,answered最多——反直觉,不支持"证据奖励必要"这个假设,但要validation确认不是train-split过拟合 |
| R3(group=4) | 值得:多花1倍时间换明显更好的结果,后续训练默认改用G=4 | 不值得:噪声没有明显改善,继续用G=2,省下时间跑更多seed/消融 | **明显好于ref**:reward+18%、Answer F1+20%、Evidence F1+51%,全指标一致——目前最值得采纳的改动 |
| R4(数据=1000) | 支持"100题训练导致死记硬背"的猜测,后续训练换更大数据池 | 100题本身不是瓶颈,但这组只跑了0.1个epoch,结果一般时不能排除"只是还没训练够",需要同epoch数对照才能下定论 | **明显差于ref**:全指标下降,max_rounds从10暴涨到15——符合"步数不变、数据变多=每题学习量被稀释"的解释,不支持单纯扩数据 |

---

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

- [x] `scripts/train_grpo.py`：`--zero-format-reward`/`--zero-evidence-reward`（互斥）已实现（commit 76acb0b），只改对应权重为0，其余权重不变，解决后写进`training_verification.json`的`reward_weights`字段。顺手修了一个之前没发现的真bug：`_reward_func`里本地`weights`变量之前从没传进`composite_reward(...)`（它自己有默认`RewardWeights()`参数），所以这个变量实际只影响了"未作答时撤销correctness/faithfulness分"和action调整那两处，主体composite分数一直在用默认权重，跟R1/R2要测的东西完全脱节——加消融测试时立刻测出来了。group size（R3，`--num-generations`）、数据子集（R4，`--limit`，训练集实际2593题够用）CLI本来就有，不用改代码。seed暂未单独加CLI（目前`GRPOConfig(seed=42)`写死），多seed复验时再加。
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

## 2026-10-26 R1-R4代码blocker排查与修复

**R3(group size)、R4(数据量)没有代码blocker**:`train_grpo.py`的`--num-generations`、`--limit`早就存在,机制上随时能跑。训练集(`corpus/qasper/questions.jsonl`按`split=train`筛)实际有2593题,R4要测的"1000题"够用。

**R1(format奖励消融)、R2(evidence奖励消融)之前真的有代码blocker,现在修了**(commit `76acb0b`):加了`--zero-format-reward`/`--zero-evidence-reward`(互斥),只把对应权重置0,其余不变,解决后写进`training_verification.json`的`reward_weights`字段。

**顺手修了一个之前没发现的真bug**:`_reward_func`里本地`weights`变量之前从没传进`composite_reward(...)`——`composite_reward`自己有默认的`weights=RewardWeights()`参数,所以这个变量实际上只影响了"未作答时撤销correctness/faithfulness分"和action调整那两处,**主体composite分数一直在用默认权重,跟R1/R2要测的东西完全脱节**。写消融的回归测试时断言的差值不对,立刻测出来了——没有这个测试,这两个flag会悄悄什么都不做。已修好(传`weights=weights`),默认行为(不加消融flag)不变。

**R1-R4共同的blocker,还没解决**:消融计划自己定的"启动条件"要求先建立一个固定的参考训练配置(测真实吞吐、冻结正式的训练步数/样本预算),我们之前只跑过2步/8题的smoke test,不算正式预算。为此把`carc_train_smoke.sh`加了额外参数透传功能(commit `3a7574d`,语法:`sbatch scripts/carc_train_smoke.sh train -- --limit 100 --max-steps 10`),2026-10-26提交了一次吞吐探测:**job 13456097**,100题、10步,PENDING,结果未知。拿到`phase_metrics.jsonl`真实耗时后才能把R1-R4的正式预算定下来。

## 2026-10-26 组合实验(first-search+final-answer一起开)已提交

实验3(first-search)和实验4(final-answer)各自独立测都是正向、没有观察到副作用,所以加了一个新的`combined`模式(`scripts/run_agent_ablation.py`的`experiment_episode`现在检查`experiment in FIRST_SEARCH_EXPERIMENTS/FINAL_ANSWER_EXPERIMENTS`而不是精确字符串匹配,commit `bfc78fb`),只在3B上测两者叠加的效果(vs什么都不加的基线),**刻意没有同时测7B**,避免同时改组合+模型规模两个维度分不清归因。**job 13455678**,目录`~/nova-papers-exp-combined`,PENDING,结果未知。

## 2026-10-26 R10推理侧(BM25 vs 混合检索)已实现并提交

`backend/hybrid_retrieval.py`:`HybridQasperCorpus`继承`QasperCorpus`,只重写`search()`——BM25排名(跟基类同一套打分)+ bge-m3向量检索余弦相似度排名,用Reciprocal Rank Fusion(k=60)融合两者,`read()`/`find_paper()`/生产`QasperCorpus`类本身都没动。`scripts/run_hybrid_retrieval.py`复用`run_baseline.main()`不改代码,靠monkeypatch`backend.qasper_corpus.QasperCorpus`换成混合检索版本(跟`run_read_hint.py`/`run_agent_ablation.py`同一个模式)。embedding用CPU跑(避免跟vLLM抢a40显存),按需对实际被问到的论文段落算,不是预先跑全部1585篇论文。按照paperlens的发现(他们自己测过reranker增益很小),这次刻意只测混合检索,没加reranker,保持单变量。

写的时候抓到一个真bug:一开始把embedder的实例化放在模块级别(`_embedder = SentenceTransformerEmbedder()`写在模块顶部),导致**单纯import这个模块**就会去加载真实的bge-m3模型,我自己验证的时候直接卡住了——改成在`main()`里才构造,import本身应该瞬间完成。

**job 13456226**,目录`~/nova-papers-hybrid-retrieval`,PENDING,结果未知。跟`13455678`(组合实验)、`13456097`(吞吐探测)一起在CARC排队,互相独立不冲突。

## 2026-10-23 独立7B训练任务准备

用户要求保留已提交3B任务，另提交7B，不设置afterany依赖。新增 `scripts/carc_train_7b_smoke.sh`，A40单卡、48G系统内存、2小时；7B HTTP probe后做2步GRPO、G=2、limit=8。独立venv、端口、日志、checkpoint，不改3B入口。vLLM训练预算0.45为待验证起点，7B真实GPU训练容量尚未确认；失败保留完整日志，不自动缩短上下文以假装同条件成功。当前只是smoke，不是效果比较；正式3B/7B同预算pilot另行设置。脚本已提交：13349738；用户最后提供的状态为PENDING/Priority，真实训练结果待查。

## 2026-10-26 三个job全部有sacct结果了:一个成功、两个失败,下一个agent接手前先看这个

**13456097(吞吐探测)——COMPLETED,真实数据已经拿到,可以直接用来定R1-R4预算**:

`--limit 100 --max-steps 10 --num-generations 2`跑了10步,`phase_metrics.jsonl`里的真实耗时:
- `rollout_and_scoring`:10步总共115.7秒,平均每步**约11.6秒**(9.8~15.7秒区间波动)
- `parameter_update`:10步总共15.4秒,平均每步**约1.5秒**
- 单步总耗时平均**约13秒**
- 显存峰值:allocated~21.8GiB,reserved~22.6GiB,跟之前3B smoke test(21.3/22.1GiB)基本一致

**据此给出的具体建议(还没拍板,需要确认后再执行)**:R1-R4共用的正式参考训练预算建议定为`--limit 100 --max-steps 100`(完整过一遍100题,≈100步×13秒≈22分钟一次实验),CARC申请1小时时限够用。这个数字已经是实测的,不是估的,可以直接拿去用。

**13455678(组合实验:first-search+final-answer)——FAILED,00:03:47,但真实报错原因还没查到**:两次贴的日志都是pip安装依赖那一大段输出(被50000字符的消息长度上限截断了,根本没贴到日志结尾),看不到真正的失败点。**下一个agent接手后第一件事**:
```bash
tail -100 ~/nova-papers-exp-combined/ablation_13455678.log
```
只看结尾,不要整段贴,避免再被截断看不到关键信息。

**13456226(R10混合检索)——FAILED,只跑了1秒,sacct显示`batch`子步骤是`CANCELLED`、ExitCode `0:53`**:这么短的时间大概率是SLURM层面的问题(资源分配/权限/环境问题),不像是跑到脚本逻辑中间才出错——但日志内容至今一次都没有被贴出来过(两次都在贴到这条前被截断)。**下一个agent接手后**:
```bash
cat ~/nova-papers-hybrid-retrieval/hybrid-retrieval_13456226.log
```
这个文件应该不长(只跑了1秒),完整贴出来就行,不用怕截断。

**当前真实状态总结,供下一个agent快速定位**:
- ✅ GRPO训练本身(3B、7B)都已验证成功过,token probe验证过,这些不用再重复做
- ✅ 吞吐探测有了真实数据,R1-R4的训练预算有了具体建议(`--limit 100 --max-steps 100`),但还没跟用户最终确认拍板
- ❌ 组合实验(first-search+final-answer叠加)、R10混合检索推理对比——代码都写好推送了,**两次提交都失败了,真实失败原因还不知道**,不是代码本身一定有问题(日志根本没看全),这是下一个agent最优先要查清楚的事
- 实验1-4(禁止重复搜索、read提示、first-search、final-answer各自独立)这四个推理侧单变量实验全部已经跑完出了真实结果,记在`docs/EXPERIMENT_RECORD.md`里,不用重跑

## 2026-10-26 CARC个人账号磁盘配额超限:完整排查过程(已解决)

**症状最早出现在**:准备给R1-R4各建一个独立worktree时,`git worktree add`直接报`fatal: could not create directory of '.git/worktrees/nova-papers-r3-g4': Disk quota exceeded`,5个worktree一个都没建成,对应的5个训练job全部没能提交。

**排查过程中走过的弯路(记录下来避免下次重复踩)**:
1. **一开始怀疑两个之前失败的job(13455678组合实验、13456226混合检索)是OOM**——`sacct`查`MaxRSS`发现13455678只用了375MB内存,`ReqMem`是32G,远远没到OOM的量级,**这个假设被推翻了**。
2. **`df -h ~`显示文件系统整体还有91TB空闲(59%用量)**——跟"配额超了"的报错矛盾,说明超的不是整个文件系统容量,是CARC对这个账号单独设的per-user限制,`df`看不出这个。
3. **试了常见的per-user配额查询命令,这台机器上全都不可用**:`quota -s`没反应(命令存在但没输出,大概率这套存储没走标准quota机制),`lfs quota`报"lfs不可用"(确认存储不是Lustre),`seff`命令不存在。**目前为止没找到这台CARC机器上能查到个人配额具体数值的命令**——如果下次又遇到,直接提工单问CARC支持(`https://www.carc.usc.edu/user-support/submit-ticket`),不要继续猜命令浪费时间。
4. **`df -i ~`(inode用量)同样只反映整个共享文件系统**(898M/196B inodes,1%用量),对诊断个人配额没用,原理跟第2点一样。

**真正定位到根因的方法**:用`find <dir> -type f | wc -l`直接数每个worktree目录里的文件数(比`du -sh`快得多,`du`在网络文件系统上扫描大量小文件时非常慢,之前卡住过一次)。结果:
- `nova-papers-throughput`(完整装过一次torch+vllm+trl):**88,492个文件**
- `nova-papers-exp-combined`(pip装到一半因为配额超了failed掉的):13,236个文件
- `nova-papers-hybrid-retrieval`(几乎没开始跑就被SLURM取消):96个文件

**根因确认**:每次给一个新实验建独立worktree、重新`pip install -r requirements-training.txt`,一套完整的torch+vllm+trl装完是**8-10万个文件**。这个项目这几天陆陆续续建了七八个worktree(`exp2`/`exp3`/`exp4`/`throughput`/`lora-reload`/`grpo-7b`/`grpo-fixed`/`7b-baseline`……),每个都重复装一遍,累计很可能到了五六十万甚至更多文件,超过了CARC给这个账号设的per-user文件数量上限(具体数值未知,因为查不到)。**13455678这次失败,很可能就是pip在装到13,236个文件时,恰好撞上了配额上限,写不进新文件导致失败——跟OOM完全无关**;13456226跑1秒就被取消,大概率是同一时间配额已经满了,job连venv创建这一步都没走到。

**修复**:确认6个已经用完、结果已经提取/backup过的完整venv目录(`find <dir> -type d \( -name ".venv-grpo" -o -name "venv" \)`逐个确认路径后),全部`rm -rf`:
```
~/nova-papers-throughput/.venv-grpo
~/nova-papers-exp3/eval/first-search/13349421/venv
~/nova-papers-exp4/eval/final-answer/13349424/venv
~/nova-papers-grpo-fixed/.venv-grpo
~/nova-papers-7b-baseline/eval/baseline-7b/13318062/venv
~/nova-papers-exp2/eval/read-hint/13310260/venv
```
加上清掉scp传过去已经不需要的`~/*.bundle`文件。清完用`git worktree add`直接测试,成功建出新worktree,确认问题解决。

**以后怎么避免再犯**:不要再给每个实验单独建一个worktree+重装一遍环境。R1-R4这5个训练job改成**共用同一个worktree和同一个`.venv-grpo`**,依次(不是并行建多个worktree)提交,第一个job装好环境后后面4个直接复用,不会再新增几十万个文件。已经用完、结果已经记录/backup好的worktree,venv目录应该及时删掉,不要一直攒着。

## 2026-10-26 配额问题解决后,R1-R4全部提交 + R10混合检索重新提交

目录`~/nova-papers-rl-ablation`,5个job共用同一个worktree和`.venv-grpo`(按上面教训改的提交方式),全部PENDING,结果未知:

| Job | 内容 |
|---|---|
| `13467662` | 共享参考配置(ref,`--limit 100 --max-steps 100 --num-generations 2`,R1-R4的对照组共用这一次) |
| `13467663` | R1:`--zero-format-reward` |
| `13467664` | R2:`--zero-evidence-reward` |
| `13467665` | R3:`--num-generations 4` |
| `13467666` | R4:`--limit 1000` |

另外`13456226`(R10混合检索)重新提交成了`13467421`(目录`~/nova-papers-hybrid-retrieval`),配额问题解决之后提交的,理论上不该再复现同样的失败,结果未知。

**下一步**:查这6个job状态
```bash
sacct -j 13467421,13467662,13467663,13467664,13467665,13467666 --format=JobID,JobName%24,State,Elapsed,ExitCode
```

## 2026-10-27 R1-R4五个job的sacct都是FAILED,但训练本身其实全部成功——根因是`carc_train_smoke.sh`的路径bug(已修)

`sacct`显示`13467662`~`13467666`全部`State=FAILED`,但读实际日志(`grpo_<jobid>.log`)发现每一个都**完整跑完了既定的训练步数**(reward有正常波动、grad_norm非零、`train_runtime`/`train_loss`都正常打印),崩溃发生在训练循环**结束之后**:

```
Traceback (most recent call last):
  ...
  File "scripts/summarize_training_metrics.py", line 38, in summarize
    for line in Path(phases).read_text().splitlines():
FileNotFoundError: [Errno 2] No such file or directory: 'checkpoints/grpo-smoke-13467662/phase_metrics.jsonl'
```

**根因**:`carc_train_smoke.sh`里训练用的`--output-dir`走`TRAIN_ARGS`数组,默认是`checkpoints/grpo-smoke-${SLURM_JOB_ID}`,但允许被`EXTRA_TRAIN_ARGS`(`-- --output-dir ...`转发进来的参数)覆盖——argparse对重复flag取最后一次出现的值,这正是R1-R4提交时用的方式(分别指向`checkpoints/ref`/`r1-format`/`r2-evidence`/`r3-g4`/`r4-limit1000`,而不是job-id命名)。但脚本收尾调用`summarize_training_metrics.py`那一行(改之前的128行)还是写死`checkpoints/grpo-smoke-${SLURM_JOB_ID}`,没有跟着`--output-dir`的实际覆盖值走,导致找错路径、`FileNotFoundError`、整个job被SLURM记成FAILED——**训练和checkpoint保存都没有问题**,是收尾的统计脚本在找错地方。

用`find ~ -name phase_metrics.jsonl`确认了真实路径:
```
/home1/xianhanh/nova-papers-rl-ablation/checkpoints/ref/phase_metrics.jsonl
/home1/xianhanh/nova-papers-rl-ablation/checkpoints/r1-format/phase_metrics.jsonl
/home1/xianhanh/nova-papers-rl-ablation/checkpoints/r2-evidence/phase_metrics.jsonl
/home1/xianhanh/nova-papers-rl-ablation/checkpoints/r3-g4/phase_metrics.jsonl
/home1/xianhanh/nova-papers-rl-ablation/checkpoints/r4-limit1000/phase_metrics.jsonl
```
`checkpoints/ref/adapter_config.json`也确认存在——`13468133`(依赖`13467662`产出的格式检查job)能COMPLETED不是竞态误报,是checkpoint真的保存成功了,它读的就是真实数据。

**修复**(`scripts/carc_train_smoke.sh`):收尾调用前改成跟`TRAIN_ARGS`数组走一遍,找最后一次出现的`--output-dir`值(复现argparse"取最后一次"的语义),而不是无条件用job-id默认路径。已修改,**还没有在CARC上重新跑一次来验证这个修复本身**,下次任何job再用`-- --output-dir ...`覆盖默认路径时应该不会再误报FAILED了。

**不需要重新提交GPU训练来补数据**——训练已经成功,只是缺"phase-summary.json"(耗时/显存汇总),直接对已存在的`phase_metrics.jsonl`补跑一次纯CPU的`summarize_training_metrics.py`即可:

```bash
cd ~/nova-papers-rl-ablation
declare -A DIRS=([13467662]=checkpoints/ref [13467663]=checkpoints/r1-format
                 [13467664]=checkpoints/r2-evidence [13467665]=checkpoints/r3-g4
                 [13467666]=checkpoints/r4-limit1000)
for j in "${!DIRS[@]}"; do
  python -m scripts.summarize_training_metrics "${DIRS[$j]}/phase_metrics.jsonl" \
    --device-samples "eval/training/$j/device-memory.csv" --output "eval/training/$j/phase-summary.json"
done
```
已经跑过,结果见下表。

### R1-R4 + ref:训练确认成功 + 耗时/显存数据

| Job | 实验 | 训练步数 | 最终epoch | 训练中reward均值 | parameter_update耗时 | rollout_and_scoring耗时 | 总耗时 | rollout阶段整卡显存峰值 | update阶段整卡显存峰值 |
|---|---|---|---|---|---|---|---|---|---|
| `13467662` | ref(参考) | 100/100 | 1.0 | 0.588 | 156.2s | 1305.5s | 1461.7s | 22255 MiB(~21.7GB) | 7599 MiB(~7.4GB) |
| `13467663` | R1(`--zero-format-reward`) | 100/100 | 1.0 | 0.526 | 154.1s | 1215.8s | 1369.9s | 22333 MiB | 7761 MiB |
| `13467664` | R2(`--zero-evidence-reward`) | 100/100 | 1.0 | 0.506 | 156.5s | 1333.0s | 1489.5s | 22369 MiB | 7995 MiB |
| `13467665` | R3(`--num-generations 4`) | 100/100 | 1.0 | 0.598 | 370.5s | 2564.2s | **2934.7s** | 22037 MiB | 7767 MiB |
| `13467666` | R4(`--limit 1000`) | 100/100 | **0.1**(设计如此) | 0.543 | 149.8s | 1264.1s | 1413.9s | 22325 MiB | 7695 MiB |

**注意**:上面"训练中reward均值"是5个实验**各自不同的reward权重**算出来的,量纲不可比,不能拿来做消融对比结论(比如不能说"R1的0.526比ref的0.588低,所以去掉format奖励让训练变差"——两者权重配置不同,composite reward的理论取值范围本身就不一样)。真正要回答"去掉format/evidence奖励分别会让哪个指标掉下去",必须让训练后的checkpoint在同一套真实rollout评估(`score_real_rollouts.py`的Answer F1/Evidence F1/repeated_format_error)上跟ref对比,这一步**R1-R4还没做**,只给ref做过(见下一节)。

`13467666`(R4)唯一epoch没到1.0是设计使然:数据量从100题扩到1000题,但`--max-steps 100`没变(刻意固定更新/生成预算,只变数据池大小),所以100步只覆盖了约0.1个epoch,不是异常。

**耗时上两个值得记的点**:①group size从2→4(R3),耗时直接翻倍(1462s→2935s),但显存峰值几乎没变,说明这个配置下group size受时间瓶颈约束,不是显存瓶颈,以后扫更大group size要先算时间预算;②ref的1461.7s÷100步≈14.6秒/步,跟吞吐探测job(`13456097`)估的"~13秒/步"基本吻合,交叉验证通过。

### ref checkpoint的真实rollout检查(`13468133`):训练确实有效,但只验证了ref一组

把`scripts/carc_check_format_rate.sh`指向`checkpoints/ref`,用vLLM `--enable-lora`方式serve起来,跑`score_real_rollouts.py`,跟A的未训练基线(`eval/rollout_scoring/summary.md`)对比:

| 指标 | 未训练基线 | 训练后(ref) | 变化 |
|---|---|---|---|
| mean reward | 0.545 | 0.601 | +0.056 |
| mean Answer F1 | 0.147 | 0.177 | +0.030 |
| mean Evidence F1 | 0.093 | 0.110 | +0.017 |
| repeated_format_error | 33/100 | 25/100 | -8 |
| answered | 60/100 | 65/100 | +5 |
| zero-variance groups(无GRPO信号) | 1/20 | 0/20 | 改善 |
| unanswerable类型 Answer F1 | 0.200 | 0.540 | 大幅提升 |

全面正向,尤其"unanswerable"类型的Answer F1从0.200跳到0.540。但样本只有20题×G=5、单seed,按消融计划自己定的规矩(候选结论要≥3个seed复验),**这只是一个积极的初步信号,不能当最终结论**。

**下一步最该做的事**:把同一个`carc_check_format_rate.sh`分别指向`checkpoints/r1-format`/`r2-evidence`/`r3-g4`/`r4-limit1000`,才能真正回答R1-R4各自要验证的问题:

```bash
cd ~/nova-papers-rl-ablation
sbatch scripts/carc_check_format_rate.sh checkpoints/r1-format r1-format
sbatch scripts/carc_check_format_rate.sh checkpoints/r2-evidence r2-evidence
sbatch scripts/carc_check_format_rate.sh checkpoints/r3-g4 r3-g4
sbatch scripts/carc_check_format_rate.sh checkpoints/r4-limit1000 r4-limit1000
```

### R10混合检索(`13467421`)和组合实验(`13467754`):都真实COMPLETED,结果有意义

**R10混合检索**(BM25 vs BM25+向量+RRF,各40题):

| 指标 | BM25(control) | 混合检索(treatment) | 变化 |
|---|---|---|---|
| Evidence F1 | 0.160 | 0.214 | +0.054 |
| Answer F1 | — | 0.169 | — |
| 平均耗时/题 | 6.6秒 | 351.1秒 | 53倍 |
| answered | 26/40 | 25/40 | 基本持平 |
| repeated_format_error | 2/40 | 4/40 | 样本太小,不显著 |

检索质量确实提升,但延迟代价巨大(53倍),现阶段不适合直接接入demo,除非先优化向量检索那部分的耗时。

**组合实验**(first-search强制 + 轮数耗尽强制作答叠加,各40题):

| 指标 | control | treatment | 变化 |
|---|---|---|---|
| Answer F1 | 0.192 | 0.222 | +0.030 |
| Evidence F1 | 0.158 | 0.175 | +0.017 |
| 平均耗时/题 | 5.08秒 | 5.78秒 | +0.7秒 |

两个质量指标都提升,延迟代价几乎可忽略,是目前性价比最好的一项改动。`结束原因`里`forced_answer_invalid: 7`说明强制作答机制真的把原本会耗尽轮数、什么都打不出分的轨迹,转成了至少"有产出能打分"的轨迹,跟Search-R1那条参考发现(轮数用完强制再生成一轮,不直接判定失败)方向一致。

**状态更新(2026-10-27,用户确认)**:组合实验**已验证有效,列为`agent_loop.py`的候选生产设计**,但暂不动代码。原因两条:①仍只有单次40题结果,没到消融计划自己定的"≥3个seed复验"门槛;②`score_real_rollouts.py`(R1-R4真实rollout检查要用的脚本)直接调`backend.agent_loop.run_group`,现在改`agent_loop.py`默认行为会让R1-R4接下来的结果跟已经记录的ref基线不再是同一套评测条件,混入新变量。等R1-R4真实效果检查做完、组合实验本身多测≥2个seed确认稳定之后,再考虑合并进`agent_loop.py`。
R1-R4跑完后,每个实验组的`checkpoints/<name>/training_verification.json`/`phase_metrics.jsonl`要跟`checkpoints/ref/`的对应数据比较——**目前这些只验证"参数真的更新了",还没有验证训练后模型在eval上的表现变化**,这是R1-R4出结果之后的下一步工作,不是这一轮就能回答完的。
