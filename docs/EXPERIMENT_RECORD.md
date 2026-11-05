# CARC 实验记录

## 2026-10-22 收到的真实结果

来源：用户提供的 CARC 作业日志与 summary，代码 1c5b36d，原版 Qwen2.5-3B-Instruct。

### 训练 smoke 13291602

FAILED，8分9秒。依赖安装、A40 CUDA 运算成功，HTTP 服务通过就绪检查。check_vllm_rollout 在发送请求前发生 `TypeError: Object of type BatchEncoding is not JSON serializable`。安装的 transformers 为 5.19.0。未进入 GRPO 参数更新，不能声称训练成功。

修复：probe 显式使用 return_dict=False，同时兼容 Mapping/input_ids，校验单条非负整数 token 列表；训练初始提示及工具后缀同样校验。原训练循环使用 encode，不是 tokenize=True 的 apply_chat_template，没有证据证明其发生了同一异常。修复仍需真实 GPU 复验。

### 实验1：禁止重复搜索提示，job 13292586

COMPLETED，12分11秒；对照和实验各40题，均0接口异常。bf16/eager，同一服务器，temperature=0、workers=1。对照仅移除完整的禁止重复搜索指导句；实验保留。

|指标|对照|实验|
|---|---:|---:|
|Answer F1|0.192|0.118|
|Evidence F1|0.158|0.150|
|成功回答|27/40|23/40|
|max_rounds|9/40|16/40|
|格式错误终止|4/40|1/40|
|至少 read 一次|4/40|6/40|
|平均轮数|3.05|3.60|
|平均秒/题|4.6|4.9|

结论仅限本次40题：总体答案质量下降，未支持采用该提示。不能从汇总判断重复搜索率变化，也未做统计显著性检验。

原始结果：~/nova-papers-exp1/eval/no-repeat/13292586/{control,treatment}/。

### 实验2：截断段落提示 read（待 GPU 执行）

独立 runner，仅实际截断的 search 段落后追加英文提示与对应 read 动作。read 全文及未截断段落不改变。两组系统提示完全相同，均移除禁止重复搜索指导句；不改排序、长度、轮数、强制作答。对照重跑，不直接用不同执行环境下的实验1数字代替对照。

scripts/carc_eval_read_hint.sh 使用独立作业内 venv、端口及 eval/read-hint/<jobid> 输出，两组顺序执行；每组必须40题且没有接口异常才成功退出。不修改生产 QasperCorpus 或训练路径。尚未提交/获得真实结果。

## 2026-10-23 更新：实验2、7B及真实两步训练

### GRPO job 13310264：训练及保存完成，后处理失败

真实 token probe 通过；training_verification.json: global_steps=2，changed_trainable_tensors=504。保存 adapter_model.safetensors（约58MB）、adapter_config.json、tokenizer与chat template、training_args.bin及phase_metrics.jsonl。尚未做checkpoint重新加载验证，也没有证明训练后效果提升。

作业 FAILED 的最终 Python 异常来自显存汇总：device-memory.csv 内容仅为 `No devices were found`，无法解包成timestamp和memory两列。不得把缺失的整卡显存补成0，也无需因此重跑已完成的训练。

|阶段|总时间/秒|PyTorch allocated峰值/GiB|PyTorch reserved峰值/GiB|
|---|---:|---:|---:|
|rollout_and_scoring|27.2451|20.9034|21.9355|
|parameter_update|3.3154|21.3371|22.1230|

同一step的parameter_update有前后两个片段，表中累加全部片段；峰值取各片段最大值，不累加。这里只是训练进程PyTorch allocator统计，不是整卡总显存，不能据此承诺7B显存足够。

待解决警告：TRL 1.16.0报告支持vLLM 0.22–0.31，而安装0.32.0；日志还出现cumem_allocator.cpp:249 CUDA invalid argument。现有证据不能证明该CUDA错误的具体来源或权重重载无问题。此轮只修复显存监控/汇总，不盲目更换已跑通两步的依赖；兼容版本需单独验证。

监控修复：将torch逻辑GPU0的UUID规范化，与nvidia-smi实际UUID清单比对，不使用主机index0兜底；预采样确认CSV格式后才启动监控。异常输出独立保存。汇总统计坏行并标注unavailable/partial，保留有效阶段指标和null整卡峰值。实机UUID原始值尚未拿到，因此不能断言原错误就是特定字符串格式造成；新采集实现仍待GPU复验。

### 实验2 job 13310260（COMPLETED，13分17秒）

|指标|3B对照|3B截断提示read|
|---|---:|---:|
|Answer F1|0.192|0.163|
|Evidence F1|0.158|0.233|
|Evidence F1 seen|0.158|0.207|
|回答数|27/40|30/40|
|max_rounds|9/40|9/40|
|格式错误终止|4/40|1/40|
|至少read一次|4/40|6/40|
|平均轮数|3.05|3.05|
|平均秒/题|4.5|4.3|

不是全面超过：证据质量、回答率改善，但答案F1下降0.029。分题型：extractive 0.126→0.150，free_form 0.073→0.121，yes_no 0.308→0.106，unanswerable 0.400→0.400。主要退步集中在yes/no；需逐题轨迹才能解释原因。

### 7B基线 job 13318062（COMPLETED，16分20秒）

|模式|Answer F1|Evidence F1|平均秒/题|
|---|---:|---:|---:|
|direct|0.175|0.125|1.7|
|rag|0.349|0.196|2.3|
|agent|0.241|0.143|3.1|

7B Agent: 回答38/40，max_rounds=2，平均轮数2.77，至少read 12/40，Evidence F1 seen=0.227。

与上述3B对照Agent相比：答案F1 0.192→0.241、回答率27→38、read次数4→12改善；Evidence F1 0.158→0.143下降；unanswerable F1 0.400→0.036下降。因此7B也不是全面超越。不同作业的延迟不能简单当作纯模型速度对比。不能用这组Agent结果推断所有模式全面胜出，7B Agent仍低于7B RAG的Answer F1。都是40题有限样本，未做显著性检验。

### 保存路径

CARC checkpoint: ~/nova-papers-grpo-fixed/checkpoints/grpo-smoke-13310264/。
原始训练日志: ~/nova-papers-grpo-fixed/grpo_13310264.log。
其他训练产物: ~/nova-papers-grpo-fixed/eval/training/13310264/。
本地尚未下载权重；需要用户在CARC打包，然后在Mac通过scp下载并核对SHA256。不得把“服务器已保存”写成“本机已备份”。

## 实验3和4实现（尚待提交CARC）

共用独立runner scripts/run_agent_ablation.py 与 scripts/carc_eval_agent_ablation.sh。
每个实验作业各自重新跑40题control/treatment，同一3B模型、系统提示词和参数，两组都删除禁止重复搜索句，不加入read提示。

实验3 first-search：仅首次实际search调用使用原问题；保留模型输出，另记model_query与executed_query。后续search正常，不额外免费搜索，不基于标准答案判断“有用信息”。未发出search则不干预。

实验4 final-answer：仅max_rounds终止后重建完整已有对话，包含第5轮工具结果，追加一次要求answer的生成；新增回合不执行任何工具，模型仍输出search/read或格式错误则记forced_answer_invalid。不干预已回答或连续格式错误提前终止。最多6次生成，不保证模型服从提示。

两组都记录服务器实际prompt/completion token用量（不是字符估计）；额外记录第6轮耗时和token、额外作答次数。comparison.json汇总成本与得分；第4项因增加计算预算，效果须结合成本解释。生产agent_loop及GRPO保持不变。


## 2026-10-23 提交与备份状态同步

- 实验3 first-search 已提交 **13349421**，~/nova-papers-exp3，433935c。结果未知，不能再写“未提交”。
- 实验4 final-answer 已提交 **13349424**，~/nova-papers-exp4，433935c。结果未知。
- LoRA备份已下载Mac：~/Desktop/nova-papers-results/nova-grpo-backup-13310264-hLHFaH/，用户确认SHA256 OK。旧phase-summary重新生成，包内原始phase_metrics保留；汇总是在打包后生成，需独立下载。
- 新重载/监控验证脚本已准备；使用旧训练venv只读、离线基础模型缓存，不更新checkpoint、不训练。检查适配器张量和有限logits、两条固定提示下开启/关闭adapter的生成与logit差异，严格要求采样窗口内有效显存数据。
- adapter配置revision未固定，这限制严格复现；实际缓存snapshot写入新报告。验证不运行vLLM，不能声称解决cumem_allocator错误。待GPU运行，不是已完成质量评估。

## 2026-10-25 实验3、4真实结果（sacct确认COMPLETED，0:0）

来源：用户CARC终端`cat`出的`comparison.json`，40题，同一对照组（两实验共用同一组control数字，answer_f1=0.19244、evidence_f1=0.15833——确认两次独立作业的control一致，不是复制粘贴错误）。

### 实验3 first-search（job 13349421，~/nova-papers-exp3）

|指标|对照|实验|
|---|---:|---:|
|Answer F1|0.1924|0.2094|
|Evidence F1|0.1583|0.1750|
|prompt tokens（均值）|2878.3|3471.5|
|completion tokens（均值）|232.7|241.3|
|latency_s（均值）|4.57|4.72|

两项指标都小幅正向，没有额外作答回合（extra_answer_attempts=0，符合设计：这项实验不触发第6轮）。prompt token涨了约20%，推测是原始问题检索出的段落跟模型自己改写后的查询检索结果不同导致上下文变长。样本量仍是40题，没做显著性检验，不能认定是稳定的提升，但方向一致、没有观察到副作用。

### 实验4 final-answer（job 13349424，~/nova-papers-exp4）——目前观察到的最大正收益

|指标|对照|实验|
|---|---:|---:|
|Answer F1|0.1924|0.2319|
|Evidence F1|0.1583|0.2167|
|额外作答次数|0|9/40|
|额外token（总计）|0|424|
|额外耗时（总计，秒）|0|8.49|

只对用满5轮没答出来的9道题生效（跟对照组的9道max_rounds题对应），强制追加一轮禁用工具的生成后至少部分题目拿到了可计分的答案。Answer F1相对提升约21%，Evidence F1约37%，额外成本很小（9题总共8.5秒、424个token）。这是四项单变量实验（禁止重复搜索、read提示、first-search、final-answer）里目前唯一一个两个指标都明显改善、且没有观察到任何子类别回归的结果——但仍然只是40题单样本，需要更大样本/多seed才能确认是稳定效应，不能据此直接改生产`agent_loop.py`。

### GPU监控修复（2c5fdb0）在LoRA重载作业（13349479）上确认生效

`eval/lora-reload/13349479/device-memory.csv`这次是真实的显存采样曲线（3 MiB线性涨到6605 MiB，约200ms间隔），不再是13310264那次的`No devices were found`——UUID精确匹配的修复在真实GPU上验证通过了。adapter本身的重载验证报告（张量对比、开关adapter的生成/logit差异）内容尚待补充，不能视为已完成。

## 2026-10-25 7B GRPO两步训练确认成功（job 13349738，~/nova-papers-grpo-7b）

`training_verification.json`：`global_steps=2，changed_trainable_tensors=392`——跟3B（504个张量，job 13310264）一样，参数真的更新了，不是服务起来就算数。`live-token-probe.json`也是真实vLLM响应（`<answer>Yes</answer>`对应真实token id/logprob）。

资源对比（同样是G=2、limit=8的smoke规模，不是完整训练规模）：

|指标|3B（13310264）|7B（13349738）|
|---|---:|---:|
|PyTorch allocated峰值|~21.3 GiB|~36.4 GiB|
|PyTorch reserved峰值|~22.1 GiB|~36.7 GiB|
|rollout_and_scoring总耗时|27.2秒|25.7秒|
|parameter_update总耗时|3.3秒|3.8秒|

a40是48GiB显存，7B这次峰值36.7GiB，有约11GiB余量，没有OOM；两步耗时跟3B同量级，不是明显更慢。**这解除了此前"3B还是7B"讨论里唯一真正不确定的变量（7B训练资源是否可行）**——G=2/limit=8规模下7B训练是可行的。但这只是最小规模smoke test，不代表G=4/8等更大group size、更长训练步数下显存和耗时会如何扩展，也不能据此直接决定把7B定为正式训练基座，需要R7消融（见TRAINING_NOTES.md）按计划单独验证。

## 2026-10-26 两个新任务已提交，均PENDING，无结果

- **组合实验（first-search+final-answer一起开）**：job `13455678`，目录`~/nova-papers-exp-combined`，对照3B什么都不加的基线。实验3、实验4各自独立都是正向结果，这次测两者叠加会不会加成、相互抵消还是出现意外交互——没有同时测7B，保持单变量。
- **吞吐探测（为冻结R1-R4的正式训练预算做准备）**：job `13456097`，100题、10步（`--limit 100 --max-steps 10`），不是正式训练，是为了拿到真实的`phase_metrics.jsonl`耗时数据，用来决定R1-R4该用多大的训练步数/样本量预算。

同时：R1（format奖励消融）、R2（evidence奖励消融）的代码blocker已解决，`train_grpo.py`加了`--zero-format-reward`/`--zero-evidence-reward`（commit `76acb0b`），顺手修了一个真bug——`_reward_func`里的`weights`变量之前从没真正传进`composite_reward(...)`，主体打分一直用默认权重，这两个消融flag不修的话会悄悄什么都不做。R3（group size）、R4（数据量，训练集实际2593题）本来就有现成CLI参数，没有代码blocker。R1-R4现在唯一剩的blocker是"吞吐探测"这一步（见上面job 13456097），结果出来才能真正开始跑消融。

## 2026-10-26 三个job的sacct结果：吞吐探测成功，另外两个失败（原因待查）

| Job | 内容 | 结果 |
|---|---|---|
| `13456097` | 吞吐探测（`--limit 100 --max-steps 10`） | ✅ COMPLETED，真实耗时数据已拿到 |
| `13455678` | 组合实验（first-search+final-answer） | ❌ FAILED，00:03:47，真实报错原因未知（日志被截断，只看到pip安装输出） |
| `13456226` | R10混合检索推理对比 | ❌ FAILED，仅00:00:01，ExitCode 0:53（像是SLURM层面问题，不像脚本逻辑中途出错），日志内容从未被完整贴出过 |

**吞吐探测的真实数字**：`rollout_and_scoring`平均每步~11.6秒，`parameter_update`平均每步~1.5秒，单步合计~13秒，显存峰值allocated~21.8GiB/reserved~22.6GiB（跟3B之前的smoke test一致）。据此建议R1-R4共用参考训练预算`--limit 100 --max-steps 100`（≈22分钟一次实验），**这是具体建议，还没跟用户最终确认拍板**。

**两个失败的job都还没查到真实原因**——不是已经确认是代码bug，只是两次贴日志都被消息长度上限截断在看到真正报错之前。下一步明确是：
```bash
tail -100 ~/nova-papers-exp-combined/ablation_13455678.log
cat ~/nova-papers-hybrid-retrieval/hybrid-retrieval_13456226.log
```
拿到这两个的真实内容后才能判断是CARC环境问题还是`run_agent_ablation.py`/`hybrid_retrieval.py`代码本身的问题。

**后续更新**：两个失败最终确认是CARC个人账号磁盘配额超限导致的（根因排查过程见`TRAINING_NOTES.md`），不是代码bug。配额问题解决后，组合实验重新提交为`13467754`，R10混合检索重新提交为`13467421`，R1-R4五个训练消融job（`13467662`~`13467666`）也已提交，全部结果待查。

## 2026-10-25 A的真实rollout reward验证（已用委托数据独立核实）

队友A提交了`scripts/score_real_rollouts.py`+`eval/rollout_scoring/`（commit `9c8e003`）：用**未训练的**Qwen2.5-3B-Instruct，20道train-split问题×G=5条轨迹（100条），采样设置跟训练一致（temperature=1.0），直接调`train_grpo._reward_func`打分——这是在花CARC时间训练之前，先确认reward函数在真实模型输出上表现正常。

**已提交的`summary.md`核心数字**（本会话重新跑`rollouts.jsonl`独立核实过，数字一致）：
- 100条轨迹，mean reward 0.545，mean Answer F1 0.147，mean Evidence F1 0.093
- 结束原因：answered 60条（mean reward 0.734）、**repeated_format_error 33条（mean reward 0.267）**、max_rounds 7条（mean reward 0.229）
- Sanity check：答对的（Answer F1≥0.5）mean reward 1.751，答错的0.506，没答出来的0.260——分层清晰，印证了之前修的`_extract_answer`那个bug（格式错误的轨迹不会再被误判成答对）在真实数据上确实生效
- 20组题里只有1组5条轨迹reward零方差（GRPO对19/20的题都有可用训练信号）

**A在微信里额外分享的细分数字**（独立重新计算`rollouts.jsonl`核实，数字基本吻合，小的出入是判断方法不同导致的，不代表数字有误）：
- 共106次`format_error`动作；其中105次（A说100次）是**完全没有任何标签**，不是标签写错，是模型直接裸写一段话
- 其中69次（A说"大多"）紧跟在`<search>`动作之后——模型搜完信息，下一轮该正式作答时却裸写
- 用严格子串匹配，39次（A说"约一半"，口径更宽松）裸写的内容里其实包含标准答案——说明不是答不出来，是不守格式
- 71次有后续轮次的机会里，38次（跟A的数字完全一致）模型在收到格式错误提示后能自己改正

**结论**：reward.py设计本身没问题，但未训练的基座模型有约三分之一的轨迹因为不遵守`<search>/<read>/<answer>`标签协议而被直接终止，其中很大一部分其实已经知道正确答案。这正是R1实验（format奖励消融，job `13467663`）要验证的——格式奖励信号能不能、以及需要多久才能把这个问题训练掉。在R1出结果之前，**不要**去改格式错误的提示词措辞（A提出过这个想法），否则会把"奖励信号的效果"和"提示词改动的效果"这两个变量混在一起，分不清谁的功劳。

## 2026-10-26 新增：训练后checkpoint的格式错误率检查工具

`scripts/carc_check_format_rate.sh`（commit `ef5d5bc`）：用vLLM的`--enable-lora --lora-modules`方式把训练产出的LoRA adapter serve起来（不需要先合并权重），然后把A上面那个`score_real_rollouts.py`原样指过去（只改`--model`参数为adapter注册名），复用同一套统计口径，直接对比训练前后的格式错误率，不用等A另外跑Answer F1评估。**还没在真实GPU上验证过跑得通**，job `13468133`（对`checkpoints/ref`，即参考配置训练job `13467662`的产出）是第一次验证，依赖`13467662`先跑完，两者之间没设置SLURM依赖，有一次性的竞态风险（先跑到会因为checkpoint还不存在而失败，重新提交一次即可）。

## 2026-10-27 R1-R4 + ref五个训练job全部完成；R10混合检索、组合实验、ref格式检查三个评估job也全部完成（已核实日志，非只看sacct状态）

**先修的一个假阳性**：`sacct`把`13467662`~`13467666`（ref+R1-R4）全部标成`FAILED`，但读实际训练日志发现5个都**完整跑完了既定步数**，崩溃发生在训练结束*之后*的`summarize_training_metrics.py`收尾步骤——根因是`scripts/carc_train_smoke.sh`收尾那一行硬编码了`checkpoints/grpo-smoke-${SLURM_JOB_ID}`路径，没有跟着这5个job实际用的`--output-dir`覆盖值（`checkpoints/ref`/`r1-format`/`r2-evidence`/`r3-g4`/`r4-limit1000`）走，找错路径导致`FileNotFoundError`。**训练和checkpoint保存都没问题**，已修好这个脚本（改成跟`TRAIN_ARGS`里最后一次出现的`--output-dir`走），还没在CARC上重新跑验证这个修复本身。详细诊断过程、修复后的耗时/显存结果表见`TRAINING_NOTES.md`"2026-10-27"一节。

**ref checkpoint真实rollout检查结果**（`13468133`，对比A的未训练基线`eval/rollout_scoring/summary.md`）：

| 指标 | 未训练基线 | 训练后(ref) | 变化 |
|---|---|---|---|
| mean reward | 0.545 | 0.601 | +0.056 |
| mean Answer F1 | 0.147 | 0.177 | +0.030 |
| mean Evidence F1 | 0.093 | 0.110 | +0.017 |
| repeated_format_error | 33/100 | 25/100 | -8 |
| unanswerable类型Answer F1 | 0.200 | 0.540 | 大幅提升 |

训练确实有效（全面正向，不只是格式错误率），但只有20题×G=5单seed，按消融计划自己的规矩，这还只是积极的初步信号，不是能当真的结论。**R1-R4四组消融还没做这一步真实rollout检查**——训练日志里的reward均值因为权重不同不可比，不能拿来回答"去掉format/evidence奖励分别会让哪个指标掉下去"这个消融本来要问的问题，下一步要把同一个检查脚本分别指向`checkpoints/r1-format`/`r2-evidence`/`r3-g4`/`r4-limit1000`。

**R10混合检索**（`13467421`，BM25 vs BM25+向量+RRF，各40题）：Evidence F1从0.160提升到0.214，但平均耗时从6.6秒/题暴涨到351.1秒/题（53倍），现阶段不适合直接接入demo。

**组合实验**（`13467754`，first-search强制+轮数耗尽强制作答叠加，各40题）：Answer F1从0.192提升到0.222，Evidence F1从0.158提升到0.175，平均耗时只多0.7秒/题——目前性价比最好的一项改动，**已验证有效，列为生产`agent_loop.py`的候选设计**，但暂不改代码：一是仍只有单次40题结果，不满足消融计划自己定的"≥3个seed复验"门槛；二是`score_real_rollouts.py`（R1-R4接下来要用来做真实rollout检查的脚本）直接调用`backend.agent_loop.run_group`，现在改`agent_loop.py`默认行为会让R1-R4接下来测出的结果跟已经记录的ref基线（25/100格式错误率等）不再是同一套评测条件，混入"agent_loop行为变了"这个新变量。**决定（2026-10-27，用户确认）**：先只记录为候选设计，等R1-R4真实效果检查做完、且组合实验本身多测≥2个seed确认稳定之后，再考虑合并进`agent_loop.py`。

**完整答题质量评估（Answer F1/Evidence F1）尚未打通**：按最新理清的分工，这部分更适合A用她现成的`run_baseline.py`指向训练后的checkpoint来测，不是B重新发明一套评估脚本；B的职责到"让checkpoint能被serve起来给A用"为止。

## 2026-10-27（续）两处方法论问题修好并提交新job；组合实验的统计显著性缺口量化出来了

**R1-R4的真实rollout检查补了一组held-out版本**：`score_real_rollouts.py`本来就有`--split`参数（默认`train`，跟GRPO训练同一个池子），`carc_check_format_rate.sh`加了`[split]`可选参数后，新提交了5个job在validation split（1005题，训练/选checkpoint全程没碰过）上复测ref+R1-R4：`13500255`(ref)/`13500256`(r1-format)/`13500257`(r2-evidence)/`13500258`(r3-g4)/`13500259`(r4-limit1000)，结果未知。跟原本train-split的`13468133`+`13499835`~`13499838`合起来，后续能同时看train-split和held-out两套数字是否一致。

**组合实验换成真随机采样复验**：`carc_eval_agent_ablation.sh`里`--temperature 0`是硬编码的贪婪解码，对固定40题换seed重跑结果不会变——这次加了`[temperature] [seed]`可选参数，提交了两个`temperature=1.0`（匹配GRPOConfig训练时的采样条件）的复验job：`13500267`(seed=601)、`13500268`(seed=602)，结果未知。

**组合实验的统计显著性缺口，用真实数据算出来了，不是估的**：拿`13467754`那组40题真实配对数据（按`question_id`对齐算Answer F1差值）算出`mean_diff=0.0297`、`std_diff=0.2209`，代入标准配对检验的样本量公式（95%置信度、80%把握），结果**需要约436题**才能把这个+0.03的提升跟噪声分开——40题这个样本量下，效应量比统计噪声还小，不能当结论用。

**为此新增的基建（已在本地测过，还没上CARC跑）**：
- `scripts/make_held_out_sample.py`：从QASPER test split（1451题，训练和checkpoint选择全程没碰过）按`score_real_rollouts.usable()`同一套筛选规则抽样，本地跑出480题（`free_form`43+`unanswerable`79+`yes_no`108全部可用数量，`extractive`从487个里抽250个凑够480），高于436的门槛。
- `carc_eval_agent_ablation.sh`加了`[samples-file]`和`[workers]`两个可选参数（默认值不变，向后兼容），去掉了硬编码的"必须是40题"断言。
- 下一步：把这两处脚本改动同步到CARC worktree，跑`python -m scripts.make_held_out_sample`生成480题文件，再提交`sbatch scripts/carc_eval_agent_ablation.sh combined 0 42 eval/sample_questions_test500.jsonl 8`（跟`13467754`保持`temperature=0`/`seed=42`一致，只变样本量这一个变量）。

## 2026-10-28 R1-R4真实rollout检查结果到手（train split）；validation split那一批第一次提交就踩了坑

**R1-R4 + ref的train-split真实rollout检查结果**（20题×G=5，temperature=1.0，seed=599，100 rollouts，job `13499835`~`13499838`+`13500255`）：

| | mean reward | Answer F1 | Evidence F1 | mean rounds | answered | repeated_format_error | max_rounds |
|---|---|---|---|---|---|---|---|
| ref | 0.604 | 0.197 | 0.088 | 3.06 | 59 | 31 | 10 |
| r1-format（format权重=0） | 0.593 | 0.169 | 0.142 | 3.00 | 57 | 34 | 9 |
| r2-evidence（evidence权重=0） | 0.628 | 0.202 | 0.095 | 3.14 | 69 | 25 | 6 |
| r3-g4（group size 2→4） | **0.715** | **0.237** | 0.133 | 2.91 | 66 | 29 | 5 |
| r4-limit1000（训练题100→1000，步数不变） | 0.571 | 0.170 | 0.116 | 3.11 | 53 | 32 | **15** |

（这组数字跟上面`13468133`测的ref基线`0.601`/`0.177`/`0.110`不完全一样，是因为`13468133`测的是train split前20题的旧抽样口径，这次`13500255`是同一套`score_real_rollouts.py --split train`重新跑的，数字上下浮动在这个样本量下是正常噪声，不代表ref本身变了。）

**初步解读（仅train split，模型训练时见过这些题，不是最终结论）**：
- **r3-g4全面领先**：reward/Answer F1/Evidence F1三项都明显高于ref，max_rounds超时最少——group size从2加到4给出更稳定的组内相对优势估计，目前唯一一个"全指标一致变好"的改动，优先级最高。
- **r4-limit1000全面落后**，尤其max_rounds从10暴涨到15——支持"训练题变多但步数不变会稀释每题学习量"的解释，不支持"单纯扩数据池"。
- **r2-evidence反直觉没有变差，部分指标反而更好**（format_error全场最低，answered最多）——不支持"去掉证据奖励会让模型靠瞎蒙答对但引错证据"这个假设，但可能是train-split上的过拟合假象，必须等validation split确认。
- **r1-format是混合信号**：Answer F1降但format_error反而也升、Evidence F1却涨——不是干净的"去掉约束就变差"的故事，暂不下结论。

**validation split那一批的提交出了问题**：`13500255`(ref)和`13500256`(r1-format)虽然在`sacct`里显示`COMPLETED exit 0:0`，但实际生成的文件是`eval/rollout_scoring-ref/summary.md`和`eval/rollout_scoring-r1-format/summary.md`（没有`-validation`后缀），说明提交时第3个参数（split）没有真正传上，等于又跑了一遍train split，重复了`13500255`本应验证的东西，**没有产生任何validation split的数据**。已撤销尚未开始跑的`13500257`/`13500258`/`13500259`，需要重新提交全部5个，且要先确认`cd`到了正确的工作目录（`~/nova-papers-rl-ablation`）再`sbatch`——之前在`~`下直接`sbatch`会报`Unable to open file`。**教训**：`sacct`的`COMPLETED exit 0:0`只能说明脚本没有崩溃，不能说明测的是预期的那个split/参数组合；提交后必须用`grep "^Done. Compare" format-check_<jobid>.log`读脚本自己打印的真实输出路径来确认。

**结论（暂定，等validation split确认后定案）**：r3-g4和r4-limit1000的效应量足够大且方向一致，大概率是真信号；r1-format和r2-evidence的效应较小或混合，必须等validation split结果排除train-split过拟合的可能才能用来决定reward权重。

## 2026-10-28（续）四条实验线同时在CARC排队；新增R5推理侧top-k消融代码

截至此时四个job都已正确提交、排队中、结果未知：

- `13534449`~`13534453`：ref+R1-R4真实rollout检查，validation split（这次参数真的带上了，不是上面那次误跑train split的重复）。
- `13500267`、`13500268`：组合实验temperature=1.0真随机采样复验（seed 601/602）。
- `13534856`：组合实验480题统计显著性确认（test split）。第一次提交（`13534505`）是在CARC worktree还处于detached HEAD、`git pull`失败、新脚本和480题样本文件都不存在的情况下交的，已`scancel`；用`git checkout origin/main`正确同步后重新生成样本文件、重新提交，这次有效。
- `13535156`：新增的R5推理侧top-k消融，第一次提交。

**R5（检索top-k消融）新增代码（commit `8bbf323`）**：`backend/qasper_corpus.py`的`QasperCorpus`、`backend/hybrid_retrieval.py`的`HybridQasperCorpus`都新增了`search_limit`构造参数（默认3，向后兼容），`search()`不传`limit`时改为跟随实例的`search_limit`，这样`agent_loop.py`里`corpus.search(paper_id, action.content)`这种不显式传`limit`的调用会自动跟着corpus配置走，不用改`agent_loop.py`本身。`scripts/run_baseline.py`加了`--search-limit`CLI参数并记录进`summary.md`。新脚本`scripts/carc_eval_search_limit.sh`复用`carc_check_format_rate.sh`的vLLM+LoRA adapter起服务方式，对同一个checkpoint分别跑`search-limit=1/3/5`三次`run_baseline.py --modes agent`，汇总成`comparison.json`。

**这是R5原计划的"推理侧简化版"，不是训练侧版本**：原计划是"训练侧改变rollout检索数量，固定推理top-k=3隔离变量"，需要对每个k值重新跑一次完整GRPO训练，成本高；现在这版只是在已经训练好的`ref` checkpoint上切换推理时的top-k，成本低很多，但只能回答"已训练模型对检索宽度敏不敏感"，回答不了"训练时该用多大top-k"这个更贵的问题。如果这版结果有明显信号，再决定要不要做训练侧版本。

## 2026-10-29 新发现一类坑：CARC worktree本地脚本过期不会被`git status`查出来，导致两批"已完成"的job其实都无效

**现象**：`13534449`~`13534453`（R1-R4 validation split第二次提交）和`13500267`/`13500268`（组合实验temperature=1.0复验）都在`sacct`里显示`COMPLETED exit 0:0`，但结果都不对：

- `13534449`（ref的validation split检查）的log打印`Done. Compare eval/rollout_scoring-ref/summary.md`——没有`-validation`后缀，说明实际跑的还是train split。
- `13500267`/`13500268`（本该是temperature=1.0的复验）的`comparison.json`里`control`/`treatment`的`answer_f1`/`evidence_f1`跟最早的`13467754`（temperature=0，seed=42）精确到小数点后14位完全相同——不可能是真的temperature=1.0跑出来的（那样会有随机性）。

**根因**：这两批job分别是从`~/nova-papers-rl-ablation`和`~/nova-papers-exp-combined`提交的，提交当时这两个worktree本地的`scripts/carc_check_format_rate.sh`/`scripts/carc_eval_agent_ablation.sh`都是本地旧版本（前者相对这个worktree当时的HEAD`f0bafc4`——比最早写这个脚本的commit`ef5d5bc`还早——是"新增"状态，说明从来没同步过任何更新；后者是没有`[temperature][seed]`参数支持的旧版本）。旧脚本不认新加的命令行参数，传了也被忽略，所以等于又重跑了一次旧配置。脚本自己的`git status --porcelain --untracked-files=no`检查只能查出"相对这个worktree自己的本地HEAD有没有未commit的改动"，查不出"这个worktree的HEAD本身是不是落后于origin/main"——这就是为什么两批job的sacct都显示干净的`COMPLETED exit 0:0`，检查形同虚设。

**修复**：用`git checkout origin/main`把两个worktree都同步到最新commit（`nova-papers-rl-ablation`→`8bbf323`，`nova-papers-exp-combined`→`eceb406`），重新提交全部7个job（5个validation split + 2个temperature=1.0复验）。

**新教训，写进流程**：以后在任何CARC worktree里提交job之前，不管看起来要不要，都先`git checkout origin/main`一次，不能假设"之前同步过就一直是最新的"。这是本项目第三次因为"脚本/参数没真正生效"而白跑job（第一次是忘传`validation`参数本身，第二次是`~`目录下sbatch找不到文件，这次是worktree过期），说明`sacct`的`COMPLETED exit 0:0`永远不能单独作为"这次测的是我以为的配置"的证据，必须配合读脚本自己打印的实际路径/配置来确认。

## 2026-10-29 团队沟通：A独立做的提示词对照测试——初步记为"跟R1互相印证"，重新核对后发现这个连接站不住，已更正

**来源**：队友A（梁熙悦）在团队沟通中口头汇报，不是走本项目追踪的CARC job（没有job ID，没有原始数字），记录的是她转述的结论，严谨度低于本文档其他有job ID支撑的条目，仅作为参考，不能单独当结论用。

**A的测试**：跑了一次"改提示词前后对比"的冒烟测试流程（改的是提示词措辞/格式相关的部分，不是本项目的reward权重）。她汇报的结果：
- RAG基本符合预期，没有意外。
- 之前猜测"改提示词格式"可能会提分，但实测下来：**因格式错误被终止的次数确实减少了，但最终答案质量（answer）跟改之前差不多，没有明显提升**。
- A的结论：换提示词不一定能直接提分，可能得找别的方向；如果真要换提示词，工作量会比较大，决定先搁置，以后再看。

**最初的错误连接（已更正）**：第一版这里写的是"A的结果跟R1(format reward=0)互相印证，格式和答案质量没有强耦合，不支持投入精力"。用户追问后重新核对发现这个连接是错的：R1的实际数据是reward、Answer F1都比ref差（Answer F1降14%），format_error也确实略微往未训练基线方向回弹（31→34）——按`TRAINING_NOTES.md`"R1-R4结果解读表"自己定的标准，这组数据其实更接近"**格式奖励必要，应保留**"这一边，不是"格式奖励多余"。之前说"互相印证格式没用"是把两个不是同一个问题的实验混成了一个结论。

**更准确的对照关系**：A测的是"提示词层面让格式更合规，会不会顺带让内容也变好"——她的答案是不会，即**格式达标本身不保证内容质量**。R1测的是"去掉训练时的格式reward这个信号，整体训练会不会变差"——结果是会（Answer F1、reward都降了），即使format_error这个表面症状没有大幅改善。这两件事不矛盾，但也不是同一个结论：合起来看更像是提示"格式reward在训练过程里起作用的机制，可能不是单纯体现在'降低格式错误率'这个表面指标上，而是通过别的路径（比如提供更容易学到、能稳定训练的信号）间接帮助了整体学习"——这是一个需要更多证据才能确认的假说，不是现在就能下的结论，仍然要等R1的validation split结果。

## 2026-11-02 四条实验线全部跑完；R1-R4的结论被证明不可信（三次重测r3-g4波动一倍）

13个job全部`COMPLETED`：R1-R4 train split干净重跑（`13634549`~`13634552`）、R1-R4 validation split（`13618776`/`13633787`~`13633790`）、R5检索top-k消融（`13633791`）、组合实验temperature=1.0复验（`13633792`/`13633793`）、组合实验480题确认（`13633796`）。

### R1-R4 train split"干净重跑"结果（seed599路径，2026-11-01跑完）

| | mean reward | Answer F1 | Evidence F1 | answered | format_error |
|---|---|---|---|---|---|
| ref | 0.620 | 0.198 | 0.097 | 66 | 25 |
| r1-format | 0.647 | 0.226 | 0.083 | 58 | 34 |
| r2-evidence | 0.626 | 0.196 | 0.103 | 60 | 32 |
| r3-g4 | 0.713 | 0.258 | 0.107 | 68 | 25 |

### R1-R4 validation split结果（2026-11-01跑完）

| | mean reward | Answer F1 | Evidence F1 | answered |
|---|---|---|---|---|
| ref | 0.676 | 0.193 | 0.195 | 54 |
| r1-format | 0.601 | 0.180 | 0.162 | 46 |
| r2-evidence | 0.698 | **0.216** | 0.193 | 52 |
| r3-g4 | 0.657 | 0.204 | 0.176 | 49 |
| r4-limit1000 | 0.651 | 0.190 | 0.168 | 57 |

### 关键发现：r3-g4的"明显领先"是噪声，不是真实效果

把同一个r3-g4 checkpoint三次在train split上测出的Answer F1放一起：第1次（已丢失的原始数据）**0.237**，第2次（被`13534452`意外覆盖后测到的）**0.1252**，第3次（这次`13634552`干净重跑）**0.258**——波动幅度接近翻倍。同期ref自己三次测（0.197/0.2078/0.198）相对稳定得多。更关键的是：提交`13634549`~`13634552`这批job时，`--seed`没真正传给vLLM生成这个bug（`backend/vllm_client.py`，commit `3d7c43b`）还没修好，所以这次"干净重跑"本质上还是未加真随机种子约束的又一次独立随机抽样，不比前两次更可信。

**validation split结果进一步推翻了"r3-g4最好"的判断**：Answer F1最高的是r2-evidence（0.216），r3-g4只有0.204，排在中间，五个配置的validation split结果全部挤在0.180~0.216这个窄区间内。

**结论**：n=20这个样本量下的测量噪声，大到足以盖过R1-R4之间的真实效应差异。之前"r3-g4应作为新默认训练配置"这个判断不能再用。下一步需要在`3d7c43b`这个seed修复实际生效的情况下重新测（CARC worktree需要先`git checkout origin/main`同步到这次commit之后），而且大概率还需要加大样本量（更大的G或更大的n）才能把真实效应从噪声里分离出来——这正是TRAINING_NOTES.md"下一轮调参计划"优先级0已经在推进的方向。

### R5（检索top-k消融，推理侧）结果——单调趋势，比单点对比更可信，但仍是单次n=40，未算显著性

| search-limit | Answer F1 | Evidence F1 | answered/40 |
|---|---|---|---|
| k=1 | **0.184** | 0.133 | 30 |
| k=3 | 0.164 | 0.168 | 23 |
| k=5 | 0.145 | **0.233** | 33 |

Answer F1随k增大单调下降，Evidence F1随k增大单调上升——检索段落越多，"抓到对的证据"概率越高（符合直觉），但答案准确率反而变差，可能是段落变多后干扰了模型判断。三个点方向一致是个积极信号，但未排除噪声，未算统计显著性。

### 组合实验temperature=1.0复验——seed601数据可信，seed602数据因job ID笔误待重查

`13633792`（seed=601）：control Answer F1=0.2236，treatment=0.2285，差异仅**+0.0048**，远小于原始temperature=0那次（13467754）的+0.03。`13633793`（seed=602，2026-11-02用正确job ID重新查证）：control Answer F1=0.19294/Evidence F1=0.225，treatment Answer F1=0.18954/Evidence F1=0.13333，差异为**-0.0034（treatment反而更差）**，extra_answer_attempts=8。两次temperature=1.0真随机复验一正一负、都接近0，跟原始temperature=0那次的+0.03完全对不上——这个改动在真随机采样下的效应量很不稳定。

### 组合实验480题统计显著性确认（`13633796`）——配对t检验已算出：不显著

control Answer F1=0.1916，treatment=0.2123，差异**+0.0207**——比原始40题那次（+0.0297）更小。用`control`/`treatment`各自的`agent.jsonl`按`question_id`配对、算Answer F1差值的配对t检验（2026-11-02实算，非估算）：

```
n=480, mean_diff=+0.0206, sample_std=0.3152, t_stat=1.435
```

`|t_stat|=1.435`低于95%置信度双尾检验的1.96门槛，**未达到统计显著**。原先按40题效应量（+0.0297）算出的436题门槛，是针对那个效应量算的；实际480题测出来的效应量更小（+0.0206）、标准差更大（0.3152，高于原估的0.2209），导致即使样本量达标也还是不显著。treatment臂的`extra_answer_attempts`达到128/480（约27%），说明"轮数耗尽强制作答"这个机制在更大、更多样的480题样本上触发频率明显高于原始40题样本，可能是效应量被稀释的一个原因。

**结论（2026-11-02）**：组合实验方向上仍然偏正，但目前拿到的最强证据（480题配对t检验）没有达到统计显著，且两次temperature=1.0复验一正一负——维持"暂不并入`agent_loop.py`生产默认行为"的决定，不是否定这个改动，是还没有达到能下结论的证据强度。

## 2026-11-03 R5确认（480题）结果：方向与40题版本一致，但是取舍而非"没有增益"

把R5（检索top-k消融）从40题放大到480题重测（`13672705`，`checkpoints/ref`在k=1/3/5分别跑一遍）：

| search-limit | Answer F1 | Evidence F1 | answered/480 |
|---|---|---|---|
| k=1 | **0.2007** | 0.2089 | 364 |
| k=3（当前默认） | 0.1967 | 0.2410 | 330 |
| k=5 | 0.1786 | **0.2498** | 338 |

跟40题那次（`13633791`）方向完全一致：Answer F1随k增大单调下降，Evidence F1随k增大单调上升，480题样本量下再次确认不是噪声。三个k的`latency_s`几乎相等（12.7~12.9秒），跟R10不同，这里**改k值没有延迟代价**。

**结论**：这不是"改了没有增益"（跟R10不是同一种情况——R10是质量确实提升但延迟代价过大，明确不该改）。R5是一个真实的指标间取舍：没有哪个k在Answer F1和Evidence F1上同时占优，k=1偏向答案准确率，k=5偏向证据召回。要不要把默认值从k=3换掉，取决于产品更看重哪个指标，不是纯统计能回答的问题。**当前决定（2026-11-03，用户确认）**：k=3作为两个指标之间比较均衡的折中，暂时保留为默认值——不是因为它在某个指标上最优，而是没有明确理由往某一个极端切换。此判断尚未对k=1 vs k=3的逐题差异做配对显著性检验（目前只有聚合后的`comparison.json`，没有按`question_id`对齐的原始数据）；如果后续产品方向明确要优先哪个指标，应先补这一步再改默认值。

## 当前排队/进行中的实验（2026-11-03）

训练数据/步数放大、learning-rate/beta/group size/evidence权重调参（优先级1-5）、R1-R3决定性复测、R5训练侧、R10训练侧、组合实验1839题确认，全部还在CARC排队或运行中，尚无结果：

| 实验 | Job ID | 状态 |
|---|---|---|
| R1决定性复测（zero-format-reward，320题） | `13708657` | 排队中 |
| R2决定性复测·ref基线（320题） | `13708659` | 排队中 |
| R2决定性复测·r2-evidence（320题） | `13708660` | 排队中 |
| R3决定性复测（g4变体，320题） | `13708661` | 排队中 |
| P1训练（数据+步数各放大到1000） | `13708663` | 排队中 |
| P2训练（group size=6） | `13708662` | 排队中 |
| P2b训练（group size=8，优先级2收尾） | `13708736` | 排队中 |
| P3效果评估（learning-rate=1e-5） | `13708682` | 排队中 |
| P4效果评估（beta=0.01） | `13708683` | 排队中 |
| P5训练（evidence权重=0.2，优先级5） | `13708740` | 排队中 |
| P5训练（evidence权重=1.0，优先级5） | `13708741` | 排队中 |
| R5训练侧（search-limit=1，推理侧固定3评估） | 待提交 | 代码已推送（`1fc829c`） |
| R5训练侧（search-limit=5，推理侧固定3评估） | 待提交 | 代码已推送（`1fc829c`） |
| R10训练侧（hybrid retriever接入GRPO rollout） | `13708664` | 排队中 |
| 组合实验1839题确认 | `13679161` | 排队中 |

R6（检索token loss mask消融）已正式关闭，不做（详见`TRAINING_NOTES.md`）。

**全量训练（QASPER train split全部2593题跑满一个epoch）还没有安排**：先等P1（数据/步数放大到1000，10倍于原始配置）的效果评估结果出来，确认"放大数据+步数"这个方向确实有收益之后，再决定是否值得投入全量规模——按P3/P4同配置（`--num-generations`默认值2）实测的每步耗时换算，全量大约需要8-9小时GPU时间，需要用`sbatch --time=X`显式覆盖脚本默认的2小时上限（P1当时用`--time=14:00:00`已验证分区接受显式覆盖，实际允许的最大时长要在CARC上用`scontrol show partition gpu | grep -i MaxTime`确认）。

**新发现的工程优化点（2026-11-03，还没验证）**：R10混合检索53倍延迟（6.6秒→351.1秒/题）的可疑主因找到了——`SentenceTransformerEmbedder`一直硬编码`device="cpu"`，即使评估本身已经分配了GPU节点，`bge-m3`向量模型也完全跑在CPU上。已改成可配置（`scripts/run_hybrid_retrieval.py`读`EMBEDDER_DEVICE`环境变量，默认仍是`cpu`保持向后兼容，不默认改行为），但**还没在CARC上实际测过`EMBEDDER_DEVICE=cuda`会不会跟vLLM的`--gpu-memory-utilization 0.80`抢显存导致OOM**——这是下一步可以验证的具体优化方向，不是"已经没有工程可优化的点了"。

## 2026-11-05 组合实验1839题决定性配对检验：Answer F1显著，已合并进生产`agent_loop.py`

用`scripts/paired_ttest_agent.py`（新写的配对检验脚本，按`question_id`对齐`control`/`treatment`各自的`agent.jsonl`）对`13679161`（1839题，温度=0，跟原始40题/480题同一套贪婪解码方法论）算出：

```
metric=answer_f1 n=1839 mean_diff=+0.0234 sample_std=0.3570 t_stat=2.811   -> SIGNIFICANT
metric=evidence_f1 n=1839 mean_diff=+0.0074 sample_std=0.2604 t_stat=1.213 -> NOT significant
```

`|t_stat|=2.811`超过95%置信度双尾检验的1.96门槛，**Answer F1这个主指标首次达到统计显著**。此前两次temperature=1.0真随机复验（seed601: +0.0048，seed602: **-0.0034**）看起来自相矛盾，但复核后那只是n=40样本量本来就测不出这个量级效应的正常表现（功效不足，不是真矛盾）——1839题是专门按480题实测的效应量/标准差反算出来、为了测准这个效应而设计的决定性样本，证据强度明显更高。

**决定（2026-11-05，用户确认）：合并进生产`agent_loop.py`**，不再只是"候选设计"。实现（`backend/agent_loop.py`的`run_episode`）：
- 每条轨迹的**第一次**`<search>`，不管模型自己想搜什么，都强制改成原始问题文本（之后的搜索不受影响）。
- 轮数耗尽时不再直接判定`max_rounds`失败，改成追加一轮强制作答（`FORCE_ANSWER_INSTRUCTION`），模型给出有效`<answer>`则记`answered`，否则记新的`forced_answer_invalid`。

**范围说明：这次只改了推理侧**，`backend/token_rollout.py`里GRPO训练用的`run_token_episode`是完全独立的一套token级实现，没有改动——组合实验当初验证的是"未训练基础模型在推理时"的行为，没有测过"训练rollout也这样做"会怎样，之后如果想让训练也用上这两个机制，需要单独做消融验证，不能直接搬。

**副作用**：`scripts/run_agent_ablation.py`（及`carc_eval_agent_ablation.sh`）的`first-search`/`final-answer`/`combined`三种实验类型现在全部**结构性失效**——它们本来靠在`run_episode`外面包一层`SearchProxy`+事后强制作答来实现"control vs treatment"对照，现在这两个效应已经被`run_episode`无条件应用，不再区分experiment/arm，这层对照逻辑已经测不出任何差异了（`tests/test_agent_ablation.py`已更新，保留测试文件但注释说明这一点，不代表这个脚本还能用来测出新东西）。这个脚本已经完成了它的历史任务（撑起了从40题到1839题的整条验证链路），但目前的三个实验类型不建议再用它重测。

**顺带修的一个asyncio陷阱（`backend/main.py`）**：改完`run_episode`后，`tests/test_main.py`的max_rounds测试一度**真的卡死**（不是变慢，是hang）——根因是`StopIteration`从worker线程传到`asyncio.to_thread`包的`Task`边界时，`asyncio.Future.set_exception()`拒绝接收裸的`StopIteration`（CPython故意设计，防止跟generator协议混淆），导致`await task`永远等不到结果，`except Exception`完全走不到。修法：在跨线程边界之前把`StopIteration`转成`RuntimeError`再抛出去。这不是这次改动引入的新问题，是一直存在的潜在脆弱点，这次因为新增了一次模型调用才被测试意外暴露出来。
