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
