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
