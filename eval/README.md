示例题集（Sample Questions）

从 QASPER validation 集中挑出 40 道题，用于训练前基线测试，以及训练后的效果对比。

文件
文件	说明
sample_questions.jsonl	最终样题集，基线测试直接读取（由 scripts/finalize_sample_questions.py 生成）
review_final.csv	人工审核记录：每道候选题的保留 / 剔除结论和理由

sample_questions.jsonl 每行一道题，字段如下：

字段	说明
question_id / paper_id / question	题目信息
answer_type	extractive / free_form / yes_no / unanswerable
answers	完整参考答案（与 questions.jsonl 一致，含每个答案各自的证据）
evidence_para_ids	所有标注者证据段落的并集
multi_evidence	证据是否分布在 2 段及以上
note	人工审核时的选题理由
挑选流程
规则筛选（scripts/select_sample_questions.py）：validation 集共 1005 道题，排除以下几类：
标注者答案类型不一致：198 道
涉及图表证据（has_non_text_evidence）：164 道
缺少证据段落：40 道
按题型配额抽样：每类题中尽量一半是多段证据题，同一篇论文最多 2 道，随机种子固定为 42。
人工审核：逐题对照证据原文检查，剔除 9 道，再从备选题中补齐。
最终构成
题型	数量	其中多段证据题
extractive	15	11
free_form	10	8
yes_no	10	8
unanswerable	5	—
合计	40	27

40 道题来自 37 篇不同的论文。

人工审核标准

保留：问题只问一件事、几位标注者答案一致、答案能在证据中找到或推出、答案文本干净。

剔除的 9 道题及原因：

题型	题目	剔除原因
extractive	How are the different components of the model trained? Is it trained end-to-end?	复合问题，标注者答案差异大，含公式
extractive	What are the baselines?	参考答案过长且含公式，F1 不可靠
extractive	What is the source of the news sentences?	两位标注者答的不是一回事，问题有歧义
extractive	What are the state of the art models?	答案只有引用标记（BIBREF），没有实际内容
free_form	How is pseudo-perplexity defined?	标注内容缺失，且依赖公式
free_form	Do they use datasets with transcribed text or do they determine text from the audio?	标注者答案矛盾
yes_no	Is the template-based model realistic?	问题含糊，证据中没有对应说法
yes_no	Is this dataset publicly available for commercial use?	标注者答案矛盾（No vs Yes）
unanswerable	What is the performance reported for the best models in the VLSP 2018 and VLSP 2019 challenges?	有歧义：论文中有相关 WER 数字，模型很可能据此作答
审核中的发现（对后续工作的提示）
列举类问题（What are the baselines / datasets）的参考答案常常整段照搬原文，Answer F1 会偏低，评估时要注意。
证据文本中有占位符：INLINEFORM / DISPLAYFORM 是被移除的公式，BIBREF / TABREF / FIGREF 是引用标记。建议在提示词中说明，避免模型把它们当成答案抄出来。
是非题标注者答案矛盾（一人 Yes、一人 No）的情况，原规则没有排除。select_sample_questions.py 已补上这条过滤。
复现
bash
python scripts/finalize_sample_questions.py   # 根据 review_final.csv 生成 sample_questions.jsonl

注意：不要再运行 select_sample_questions.py，否则会用未经人工审核的结果覆盖 sample_questions.jsonl。
