# Real-rollout reward check

Model `Qwen/Qwen2.5-3B-Instruct`, 20 train-split questions x G=5, temperature 1.0, max 500 tokens/round, seed 599.

## Overall

| metric | value |
| --- | --- |
| rollouts | 100 |
| mean reward | 0.545 |
| mean Answer F1 | 0.147 |
| mean Evidence F1 | 0.093 |
| mean rounds | 3.20 |
| groups with zero reward variance (no GRPO signal) | 1/20 |
| mean seconds per question (G rollouts) | 25.6 |

## How rollouts ended

| terminated_reason | count | mean reward |
| --- | --- | --- |
| answered | 60 | 0.734 |
| repeated_format_error | 33 | 0.267 |
| max_rounds | 7 | 0.229 |

## By question type

| type | questions | mean reward | mean Answer F1 | zero-variance groups |
| --- | --- | --- | --- | --- |
| extractive | 7 | 0.587 | 0.179 | 1/7 |
| free_form | 5 | 0.438 | 0.057 | 0/5 |
| yes_no | 5 | 0.541 | 0.160 | 0/5 |
| unanswerable | 3 | 0.629 | 0.200 | 0/3 |

## Sanity check: reward vs. answer quality

| group | rollouts | mean reward |
| --- | --- | --- |
| answered, Answer F1 >= 0.5 | 11 | 1.751 |
| answered, Answer F1 < 0.5 | 49 | 0.506 |
| no answer | 40 | 0.260 |

Actions used: {'search': 126, 'format_error': 106, 'answer': 60, 'read': 28}
