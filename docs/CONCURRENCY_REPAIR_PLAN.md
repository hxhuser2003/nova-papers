# 并发保护与诊断计划（2026-11-22）

## 实施范围

P0/P1已实现。每个应用进程最多2个活跃会话、5个等待会话，等待60秒超时；队列满返回SSE error code=queue_full，超时code=queue_timeout。SSE响应可能仍是HTTP200，因此压测必须解析error事件。仅支持单个uvicorn worker；多worker/副本不能靠这个进程内限流共同保护同一GPU。

**P0已在真实Colab部署验证（2026-11-22，`app_server.log`的`chat_timing`日志，非压测脚本自己的错误样例采样）**：并发16那一档，日志显示2+5=7个请求正常进入`admission_wait_start`→`admission_acquired`，随后连续9个请求的`admission_wait_start`后立即跟着`queue_full`（间隔20~130毫秒），跟"2运行+5等待=7个占满、其余全部立即拒绝"的设计预期精确吻合。被接纳的请求里观察到的最长等待约40.5秒，均在60秒超时阈值内成功`admission_acquired`，这次日志里没有出现`queue_timeout`。**结论：限流/排队机制本身按设计工作，验证通过。**（这件事不能单靠压测脚本的汇总JSON看出来——脚本每档只打印`errors[0]`一条错误样例，并发4/8/16里看到的"forced_answer_invalid"/"repeated_format_error"是被接纳后agent自己跑输了，不代表限流没生效；必须看`chat_timing`日志里的`queue_full`计数才能确认。）

同一次压测的并发1（零竞争）耗时27.93秒，而本文档此前一轮压测（旧Colab会话）同样零竞争的并发1只要6.35秒——两轮压测是不同的Colab虚拟机/物理GPU分配，基线延迟本身就有4倍以上差异，**不能跨会话直接比较延迟/吞吐数字**，只能比较"限流机制是否按设计触发"这类结构性结论。

客户端断开后，不再启动下一次模型调用，但已经在同步HTTP中的调用不会被强制终止；名额保留到worker结束。vLLM HTTP超时是独立设置，队列60秒并不限制已开始的整个Agent会话耗时。

将queue.Queue + to_thread(queue.get)改为asyncio.Queue + call_soon_threadsafe，去掉等待SSE进度时额外占用的线程。这是明确的资源占用改善，**不是对历史并发8卡死根因的已验证结论**。旧注释“N个串行Agent产生远多于N个同时vLLM请求”不准确；当前每个Agent逐轮串行调用。

## P1日志

日志前缀chat_timing，包含request_id、wall_time、elapsed_s、stage；响应头X-Request-ID可关联。

- request_received / find_paper_start/end
- admission_wait_start / admission_acquired / queue_full / queue_timeout
- worker_submitted / worker_started：差值为线程调度等待
- vllm_call_start/end/error：每轮模型客户端调用，带call序号
- agent_step_end、worker_finished、admission_released、stream_closed

不记录问题、模型思考或原始证据。vllm_call时间包含HTTP连接池/网络/vLLM排队与推理，不能只凭长调用认定vLLM排队。采集同时间段vLLM请求运行数/等待数、服务日志、GPU采样后才能进一步定位。find_paper目前同步运行，其耗时已可见；若它阻塞事件循环再针对修复。

## P1部署复验与P3稳定性（第2项已用真实日志初步验证，其余待做）

1. 单worker启动，记录代码commit及dirty diff、模型revision、vLLM版本和参数、MAX_CONCURRENT_CHATS=2、MAX_QUEUED_CHATS=5、CHAT_QUEUE_TIMEOUT_SECONDS=60。**（本次验证未记录commit/dirty diff/模型revision，下次复验应补上）**
2. ✅ 先单请求探活，再并发8。8个同时到达时，2运行+最多5等待，其余queue_full属于预期保护，应与vLLM/Agent失败分别统计。**2026-11-22已用并发16验证：9个请求被`queue_full`立即拒绝，其余7个正常排队/运行，无`queue_timeout`。**
3. 并发2、3各重复5轮，每轮用同一28题清单，保留请求原始JSONL和request_id、冷启动/预热条件。并发3指客户端负载，不是把服务端上限调成3。
4. 在GPU主机同期运行：`nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,memory.used --format=csv --loop=1 > gpu-load.csv`。启动/结束时记录UTC；核对模型进程实际GPU UUID。GPU 0%可能是采样时间、错卡、CPU/网络阻塞或模型退出，不能直接定因。
5. 分别报告成功请求p50/p95、所有请求耗时、成功吞吐、queue_full/queue_timeout、HTTP/vLLM/Agent失败。记录活跃及等待时段，避免用快速拒绝“优化”平均延迟。

## P2条件决策及P4边界

没有真实诊断前不改max-num-seqs或gpu-memory-utilization。若线程调度等待高，检查线程使用/事件循环阻塞；若模型服务排队高，结合vLLM指标在应用上限2以内逐项调参并看OOM。不能凭8小于默认线程数排除线程池饥饿：当前旧实现每会话多个线程任务，部署线程上限和其他工作负载也需核查。

课程演示以明确背压、有限等待、可诊断错误为交付目标。更高吞吐需经过测量评估批处理/模型成本优化，超出单卡容量则需要多GPU或副本和负载均衡；不宣称P0能修复GPU 0%或提升模型吞吐。
