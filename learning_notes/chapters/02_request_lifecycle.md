# 第二章：一个请求从文本到最终输出

源码基线：`f9c98bd1ebbddb9acedb1e6d7505ad9588f9a266`。承接第一章最后的问题；本章沿同一条离线、单 GPU、MRV2、同步调度、eager、无 prefix cache、`n=1` 路径，把输入和输出接起来。本文调用链已查源码，未采集运行 trace。

## 1. 本章要回答的问题

1. 文本究竟在哪一步变成 token ID？`InputProcessor` 是否就是 tokenizer？
2. 跨进程传输的请求与 Scheduler 中的请求是否是同一个对象？
3. 模型前向怎样接上 logits、采样和下一轮输入？
4. 谁判断长度／EOS 结束，谁处理 stop string？
5. 离线 API 为什么最后只返回一次，内部却运行多轮？

## 2. 先建立直观理解

一次生成有两条相向的数据流。进去的是“输入 token 和生成条件”，出来的是“新增 token 和完成原因”。前端负责理解用户输入和整理文本；核心负责安排、推进请求；Runner 负责把一轮计划变成设备工作。

第一章的教学请求 R 有四个输入 token，生成 55、66 后结束。关键点是：55 第一次出现时是模型的预测结果；下一轮它才作为输入产生自己的 KV。用户不需要看到这个中间协议，只需要最终文本。

**反分词（detokenization）**是把 token ID 增量转换成文本的过程。token 不一定对应一个汉字、单词或完整字节序列，因此终止字符串检查不能简单等同于一个 stop token ID 比较。

## 3. 与 nano-vLLM 的对应关系

可以沿用“加入请求→调度→执行→追加输出”的概念。需要新增两层：一是前端的渲染／校验／输出状态，二是跨进程消息。工作区没有 nano-vLLM 对照源码，本节不指定其函数名。

不要把当前 `LLMEngine.step` 直接套成“Scheduler 执行一轮”：在 MP 路径中，它是在收取核心结果。`EngineCore.step` 才组织调度和执行。第一章已解释通信方式，本章只追踪交接的内容。

## 4. 源码阅读地图

按下表顺序读每个方法的相关分支即可。

| 顺序 | 路径、符号与行号 | 本次要找到的交接 |
| --- | --- | --- |
| 1 | [vllm/entrypoints/offline_utils.py:298](../../vllm/entrypoints/offline_utils.py#L298)，`_add_completion_requests` | 为每条 prompt 生成已预处理的 EngineInput，再加入请求 |
| 2 | [vllm/renderers/base.py:1105](../../vllm/renderers/base.py#L1105)，`BaseRenderer.render_cmpl`；[vllm/renderers/base.py:582](../../vllm/renderers/base.py#L582)，`_tokenize_prompt` | 文本正规化、tokenization、引擎输入构造 |
| 3 | [vllm/v1/engine/input_processor.py:281](../../vllm/v1/engine/input_processor.py#L281)，`InputProcessor.process_inputs` | 校验／参数克隆后构造 EngineCoreRequest，不能把它当成首次文本 tokenization 的唯一入口 |
| 4 | [vllm/v1/engine/llm_engine.py:223](../../vllm/v1/engine/llm_engine.py#L223)，`LLMEngine.add_request` | 分配内部 ID，登记输出状态，将消息交 client |
| 5 | [vllm/v1/engine/core.py:999](../../vllm/v1/engine/core.py#L999)，`preprocess_add_request` | 解码后的消息转换为核心 Request |
| 6 | [vllm/v1/engine/core.py:608](../../vllm/v1/engine/core.py#L608)，`EngineCore.step` | 调度、执行、采样、更新 |
| 7 | [vllm/v1/worker/gpu/model_runner.py:1444](../../vllm/v1/worker/gpu/model_runner.py#L1444)，`GPUModelRunner.sample` | 从哪些 hidden states 计算 logits，何时走普通 sampler |
| 8 | [vllm/v1/core/sched/scheduler.py:2305](../../vllm/v1/core/sched/scheduler.py#L2305)，`_update_request_with_output`；[vllm/v1/core/sched/utils.py:94](../../vllm/v1/core/sched/utils.py#L94)，`check_stop` | 逐个追加 token，再检查核心终止条件 |
| 9 | [vllm/v1/engine/output_processor.py:622](../../vllm/v1/engine/output_processor.py#L622)，`OutputProcessor.process_outputs` | 增量反分词、stop string、前端收尾 |
| 10 | [vllm/entrypoints/offline_utils.py:581](../../vllm/entrypoints/offline_utils.py#L581)，`_run_engine` | 收集 finished 输出并返回用户列表 |

## 5. 真实调用链

为避免重复第一章的大图，本章用“数据包”给链路分段：

1. **前端普通调用**：`LLM.generate` → `_run_completion` → `_add_completion_requests`。其生成器对一条 prompt 调用 `_preprocess_cmpl_one` → `_preprocess_cmpl` → `renderer.render_cmpl`。`render_prompts` 处理输入形式，`tokenize_prompts` 最终在纯文本分支调用 `_tokenize_prompt`，得到 `prompt_token_ids`，再经 `process_for_engine` 返回 EngineInput。
2. **前端普通调用**：`_add_request` → `LLMEngine.add_request` → `InputProcessor.process_inputs`。生成参数被克隆、结合模型生成配置更新，随后构造 EngineCoreRequest。`assign_request_id` 保存 external ID 并产生内部唯一 ID；`OutputProcessor.add_request` 登记前端 RequestState。
3. **ZMQ 消息→核心线程队列**：`SyncMPClient.add_request` 发送 ADD；核心输入线程解码 EngineCoreRequest，经 `preprocess_add_request` 调用 `Request.from_engine_core_request`。交进 input_queue 的已经是核心 Request 与 wave 信息。核心主循环再调用 `EngineCore.add_request`，进入 Scheduler。消息不携带 Python 对象身份。
4. **核心普通调用→GPU 工作**：`schedule` 生成 SchedulerOutput；uni 分发到 Worker 和 MRV2。Runner 准备 inputs、positions、attention metadata 后发起前向，保存 hidden states；`sample_tokens` → `sample` 选取需要的 hidden states，计算 logits，普通生成调用 sampler。
5. **设备结果回收→核心普通调用**：获得 ModelRunnerOutput 后，`update_from_output` 用 `req_id_to_index` 匹配请求，追加生成 token，检查终止，构造 EngineCoreOutput。不要按 Scheduler 字典顺序直接索引 GPU 输出；映射见 [vllm/v1/core/sched/scheduler.py:1929](../../vllm/v1/core/sched/scheduler.py#L1929)。
6. **核心线程队列→ZMQ→前端队列**：内部输出回到 `LLMEngine.step`；OutputProcessor 查内部 ID 对应的 RequestState，反分词并整理为用户 RequestOutput。FINAL_ONLY 的中间结果不对用户输出。结束后 `_run_engine` 返回完成列表。

**源码事实：**核心可因 token ID 和长度结束；前端可以在 detokenization 时发现 stop string，向核心发 abort。两处终止检查有不同的输入和资源所有权，不能删掉其中一处当作“去重优化”。

## 6. 关键源码精读

### 6.1 文本转 token 的确切位置

`BaseRenderer._tokenize_prompt`：

<!-- source: vllm/renderers/base.py:587-597 -->

```python
tokenizer = self.get_tokenizer()
want_offsets = self._wants_offsets(prompt, params)
kwargs = params.get_encode_kwargs()
if want_offsets:
    kwargs = {**kwargs, "return_offsets_mapping": True}
encoding = tokenizer(prompt["prompt"], **kwargs)
return self._build_tokens_prompt(
    encoding["input_ids"],
    prompt,
    offset_mapping=encoding["offset_mapping"] if want_offsets else None,
)
```

输入为文本 prompt 和 tokenization 参数；输出 TokensPrompt。这里真正调用 tokenizer，保留必要的 prompt 信息以及可选字符映射。**设计解释：**模型输入准备不只是 `encode(text)`，还要支持输入选项和之后的输出／日志关联。用户已经给 token ID 时会绕过此文本分支，不应再无条件编码。

### 6.2 从 hidden states 到 logits

`GPUModelRunner.sample` 的非 batch-sharding 分支：

<!-- source: vllm/v1/worker/gpu/model_runner.py:1465-1467 -->

```python
else:
    sample_hidden_states = hidden_states[input_batch.logits_indices]
    logits = self.model.compute_logits(sample_hidden_states)
```

输入是前向 hidden states 和本轮 `logits_indices`；输出用于选择下一 token 的 logits。它没有改 Scheduler 的 token 列表，而是给后面的 sampler 提供设备张量。**设计解释：**prefill 的每个输入位置都参与模型计算，不代表每个位置都需要成为对外新增输出；需要哪些预测行，由 logits 索引和后续有效输出处理共同决定。无投机时调用普通 sampler 的分支在同文件第 1484 行。

### 6.3 追加之后再判断结束

`Scheduler._update_request_with_output`：

<!-- source: vllm/v1/core/sched/scheduler.py:2312-2322 -->

```python
stopped = False
for num_new, output_token_id in enumerate(new_token_ids, 1):
    request.append_output_token_ids(output_token_id)

    # Check for stop and update request state.
    # This must be called before we make the EngineCoreOutput.
    stopped = check_stop(request, self.max_model_len)
    if stopped:
        del new_token_ids[num_new:]  # Trim new tokens if needed.
        break
return new_token_ids, stopped
```

输入是一轮生成的 token 列表，普通生成通常一个，投机时可以多个；输出是截断后的有效列表与 stopped 标记。它同时改变 Request 的输出序列和终态；后面的 EngineCoreOutput 使用这些结果。这样长度上限或 EOS 在一批 token 中间触发时，后续 token 不会继续提交。

### 6.4 FINAL_ONLY 仍需消费中间结果

`RequestState.make_request_output`，见 [vllm/v1/engine/output_processor.py:283](../../vllm/v1/engine/output_processor.py#L283)：

<!-- source: vllm/v1/engine/output_processor.py:292-297 -->

```python
finished = finish_reason is not None
final_only = self.output_kind == RequestOutputKind.FINAL_ONLY

if not finished and final_only:
    # Only the final output is required in FINAL_ONLY mode.
    return None
```

输入包括新 token 和完成原因；输出可以是 None。到这里之前 detokenizer 已经消费新 token，前端状态并没有因为 FINAL_ONLY 而停滞。**设计解释：**内部持续推进，外部按 API 输出模式决定可见性。

stop string 的后续闭环在 [vllm/v1/engine/output_processor.py:695](../../vllm/v1/engine/output_processor.py#L695) 与第 735 行：反分词发现字符串后可结束前端；若核心尚未结束，将内部 ID 放入 `reqs_to_abort`。`LLMEngine.step` 再通知核心，不能只在前端删一个字典项就认为 KV 已经释放。

## 7. 贯穿案例

**教学示例，非运行日志。**仍用四个 prompt token、两个输出，预算充足、无 EOS。

| 边界 | 进入的内容 | 离开的内容／状态 |
| --- | --- | --- |
| renderer | 一段字符串 | 教学 token ID `[11,22,33,44]`，不承诺真实文本对应这个编码 |
| InputProcessor | EngineInput、SamplingParams | EngineCoreRequest；`max_tokens=2`；external ID 与内部 ID 分开 |
| 核心输入线程 | 解码后的消息 | 新核心 Request，初始 WAITING，computed=0 |
| 第一轮核心更新 | 本轮计算 4 个输入，采样得到 55 | Request 输出 `[55]`；返回内部 new_token_ids `[55]` |
| 第一轮前端处理 | `[55]` 且无 finish_reason | 更新累计 token 和文本，FINAL_ONLY 返回 None |
| 第二轮核心更新 | 输入 55，采样得到 66 | 输出累计 `[55,66]`；达到 max_tokens，finish_reason 对应 length |
| 第二轮前端处理 | `[66]` 及完成原因 | 生成包含累计输出的 RequestOutput，外部 ID 返回用户 |

KV 仍遵循第一章表格：第一轮 prompt 的 KV 有效；第二轮新增 55 的 KV；结束时无需再计算 66 的 KV。若改成 stop string 场景，结束原因可能由前端发现，核心通过后续 abort 收尾，而不是沿上述长度终止分支。

## 8. 动手实验

**本轮已运行验证**：源码检索、摘录与引用检查。**本轮未运行**：tokenizer、pytest、GPU 推理。

先在 PowerShell 做无需依赖的定位练习：

```powershell
rg -n 'tokenizer\(prompt|return EngineCoreRequest|from_engine_core_request' vllm/renderers/base.py vllm/v1/engine/input_processor.py vllm/v1/engine/core.py
rg -n 'logits_indices|compute_logits|self.sampler\(' vllm/v1/worker/gpu/model_runner.py
rg -n 'check_stop|reqs_to_abort|final_only' vllm/v1/core/sched/scheduler.py vllm/v1/engine/output_processor.py
```

观察每处输入、返回值和调用者，画出上面六个交接。预期能指出 tokenization 与 InputProcessor 的不同职责，而不是只找到同名 token 字段。

**仅提供步骤，待第一章环境与本地模型条件满足后**，运行第一章的 `minimal_generate.py`。观察真实 `prompt_token_ids`、`token_ids`、`finish_reason`，把第七节教学 ID 换成真实记录；不要比较不同 tokenizer 的 token 个数。脚本没有内部逐步日志，仅最终输出，不能拿它的输出假装已经观测到 Scheduler 中间状态。

没有 GPU 时，把 max_tokens 改为 1 作纸面推演：第一轮即可完成，无需第二轮 forward；再解释“前端发现 stop string”的 abort 边。标准库源码检查命令沿用第一章，不需要安装依赖。

## 9. 易错点与自我检验

1. 对纯文本输入，在哪个方法可以看到 tokenizer 的实际调用？
2. 核心输入队列中的 Request 与前端 EngineCoreRequest 为什么不能比较对象身份？
3. 为什么 logits 只取某些 hidden-state 行？
4. 第一轮前端没有返回 RequestOutput，是否意味着没处理 55？
5. 核心检查了 EOS，前端为什么还要检查 stop string？
6. 一次返回多个 token 时，长度上限在中间命中怎么办？
7. 最终 RequestOutput 的 request_id 为什么不应直接照抄内部随机后缀 ID？

## 10. 本章总结

现在应能从文本入口一路定位 tokenization、请求消息、核心 Request、forward、logits、采样、停止与最终输出，解释每个交接是谁产生、谁消费。仍未展开字段生命周期、分配策略、设备流和采样概率细节。

下一章固定这些对象的契约：同一个请求在前端、核心、Worker 各有什么状态，哪些 ID／计数必须对齐，结束或取消后各由谁清理。这是修改调度和投机逻辑前的必要条件。

### 自我检验答案

1. `BaseRenderer._tokenize_prompt`，实际调用 `tokenizer(prompt["prompt"], **kwargs)`。
2. 先经过序列化、跨进程和 `Request.from_engine_core_request`；协议字段关联请求，不依赖同一 Python 实例。
3. 前向需要处理输入序列，但本轮只对需要预测的位置做输出选择；看 `logits_indices`，不要用输入行数代替新增 token 数。
4. 已处理。detokenizer 先更新，FINAL_ONLY 只限制对外产出。
5. 两者输入不同：核心有 token 与长度，前端有增量文本；字符串结束可能还需通知核心 abort。
6. 逐个追加、逐个 check_stop，命中后截断本轮剩余新 token。
7. RequestState 保存 external ID；用户输出使用该 ID，内部 ID 用于消息路由和状态查找。
