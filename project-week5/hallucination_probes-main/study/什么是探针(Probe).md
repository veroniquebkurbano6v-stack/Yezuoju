## 什么是探针（Probe）？

**探针是一种诊断工具**，用于"探测"模型内部的状态。类比：

- **医学探针**：插入身体检查内部状况
- **代码探针**：插入模型中间层检查内部表示

### 在LLM中的具体用途：

这个`ValueHeadProbe`是一个**幻觉检测探针**：

- 连接到模型的**特定层**（比如第20层）
- 从该层提取**隐藏状态**（hidden states）
- 通过一个小型线性层**预测每个token是否是幻觉**

```
输入句子 → [模型前N层] → [探针接在这里] → [模型剩余层] → 正常输出
                              ↓
                      [线性层] → 幻觉分数
```

## 代码核心功能解析

### 1. **冻结参数（Freezing Parameters）**

```python
# 在setup_probe函数中：
for _, param in model.named_parameters():
    param.requires_grad = False  # ✅ 正确，冻结所有参数
```

**作用**：

- `requires_grad = False`：该参数在反向传播时**不会更新**
- 只训练探针的线性层，**不改变原始模型权重**
- 保护预训练知识，避免灾难性遗忘

### 2. **探针工作流程**

#### (1) **挂接钩子（Hook）**

```python
def _get_hook_fn(self):
    def hook_fn(module, module_input, module_output):
        # 捕获第layer_idx层的输出
        self._hooked_hidden_states = module_output

    return hook_fn
```

- 在`forward()`时自动调用
- 捕获指定层的隐藏状态

#### (2) **前向传播**

```python
def forward(self, input_ids, attention_mask, labels):
    # 1. 设置钩子
    fwd_hooks = [(self.target_module, self._hook_fn)]

    # 2. 模型正常前向传播
    with add_hooks(fwd_hooks):
        outputs = self.model(input_ids, ...)  # 正常生成文本

    # 3. 同时用隐藏状态计算幻觉分数
    probe_logits = self.value_head(self._hooked_hidden_states)
    # shape: [batch_size, seq_len, 1]
```

**同时输出两个结果**：

- `lm_logits`：模型原本的文本生成结果
- `probe_logits`：每个token的幻觉分数

### 3. **训练和使用场景**

#### **训练阶段**：

```python
# 假设有标注数据：
输入："ChatGPT是OpenAI在2025年发布的"  # 事实错误
标签：[0, 0, 0, 1, 0, 0, 0]  # "2025"位置标注为1（幻觉）

# 训练探针：
probe_output = probe(input_ids, labels=lm_labels)
probe_loss = binary_cross_entropy(probe_output["probe_logits"], hallucination_labels)
```

#### **推理阶段**：

```python
# 实时检测：
output = probe.generate("请描述火星上的城市")
hallucination_scores = output["probe_logits"]  # 每个token的幻觉可能性

# 可以：
# 1. 高幻觉分数时提示用户
# 2. 自动修正或重新生成
# 3. 作为置信度指标
```

### 4. **LoRA集成**

```python
# 可选的增强方式
if probe_config.lora_layers:
    model = setup_lora_for_layers(model, probe_config.lora_layers)
```

**为什么需要LoRA？**

- 如果**仅用线性层不够准确**，可以给某些层加LoRA
- LoRA参数很少（~0.1%），微调部分权重
- 与探针**一起训练**，提升检测精度

## 实际应用例子

假设我们想在GPT生成时检测幻觉：

```python
# 1. 加载预训练探针
probe = ValueHeadProbe.load("path/to/probe")

# 2. 生成文本并检测
prompt = "爱因斯坦发现了相对论在哪一年？"
input_ids = tokenizer.encode(prompt)

with torch.no_grad():
    output = probe(input_ids)

    # 原始生成
    generated_text = tokenizer.decode(output["lm_logits"].argmax(-1)[0])

    # 幻觉检测
    hallucination_scores = torch.sigmoid(output["probe_logits"]).squeeze(-1)

    print(f"生成: {generated_text}")
    print(f"幻觉分数: {hallucination_scores.tolist()}")
    # 可能输出: [0.1, 0.1, 0.05, ..., 0.8, ...]
    # 分数>0.5的可能为幻觉
```

## 关键设计理念

1. **非侵入式**：不改变模型主要架构
2. **轻量级**：仅添加一个线性层（~几MB）
3. **可插拔**：随时添加/移除
4. **多功能**：可用于：
    - 幻觉检测
    - 事实性检查
    - 毒性检测
    - 输出置信度估计

## 你的理解总结

✅ **正确**：`requires_grad = False`就是冻结参数

✅ **探针本质**：是一个"监听器"+ "分类器"

- **监听**：捕获模型中间表示
- **分类**：判断该表示是否对应幻觉

✅ **优势**：不需要重新训练整个大模型，只需训练一个小型探针，**高效且保护原始能力**。