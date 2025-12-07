## 关键方法

### _process_item()

#### 主要流程：

1. 将对话（prompt + completion）格式化为模型输入
2. 使用tokenizer进行编码
3. 计算每个token的标签（支持/幻觉/忽略）
4. 返回包含输入、标签、权重的字典

### _compute_positional_labels()

#### 标注策略

- 标签类型：
    * 1.0：幻觉（不真实的内容）
    * 0.0：有支持的事实
    * -100.0：忽略（包括prompt部分、padding、buffer区域等）

- 标注过程：
    1. 找到assistant回复的起始位置
    2. 对每个标注的span（文本片段）：
        - 在tokenized序列中定位该span
        - 根据span.label设置token标签
        - 如果设置了ignore_buffer，标注周围的token为忽略
        - 如果设置了last_span_token，只标注span的最后一个token
    3. 设置权重：正负样本可以有不同的权重

### 数据处理流程

输入: ProbingItem (包含prompt, completion, spans列表)
↓
应用聊天模板: 格式化对话
↓
Tokenization: 转换为token序列
↓
Span对齐: 将文本span映射到token位置
↓
标签分配: 为每个token分配标签和权重
↓
输出: 包含完整训练数据的字典

### 标注示例

用户：巴黎是哪个国家的首都？
助手：巴黎是法国的首都，人口约2200万。

标注：span="2200万", label=1.0（幻觉，实际约210万）
Token级别的标注可能是：
[巴黎, 是, 法国, 的, 首都, ，, 人口, 约, 2200, 万, 。]
标签：[忽略, 忽略, 忽略, 忽略, 忽略, 忽略, 忽略, 忽略, 1.0, 1.0, 忽略]

### 核心数据结构

{
"input_ids": torch.Tensor, # token IDs
"attention_mask": torch.Tensor, # 注意力掩码
"classification_labels": torch.Tensor, # 分类标签 (1.0/0.0/-100.0)
"classification_weights": torch.Tensor, # 每个token的权重
"pos_spans": List[List[int]], # 幻觉片段的位置
"neg_spans": List[List[int]], # 支持事实的位置
"lm_labels": torch.Tensor, # 语言模型标签（用于对比学习）
}

### 辅助功能

调试模式：print_token_labels()用颜色显示token标注情况

数据集拼接：支持用+操作符合并两个数据集

动态处理：可配置为预先处理或实时处理数据

数据统计：自动统计正负样本数量和跳过的span

### 创建函数 create_probing_dataset()

主要步骤：

1. 从HuggingFace加载原始数据集
2. 根据数据集名称选择合适的预处理函数
3. 转换为统一的ProbingItem格式
4. 创建TokenizedProbingDataset实例

### 数据加载器适配 tokenized_probing_collate_fn()

将不同长度的序列padding成批次数据，确保数据加载器能正确处理。