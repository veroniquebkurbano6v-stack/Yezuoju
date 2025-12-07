# 长文本生成中幻觉实体的实时检测

这是论文《长文本生成中幻觉实体的实时检测》对应的代码库：

- 论文链接: [arxiv.org/abs/2509.03531](https://arxiv.org/abs/2509.03531)
- 项目网站: [hallucination-probes.com](https://www.hallucination-probes.com/)

## 数据集

所有长文本数据集都作为[HuggingFace集合](https://huggingface.co/collections/obalcells/hallucination-probes-68bb658a4795f9294a73b991)
提供。包括：

- 长文本生成的标记级注释：
    - [LongFact 注释](https://huggingface.co/datasets/obalcells/longfact-annotations)
    - [LongFact++ 注释](https://huggingface.co/datasets/obalcells/longfact-augmented-annotations)
    - [HealthBench 注释](https://huggingface.co/datasets/obalcells/healthbench-annotations)
- 用于生成长文本的提示词：
    - [LongFact++ 提示词](https://huggingface.co/datasets/obalcells/longfact-augmented-prompts)

## 预训练探针

各种大语言模型的预训练幻觉检测探针可在以下位置获取：[obalcells/hallucination-probes](https://huggingface.co/obalcells/hallucination-probes)

我们提供三种类型的探针：

- **线性探针**（`*_linear`）：在模型隐藏状态上训练的简单线性分类器
- **带KL正则化的LoRA探针**（`*_lora_lambda_kl_0_05`）：带KL散度正则化（λ=0.05）的LoRA适配器，对生成质量影响最小
- **带LM正则化的LoRA探针**（`*_lora_lambda_lm_0_01`）：带交叉熵损失正则化（λ=0.01）的LoRA适配器

支持的模型包括：

- Llama 3.3 70B
- Llama 3.1 8B
- Gemma 2 9B
- Mistral Small 24B
- Qwen 2.5 7B

## 代码

### 环境设置

要设置环境变量，请将 `env.example` 复制到 `.env` 并填写值。

使用 `uv` 进行环境设置的步骤如下：

```bash
# Install Python 3.10 and create env
uv python install 3.10
uv venv --python 3.10

# Sync dependencies
uv sync
```

### 训练探针

根据需要编辑 `configs/train_config.yaml`（模型、数据集、LoRA层、学习率等）。然后运行：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m probe.train --config configs/train_config.yaml
```

输出（默认）保存在 `value_head_probes/{probe_id}` 下。要上传到Hugging Face，请在配置中设置 `upload_to_hf: true`，并确保在
`.env` 文件中设置 `HF_WRITE_TOKEN`。

### 运行注释管道

此管道使用前沿大语言模型和网络搜索来标记实体并对齐标记级跨度。所需环境变量：

```bash
export ANTHROPIC_API_KEY=...   # for annotation
export HF_WRITE_TOKEN=...      # to push to HF datasets
```

运行（完整参数请参见 `annotation_pipeline/README.md` 和 `run.py`）：
vkjxzlvjh

```bash
uv run python -m annotation_pipeline.run \
  --model_id "ANTHROPIC_MODEL_ID" \
  --hf_dataset_name "ORG/DATASET" \
  --hf_dataset_subset "SUBSET" \
  --hf_dataset_split "SPLIT" \
  --output_hf_dataset_name "ORG/OUTPUT_DATASET" \
  --output_hf_dataset_subset "SUBSET" \
  --parallel true \
  --max_concurrent_tasks N_CONNCURRENT
```

作为示例命令，您可以运行：

```bash
uv run python -m annotation_pipeline.run \
  --model_id "claude-sonnet-4-20250514" \
  --hf_dataset_name "obalcells/labeled-entity-facts" \
  --hf_dataset_subset "annotated_Meta-Llama-3.1-8B-Instruct" \
  --hf_dataset_split "test" \
  --output_hf_dataset_name "andyrdt/labeled-entity-facts-test" \
  --output_hf_dataset_subset "annotated_Meta-Llama-3.1-8B-Instruct" \
  --parallel true \
  --max_concurrent_tasks 10
```

### 演示界面

该演示提供了文本生成过程中幻觉检测的实时可视化。它包括：

- **后端**：`demo/modal_backend.py` - 一个带有vLLM的Modal应用，加载目标模型并应用探针头（和可选的LoRA）来计算生成过程中的标记级概率。
- **前端**：`demo/probe_interface.py` - 一个Streamlit界面，连接到Modal后端并可视化标记级置信度分数。

#### 先决条件

1. **设置Modal**：
    - 在[https://modal.com/signup](https://modal.com/signup)创建Modal账户（截至2025年8月，他们为新账户提供30美元免费额度）
    - 安装Modal：`pip install modal`
    - 运行 `modal setup` 进行身份验证

2. **环境变量**（添加到 `.env`）：

   ```bash
    HF_TOKEN=your_huggingface_token_id
   ```

3. **选择探针**：Modal后端需要您指定要加载的探针。可用的探针名称包括：

   对于Llama 3.1 8B：
    - `llama3_1_8b_lora_lambda_kl_0_05` - 带高KL正则化的LoRA探针（推荐）
    - `llama3_1_8b_linear` - 线性探针
    - `llama3_1_8b_lora_lambda_lm_0_01` - 带LM正则化的LoRA探针

   对于Llama 3.3 70B：
    - `llama3_3_70b_lora_lambda_kl_0_05` - 带高KL正则化的LoRA探针（推荐）
    - `llama3_3_70b_linear` - 线性探针
    - `llama3_3_70b_lora_lambda_lm_0_01` - 带LM正则化的LoRA探针

   **推荐**：使用 `*_lora_lambda_kl_0_05` 探针以获得最佳结果和对生成质量的最小影响。

#### 运行演示

Modal后端和Streamlit前端都必须在 `demo/` 目录内运行：

```bash
# 导航到demo目录
cd demo

# 部署Modal后端
modal deploy modal_backend.py

# 运行Streamlit前端（同样在demo/目录下）
streamlit run probe_interface.py
```

打开浏览器使用界面。该界面将连接到您部署的Modal后端，允许您输入提示、生成文本，并基于探针的置信度分数查看带有颜色编码标记的实时幻觉检测。

## 引用

```bibtex
@misc{obeso2025realtimedetectionhallucinatedentities,
      title={Real-Time Detection of Hallucinated Entities in Long-Form Generation}, 
      author={Oscar Obeso and Andy Arditi and Javier Ferrando and Joshua Freeman and Cameron Holmes and Neel Nanda},
      year={2025},
      eprint={2509.03531},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.03531}, 
}
```
