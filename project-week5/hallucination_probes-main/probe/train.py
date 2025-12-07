"""幻觉检测探针的训练脚本。"""

import os
import json
import atexit
from pathlib import Path
from typing import List
from dataclasses import asdict
import argparse

import torch
import wandb
from torch.utils.data import Subset
from transformers import TrainingArguments
from dotenv import load_dotenv

from utils.file_utils import save_jsonl, save_json, load_yaml
from utils.model_utils import load_model_and_tokenizer, print_trainable_parameters
from utils.probe_loader import upload_probe_to_hf

from .dataset import TokenizedProbingDataset, create_probing_dataset, tokenized_probing_collate_fn
from .config import TrainingConfig
from .value_head_probe import setup_probe
from .trainer import ProbeTrainer


def main(training_config: TrainingConfig):
    """主训练函数。"""

    # Load environment variables from .env if present
    load_dotenv()

    if training_config.upload_to_hf:
        assert os.environ.get("HF_WRITE_TOKEN", None) is not None
    # 确保环境变量HF_WRITE_TOKEN存在，否则程序会报错并终止执行。
    wandb.init(project=training_config.wandb_project, name=training_config.probe_config.probe_id)

    print("Training config:")
    for key, value in asdict(training_config).items():
        print(f"\t{key}: {value}")
    # 将数据类实例转换为字典并打印所有配置项

    # 加载模型和分词器
    print(f"正在加载模型: {training_config.probe_config.model_name}")
    model, tokenizer = load_model_and_tokenizer(
        training_config.probe_config.model_name
    )

    if hasattr(model, 'config'):
        try:
            model.config.use_cache = False
        except Exception:
            pass
    if training_config.enable_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    print(f"Setting up probe: {training_config.probe_config.probe_id}")
    model, probe = setup_probe(model, training_config.probe_config)

    print_trainable_parameters(probe)

    # 加载数据集
    print("正在加载数据集:")
    train_datasets: List[TokenizedProbingDataset] = [
        create_probing_dataset(config, tokenizer)
        for config in training_config.train_dataset_configs
    ]
    eval_datasets: List[TokenizedProbingDataset] = [
        create_probing_dataset(config, tokenizer)
        for config in training_config.eval_dataset_configs
    ]

    # 合并训练数据集
    train_dataset = train_datasets[0]
    for dataset in train_datasets[1:]:
        train_dataset += dataset

    # 如果需要，对训练数据进行随机采样以获得固定数量的样本
    if training_config.num_train_samples is not None:
        total = len(train_dataset)
        num = max(0, min(int(training_config.num_train_samples), total))  # 确保采样数量在合理范围内
        if num < total:
            g = torch.Generator()
            g.manual_seed(training_config.seed)  # 固定随机种子以保证结果可复现
            perm = torch.randperm(total, generator=g).tolist()  # 生成随机排列索引
            selected_indices = perm[:num]  # 选取前num个索引
            train_dataset = Subset(train_dataset, selected_indices)  # 创建子集，不复制数据
            print(f"使用训练数据集的子集: {num}/{total} 个样本")

    # 设置训练参数
    training_args = TrainingArguments(
        # 输出目录，用于保存模型和检查点
        output_dir=str(training_config.probe_config.probe_path),
        # 如果输出目录已存在，是否覆盖
        overwrite_output_dir=True,
        # 每个设备（GPU/CPU）上的训练批次大小
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        # 每个设备上的评估批次大小
        per_device_eval_batch_size=training_config.per_device_eval_batch_size,
        # 最大训练步数
        max_steps=training_config.max_steps,
        # 训练轮数
        num_train_epochs=training_config.num_train_epochs,
        # 每隔多少步记录一次日志
        logging_steps=training_config.logging_steps,
        # 每隔多少步进行一次评估
        eval_steps=training_config.eval_steps,
        # 是否移除数据集中未使用的列
        remove_unused_columns=False,
        # 标签名称列表，用于模型的输出
        label_names=["classification_labels", "lm_labels"],
        # 将训练日志报告到wandb平台
        report_to="wandb",
        # wandb中运行的名称
        run_name=training_config.probe_config.probe_id,
        # 评估策略：按步数评估或不评估
        eval_strategy="steps" if training_config.eval_steps else "no",
        # 是否在第一步就记录日志
        logging_first_step=True,
        # 日志记录策略：按步数记录
        logging_strategy="steps",
        # 梯度裁剪的最大范数，用于防止梯度爆炸
        max_grad_norm=training_config.max_grad_norm,
        # 梯度累积步数，用于模拟更大的批次大小
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        # 学习率
        learning_rate=training_config.learning_rate,
        # 随机种子，用于可复现性
        seed=training_config.seed,
    )

    # 为不同组件设置独立的学习率
    training_args.probe_head_lr = training_config.probe_head_lr
    training_args.lora_lr = training_config.lora_lr

    # 禁用检查点保存功能
    # (训练过程中保存检查点可能导致一些奇怪的bug)
    training_args.set_save(strategy="no")

    trainer = ProbeTrainer(
        probe=probe,
        eval_datasets=eval_datasets,
        cfg=training_config,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,  # this is a dummy argument is for the HF base Trainer class
        data_collator=tokenized_probing_collate_fn,
        eval_steps=training_config.eval_steps,
        tokenizer=tokenizer,
    )

    def save_model_callback():
        """保存探针权重、分词器和训练配置到磁盘。"""
        probe.save(training_config.probe_config.probe_path)
        tokenizer.save_pretrained(training_config.probe_config.probe_path)
        save_json(
            training_config,
            training_config.probe_config.probe_path / "training_config.json"
        )

    # 注册保存回调函数，用于处理意外退出情况
    atexit.register(save_model_callback)

    print("开始训练...")
    trainer.train()

    # 保存模型
    print(f"正在保存模型到 {training_config.probe_config.probe_path}")
    save_model_callback()

    # 最终评估
    eval_metrics = trainer.evaluate(
        save_roc_curves=training_config.save_roc_curves,
        dump_raw_eval_results=training_config.dump_raw_eval_results,
        verbose=True,
    )

    if training_config.save_evaluation_metrics:
        save_json(
            eval_metrics,
            training_config.probe_config.probe_path / "evaluation_results.json"
        )

    wandb.finish()

    if training_config.upload_to_hf:
        print(f"正在上传探针到HuggingFace Hub...")
        upload_probe_to_hf(
            repo_id=training_config.probe_config.hf_repo_id,
            probe_id=training_config.probe_config.probe_id,
            token=os.environ.get("HF_WRITE_TOKEN"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练一个幻觉检测探针")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="训练配置文件的路径"
    )

    args = parser.parse_args()

    # 从YAML文件加载配置
    training_config = TrainingConfig(**load_yaml(args.config))

    main(training_config)
