"""用于探针训练的带token级别标签的分词后数据集类。"""

import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import datasets
from jaxtyping import Float, Int
from termcolor import colored
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.tokenization import find_assistant_tokens_slice, find_string_in_tokens, slice_to_list
from .types import AnnotatedSpan, ProbingItem
from .dataset_converters import get_prepare_function


@dataclass
class TokenizedProbingDatasetConfig:
    """用于在token级别对探针数据集进行分词和标注的配置类。"""

    dataset_id: str  # 数据集ID标识符
    hf_repo: str  # HuggingFace仓库路径
    subset: Optional[str] = None  # 数据集子集名称
    split: str = "train"  # 数据集划分：train/validation/test
    max_length: int = 2048  # 最大序列长度
    ignore_buffer: int = 0  # 标注片段周围要忽略的token数量（缓冲区）
    default_ignore: bool = False  # 如果为True，则忽略不在任何标注片段中的token
    last_span_token: bool = False  # 如果为True，只标注每个片段的最后一个token
    pos_weight: float = 1.0  # 正样本（幻觉）token的权重
    neg_weight: float = 1.0  # 负样本（有支持的事实）token的权重
    shuffle: bool = True  # 是否打乱数据顺序
    seed: int = 42  # 随机种子
    process_on_the_fly: bool = False  # 是否动态处理数据（节省内存）
    max_num_samples: Optional[int] = None  # 最大样本数量限制


class TokenizedProbingDataset(Dataset):
    """用于探针模型激活的数据集，带有标注的文本片段。"""

    def __init__(
            self,
            items: List[ProbingItem],  # 原始探针数据项列表
            config: TokenizedProbingDatasetConfig,  # 数据集配置
            tokenizer: AutoTokenizer,  # 分词器
    ):
        self.config = config  # 保存配置
        self.tokenizer = tokenizer  # 保存分词器
        self.items = deepcopy(items)  # 深拷贝原始数据项
        self.processed_items = [None] * len(items)  # 初始化处理后的数据项列表
        self.debug_mode = False  # 调试模式标志
        self.print_first_example = False  # 是否打印第一个示例的标志

        self._num_skipped_spans: int = 0  # 跳过的标注片段计数
        self._num_added_spans: int = 0  # 成功添加的标注片段计数

        if self.config.shuffle:  # 如果需要打乱数据
            self._shuffle_items()  # 打乱数据项顺序

        # 如果指定了最大样本数，进行限制（在打乱之后进行）
        if self.config.max_num_samples:
            self.items = self.items[:self.config.max_num_samples]
            self.processed_items = self.processed_items[:self.config.max_num_samples]

        # 如果不是动态处理模式，预处理所有数据项
        if not self.config.process_on_the_fly:
            self._process_items()  # 预处理所有数据

    def _process_items(self):
        """预处理数据集中的所有数据项。"""
        # 使用进度条遍历所有数据项
        for i, item in tqdm(enumerate(self.items), desc=f"处理数据项 ({self.config.dataset_id})",
                            total=len(self.items)):
            # 如果是第一个示例且需要打印，则开启调试模式
            if i == 0 and self.print_first_example:
                self.debug_mode = True
            else:
                self.debug_mode = False

            # 处理单个数据项
            processed_item = self._process_item(item)
            if processed_item:  # 如果处理成功
                self.processed_items[i] = processed_item  # 保存处理后的数据

        # 打印数据集统计信息
        print(f"数据集 {self.config.dataset_id} 统计:")
        print(f"\t- 成功添加的标注片段数量: {self._num_added_spans}")
        print(f"\t- 跳过的标注片段数量: {self._num_skipped_spans} / {self._num_added_spans + self._num_skipped_spans}")
        print(f"\t- 数据项总数: {len(self.items)}")

    def _process_item(self, item: ProbingItem) -> Dict:
        """处理单个示例，将其转换为分词后的格式和标签。"""
        # 构建对话格式
        conversation = [
            {'role': 'user', 'content': item.prompt},  # 用户输入
            {'role': 'assistant', 'content': item.completion}  # 助手回复
        ]
        # 应用聊天模板，获取完整文本
        full_text = self.tokenizer.apply_chat_template(conversation, tokenize=False)

        # 如果分词器有BOS token且出现在文本中，移除它
        if self.tokenizer.bos_token and self.tokenizer.bos_token in full_text:
            full_text = full_text.replace(self.tokenizer.bos_token, '')

        # 对文本进行分词
        encoding = self.tokenizer(
            full_text,
            truncation=True,  # 启用截断
            max_length=self.config.max_length,  # 最大长度限制
            padding='max_length',  # 填充到最大长度
            return_tensors='pt',  # 返回PyTorch张量
            padding_side='right'  # 右侧填充
        )

        # 提取输入ID和注意力掩码
        input_ids: Int[Tensor, "seq_len"] = encoding["input_ids"][0]
        attention_mask: Int[Tensor, "seq_len"] = encoding["attention_mask"][0]

        # 计算位置标签（分类标签、权重、正负标注片段）
        labels, weights, pos_spans, neg_spans = self._compute_positional_labels(
            input_ids=input_ids,
            item=item  # 原始数据项
        )

        # 解码输入ID为字符串，用于后续处理
        input_str: str = self.tokenizer.decode(input_ids)
        # 找到助手回复的token位置范围
        assistant_tokens_slice = find_assistant_tokens_slice(input_ids, input_str, self.tokenizer)
        completion_start_idx = assistant_tokens_slice.stop  # 助手回复的起始索引

        # 创建语言模型标签（用于语言建模任务）
        lm_labels = input_ids.clone()  # 克隆输入ID
        lm_labels[:completion_start_idx] = -100  # 忽略prompt中的所有token
        lm_labels[attention_mask == 0] = -100  # 忽略填充token

        # 返回处理后的数据字典
        return {
            "input_ids": input_ids,  # 输入token ID序列，Int[Tensor, "seq_len"]
            "attention_mask": attention_mask,  # 注意力掩码，Int[Tensor, "seq_len"]
            "classification_labels": labels,  # 分类标签，Float[Tensor, "seq_len"]
            "classification_weights": weights,  # 分类权重，Float[Tensor, "seq_len"]
            "pos_spans": pos_spans,  # 正样本（幻觉）标注片段列表，List[List[int]]
            "neg_spans": neg_spans,  # 负样本（有支持的事实）标注片段列表，List[List[int]]
            "lm_labels": lm_labels,  # 语言模型标签，Int[Tensor, "seq_len"]
        }

    def print_token_labels(
            self,
            input_ids: torch.Tensor,  # 输入token ID
            positive_indices: List[int],  # 正样本（幻觉）token索引列表
            negative_indices: List[int],  # 负样本（有支持的事实）token索引列表
            ignore_indices: List[int],  # 忽略的token索引列表
            spans: List[AnnotatedSpan]  # 原始标注片段列表
    ):
        """调试方法：打印token如何被标注。"""

        # 解码所有token
        tokens = [self.tokenizer.decode(tok) for tok in input_ids]

        # 打印统计信息
        print(f"================================================")
        print(f"标注片段总数: {len(spans)}")
        print(f"非事实（幻觉）标注片段数量: {len([f for f in spans if f.label == 1.0])}")
        print(f"不适用（N/A）标注片段数量: {len([f for f in spans if f.label == -100])}")
        print(f"事实标注片段数量: {len([f for f in spans if f.label == 0.0])}")
        print(f"图例: 红色 - 正样本（幻觉）, 绿色 - 负样本（有支持的事实）, 蓝色 - 忽略")

        # 遍历所有token，根据标注类型打印不同颜色
        for i, token in enumerate(tokens):
            if token == self.tokenizer.eos_token:  # 如果是EOS token，跳过
                continue

            # 根据token的标注类型选择颜色
            if i in positive_indices:
                print(colored(token, 'red'), end='')  # 红色：正样本（幻觉）
            elif i in negative_indices:
                print(colored(token, 'green'), end='')  # 绿色：负样本（有支持的事实）
            elif i in ignore_indices:
                print(colored(token, 'blue'), end='')  # 蓝色：忽略
            else:
                print(token, end='')  # 默认颜色

        print(f"================================================")

    def _compute_positional_labels(
            self,
            input_ids: torch.Tensor,  # 输入token ID
            item: ProbingItem  # 探针数据项
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[int]], List[List[int]]]:
        """
        基于标注片段计算token序列的位置标签。

        对于每个标注片段：
        - 如果是幻觉（标签1.0）：将片段内的token设置为1.0，并将ignore_buffer内的附近token设置为-100.0
        - 如果是有支持的事实（标签0.0）：将片段内的token设置为0.0，并将ignore_buffer内的附近token设置为-100.0
        - 如果是未标注/未确定（标签-100.0）：将片段内的token设置为-100.0

        参数:
            input_ids: 输入token ID
            item: 包含标注片段及其标签的ProbingItem

        返回:
            元组 (labels, weights, pos_spans, neg_spans)
        """
        # 解码输入ID为字符串
        input_str: str = self.tokenizer.decode(input_ids)
        completion: str = item.completion  # 助手回复内容

        positive_indices: List[int] = []  # 幻觉片段的token索引
        negative_indices: List[int] = []  # 有支持的事实的token索引
        ignore_indices: List[int] = []  # 训练中要忽略的token索引

        positive_spans: List[List[int]] = []  # 正样本（幻觉）标注片段范围
        negative_spans: List[List[int]] = []  # 负样本（有支持的事实）标注片段范围

        def get_nearby_indices(span_indices: List[int]) -> List[int]:
            """获取标注片段附近要忽略的token索引。"""
            # 左侧窗口：从span开始前ignore_buffer个token到span开始前
            left_window = list(range(max(0, span_indices[0] - self.config.ignore_buffer), span_indices[0]))
            # 右侧窗口：从span结束后到span结束后ignore_buffer个token
            right_window = list(
                range(span_indices[-1] + 1, min(len(input_ids), span_indices[-1] + 1 + self.config.ignore_buffer)))
            return left_window + right_window  # 合并左右窗口

        # 找到助手token的位置范围，知道从哪里开始寻找标注片段
        assistant_tokens_slice = find_assistant_tokens_slice(
            input_ids,
            input_str,
            self.tokenizer
        )
        completion_start_idx = assistant_tokens_slice.stop  # 助手回复开始位置
        cur_idx = assistant_tokens_slice.stop  # 当前搜索位置

        # 按照在文本中的索引对标注片段进行排序
        spans = sorted(item.spans, key=lambda x: x.index)

        # 遍历所有标注片段
        for span in spans:
            # 检查标注片段是否在输入字符串中
            if span.span not in input_str:
                self._num_skipped_spans += 1  # 统计跳过的片段
                continue  # 跳过不在字符串中的片段

            try:
                # 首先尝试在助手token之后查找标注片段
                positions_slice = find_string_in_tokens(span.span, input_ids[cur_idx:], self.tokenizer)
                positions_slice = slice(positions_slice.start + cur_idx, positions_slice.stop + cur_idx)
            except (AssertionError, ValueError):
                try:
                    # 如果没找到，尝试在整个input_ids中查找
                    print(
                        f"在input_ids[cur_idx:]中查找标注片段 {repr(span.span)} 失败，重新在整个input_ids中搜索: {repr(self.tokenizer.decode(input_ids[cur_idx:]))[:50]}...")
                    positions_slice = find_string_in_tokens(span.span, input_ids, self.tokenizer)
                except (AssertionError, ValueError) as e:
                    print(f"标注片段 {repr(span.span)} 在input_ids中未找到，跳过实体")
                    self._num_skipped_spans += 1  # 统计跳过的片段
                    continue

            if positions_slice is None:  # 如果仍未找到，跳过
                continue

            # 将slice转换为索引列表
            span_indices = slice_to_list(positions_slice, len(input_ids))
            if not span_indices:  # 如果索引列表为空，跳过
                continue

            cur_idx = positions_slice.start  # 更新当前搜索位置

            # 如果last_span_token为True，只使用片段的最后一个token
            if self.config.last_span_token:
                span_indices = [span_indices[-1]]  # 只保留最后一个token索引

            # 获取该片段附近要忽略的token索引
            nearby_indices = get_nearby_indices(span_indices)

            # 根据片段标签进行处理
            if span.label == 1.0:  # 幻觉
                positive_indices.extend(span_indices)  # 添加到正样本索引
                ignore_indices.extend(nearby_indices)  # 添加到忽略索引
                positive_spans.append([span_indices[0], span_indices[-1]])  # 记录片段范围
            elif span.label == 0.0:  # 有支持的事实
                negative_indices.extend(span_indices)  # 添加到负样本索引
                negative_spans.append([span_indices[0], span_indices[-1]])  # 记录片段范围
            else:  # -100.0（忽略）
                ignore_indices.extend(span_indices)  # 直接添加到忽略索引

            self._num_added_spans += 1  # 统计成功添加的片段

        # 去除重复索引并排序
        positive_indices = sorted(list(set(positive_indices)))  # 正样本索引去重排序
        negative_indices = sorted(list(set(negative_indices) - set(positive_indices)))  # 负样本索引去重排序，排除正样本索引
        ignore_indices = sorted(
            list(set(ignore_indices) - set(positive_indices) - set(negative_indices)))  # 忽略索引去重排序，排除正负样本索引

        # 初始化标签张量
        default_label = -100.0 if self.config.default_ignore else 0.0  # 默认标签
        labels = torch.full((len(input_ids),), default_label, dtype=torch.float32)  # 创建标签张量

        # 设置标签
        labels[input_ids == self.tokenizer.pad_token_id] = -100.0  # 填充token设置为忽略
        labels[:completion_start_idx] = -100.0  # prompt部分设置为忽略
        labels[ignore_indices] = -100.0  # 忽略索引的token设置为忽略
        labels[positive_indices] = 1.0  # 正样本token设置为1.0
        labels[negative_indices] = 0.0  # 负样本token设置为0.0

        # 初始化权重张量
        weights = torch.full((len(input_ids),), 1.0, dtype=torch.float32)  # 创建权重张量
        weights[ignore_indices] = 0.0  # 忽略索引的token权重为0.0
        weights[positive_indices] = self.config.pos_weight  # 正样本token使用正样本权重
        weights[negative_indices] = self.config.neg_weight  # 负样本token使用负样本权重

        # 如果处于调试模式，打印token标注情况
        if self.debug_mode:
            self.print_token_labels(input_ids, positive_indices, negative_indices, ignore_indices, spans)

        return labels, weights, positive_spans, negative_spans

    def _shuffle_items(self):
        """使用配置的随机种子打乱数据项顺序。"""
        random.seed(self.config.seed)  # 设置随机种子
        random.shuffle(self.items)  # 打乱原始数据项
        random.seed(self.config.seed)  # 重新设置随机种子（保持一致）
        random.shuffle(self.processed_items)  # 打乱处理后的数据项

    def __len__(self):
        """返回数据集长度。"""
        return len(self.items)  # 返回数据项数量

    def __getitem__(self, idx):
        """获取指定索引的数据项。"""
        # 如果是动态处理模式且该数据项尚未处理，先处理它
        if self.config.process_on_the_fly and self.processed_items[idx] is None:
            self.processed_items[idx] = self._process_item(self.items[idx])

        return self.processed_items[idx]  # 返回处理后的数据项

    def __add__(self, other):
        """
        拼接两个TokenizedProbingDataset实例。

        参数:
            other: 要拼接的另一个TokenizedProbingDataset实例

        返回:
            TokenizedProbingDataset: 包含两个数据集数据项的新数据集
        """
        if not isinstance(other, TokenizedProbingDataset):  # 类型检查
            raise TypeError(f"只能与另一个TokenizedProbingDataset拼接，得到的是 {type(other)}")

        if self.config.max_length != other.config.max_length:  # 检查最大长度是否一致
            raise ValueError("无法拼接不同token长度的数据集")

        if self.config.shuffle != other.config.shuffle:  # 检查打乱设置是否一致
            raise ValueError("如果一个数据集打乱而另一个不打乱，无法拼接数据集")

        # 创建包含两个数据集数据项的新列表
        combined_items = self.items + other.items
        combined_processed_items = self.processed_items + other.processed_items

        # 使用第一个数据集的配置创建新数据集
        new_dataset = TokenizedProbingDataset(
            items=[],  # 我们不希望重新计算所有内容
            tokenizer=self.tokenizer,
            config=self.config,
        )

        # 设置新数据集的数据项
        new_dataset.items = combined_items
        new_dataset.processed_items = combined_processed_items

        # 如果需要打乱，打乱新数据集
        if self.config.shuffle:
            new_dataset._shuffle_items()

        return new_dataset  # 返回新数据集

    def __radd__(self, other):
        """反向加法操作，调用__add__方法。"""
        return self.__add__(other)


def tokenized_probing_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    DataLoader的批处理函数，处理变长分词后的序列。

    参数:
        batch: 分词后数据集项的列表

    返回:
        包含填充后序列的批处理字典
    """
    # 查找批次中的最大长度
    max_len = max(len(item["input_ids"]) for item in batch)

    # 初始化批处理张量
    batch_size = len(batch)  # 批次大小
    input_ids = torch.full((batch_size, max_len), 0, dtype=torch.long)  # 输入ID张量
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)  # 注意力掩码张量
    classification_labels = torch.full((batch_size, max_len), -100.0, dtype=torch.float32)  # 分类标签张量
    classification_weights = torch.zeros((batch_size, max_len), dtype=torch.float32)  # 分类权重张量
    lm_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)  # 语言模型标签张量

    # 标注片段列表
    pos_spans = []  # 正样本片段列表
    neg_spans = []  # 负样本片段列表

    # 填充批次数据
    for i, item in enumerate(batch):
        seq_len = len(item["input_ids"])  # 当前序列长度
        input_ids[i, :seq_len] = item["input_ids"]  # 填充输入ID
        attention_mask[i, :seq_len] = item["attention_mask"]  # 填充注意力掩码
        classification_labels[i, :seq_len] = item["classification_labels"]  # 填充分类标签
        classification_weights[i, :seq_len] = item["classification_weights"]  # 填充分类权重
        lm_labels[i, :seq_len] = item["lm_labels"]  # 填充语言模型标签
        pos_spans.append(item["pos_spans"])  # 添加正样本片段
        neg_spans.append(item["neg_spans"])  # 添加负样本片段

    # 返回批处理字典
    return {
        "input_ids": input_ids,  # 填充后的输入ID
        "attention_mask": attention_mask,  # 填充后的注意力掩码
        "classification_labels": classification_labels,  # 填充后的分类标签
        "classification_weights": classification_weights,  # 填充后的分类权重
        "lm_labels": lm_labels,  # 填充后的语言模型标签
        "pos_spans": pos_spans,  # 正样本片段列表
        "neg_spans": neg_spans,  # 负样本片段列表
    }


def create_probing_dataset(
        cfg: TokenizedProbingDatasetConfig,  # 数据集配置
        tokenizer: AutoTokenizer  # 分词器
) -> TokenizedProbingDataset:
    """
    从配置创建探针数据集。

    这会从HuggingFace加载数据集，并使用适当的数据集特定准备函数进行处理。
    """
    # 懒加载以避免循环依赖

    # 从HuggingFace加载数据集
    if cfg.subset:  # 如果有子集名称
        raw_hf_dataset = datasets.load_dataset(cfg.hf_repo, cfg.subset, split=cfg.split)
    else:  # 如果没有子集名称
        raw_hf_dataset = datasets.load_dataset(cfg.hf_repo, split=cfg.split)

    # 处理最大样本数量限制
    if cfg.max_num_samples is not None:
        if cfg.shuffle:  # 如果需要打乱
            print(f"打乱数据集并截断到 {cfg.max_num_samples} / {len(raw_hf_dataset)} 个样本")
            assert cfg.seed is not None, "如果shuffle为True，必须提供seed"
            raw_hf_dataset = raw_hf_dataset.shuffle(seed=cfg.seed)  # 打乱数据集
        else:  # 如果不需要打乱
            print(f"截断数据集到前 {cfg.max_num_samples} / {len(raw_hf_dataset)} 个样本")
        # 选择指定数量的样本
        raw_hf_dataset = raw_hf_dataset.select(range(min(cfg.max_num_samples, len(raw_hf_dataset))))

    print(f"加载数据集: {cfg.hf_repo} | {cfg.subset} | {cfg.split}")  # 打印加载信息

    # 获取适当的准备函数
    prepare_function = get_prepare_function(cfg.hf_repo, cfg.subset)

    # 将HF数据集转换为探针数据项列表
    probing_items: List[ProbingItem] = prepare_function(raw_hf_dataset)

    # 创建分词后的探针数据集
    tokenized_probing_dataset = TokenizedProbingDataset(
        items=probing_items,  # 探针数据项
        config=cfg,  # 配置
        tokenizer=tokenizer  # 分词器
    )

    return tokenized_probing_dataset  # 返回创建的数据集
