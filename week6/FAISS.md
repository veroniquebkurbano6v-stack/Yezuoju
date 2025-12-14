# FAISS (Facebook AI Similarity Search) 笔记

## 简介

FAISS（Facebook AI Similarity Search）是由Facebook开发的一个用于高效相似性搜索和密集向量聚类的库。它特别适用于大规模向量检索场景，是RAG（检索增强生成）系统中的关键组件。

## 核心功能

1. **向量相似性搜索**：快速找到与查询向量最相似的向量
2. **聚类**：对向量进行聚类分组
3. **索引构建**：提供多种索引结构以平衡搜索速度和内存使用
4. **GPU支持**：利用GPU加速计算

## FAISS在RAG系统中的作用

在RAG系统中，FAISS主要用于：

- **知识库索引**：将文档转换为向量后，使用FAISS构建高效索引
- **相似文档检索**：根据用户查询向量，快速检索最相关的文档片段
- **语义搜索**：实现基于语义而非关键词的搜索能力

## FAISS索引类型

### 1. Flat索引

最简单的索引类型，通过暴力搜索找到最相似的向量。

```python
import faiss
import numpy as np

# 创建随机向量数据
d = 64  # 向量维度
nb = 1000  # 数据库大小
nq = 100  # 查询数量
np.random.seed(1234)
xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')

# 创建Flat索引
index = faiss.IndexFlatL2(d)  # 使用L2距离
print(index.is_trained)  # 输出: True，Flat索引不需要训练
index.add(xb)  # 添加向量到索引
print(index.ntotal)  # 输出: 1000

# 搜索
k = 4  # 返回4个最近邻
D, I = index.search(xq, k)  # D是距离，I是索引
```

### 2. IVF（倒排文件）索引

通过聚类减少搜索空间，提高搜索速度。

```python
nlist = 100  # 聚类中心数量
quantizer = faiss.IndexFlatL2(d)  # 量化器
index = faiss.IndexIVFFlat(quantizer, d, nlist)

# 训练索引
index.train(xb)
index.add(xb)

# 搜索
index.nprobe = 10  # 搜索时检查的聚类中心数量
D, I = index.search(xq, k)
```

### 3. PQ（乘积量化）索引

通过量化减少内存使用，适合大规模数据。

```python
m = 8  # PQ子向量数量
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)  # 最后一个参数是每个子向量的编码位数

# 训练和添加
index.train(xb)
index.add(xb)

# 搜索
D, I = index.search(xq, k)
```

## FAISS高级功能

### 1. 索引合并与分割

```python
# 合并索引
index1 = faiss.IndexFlatL2(d)
index2 = faiss.IndexFlatL2(d)
index1.add(xb[:500])
index2.add(xb[500:])

index = faiss.IndexFlatL2(d)
faiss.merge_index(index, [index1, index2])

# 分割索引
index1, index2 = faiss.split_index(index, 500)
```

### 2. GPU加速

```python
# 检查GPU可用性
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)  # 0是GPU ID

# 在GPU上搜索
D, I = gpu_index.search(xq, k)

# 转回CPU索引
cpu_index = faiss.index_gpu_to_cpu(gpu_index)
```

### 3. 索引序列化

```python
# 保存索引
faiss.write_index(index, "my_index.faiss")

# 加载索引
loaded_index = faiss.read_index("my_index.faiss")
```

## 实际应用示例

以下是在RAG系统中使用FAISS的完整示例：

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. 加载预训练模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. 准备文档
documents = [
    "人工智能是计算机科学的一个分支",
    "机器学习是人工智能的子领域",
    "深度学习是机器学习的一种方法",
    "自然语言处理是AI的重要应用"
]

# 3. 将文档转换为向量
doc_embeddings = model.encode(documents)
doc_embeddings = np.array(doc_embeddings).astype('float32')

# 4. 创建FAISS索引
d = doc_embeddings.shape[1]  # 向量维度
index = faiss.IndexFlatL2(d)
index.add(doc_embeddings)

# 5. 查询
query = "什么是深度学习？"
query_embedding = model.encode([query])
query_embedding = np.array(query_embedding).astype('float32')

# 6. 搜索相似文档
k = 2  # 返回2个最相似的文档
D, I = index.search(query_embedding, k)

# 7. 输出结果
for i in range(k):
    print(f"文档: {documents[I[0][i]]}")
    print(f"距离: {D[0][i]}")
    print("---")
```

## 性能优化技巧

1. **选择合适的索引类型**：
   - 小规模数据：使用Flat索引
   - 中等规模数据：使用IVF索引
   - 大规模数据：使用IVF+PQ索引

2. **调整参数**：
   - IVF索引：增加`nlist`提高精度但增加内存
   - IVF索引：增加`nprobe`提高精度但降低速度
   - PQ索引：增加`m`提高精度但增加内存

3. **使用GPU加速**：
   - 大规模数据集建议使用GPU
   - 批量搜索比单次搜索更高效

4. **预处理数据**：
   - 归一化向量可以提高某些索引的效率
   - PCA降维可以减少内存使用和计算时间

## 常见问题与解决方案

1. **内存不足**：
   - 使用更高效的索引类型（如IVF+PQ）
   - 减少向量维度（PCA降维）
   - 使用索引分片（Sharding）

2. **搜索速度慢**：
   - 增加IVF索引的`nprobe`值
   - 使用GPU加速
   - 考虑使用近似搜索而非精确搜索

3. **精度不足**：
   - 增加IVF索引的`nlist`值
   - 使用更精确的量化方法
   - 考虑使用混合索引策略

## 在实际应用中先要准备索引文件路径和元数据文件路径
```python
# 使用 FAISS 作为向量数据库
self.index_path = self.cache_dir / "faiss_index.bin" #索引文件路径
self.metadata_path = self.cache_dir / "faiss_metadata.pkl" #元数据文件路径
```
## 初始化或加载 FAISS 索引
```python
    def _init_faiss_index(self):
        """初始化或加载 FAISS 索引"""
        if self.index_path.exists() and self.metadata_path.exists(): #检查索引文件路径和元数据文件路径是否存在
            # 加载已存在的索引
            self.faiss_index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
            print(f"已加载 FAISS 索引，包含 {len(self.metadata['ids'])} 个文档")
        else:
            # 创建新索引
            embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            self.faiss_index = faiss.IndexFlatIP(embedding_dim)  # 使用内积作为相似度度量
            self.metadata = {"ids": [], "texts": [], "sources": [], "langs": [], "chunk_ids": [], "mtimes": []}
            print("已创建新的 FAISS 索引")
```
### 方法详解
1. faiss.read_index(str(self.index_path)):从二进制文件加载向量索引,注方法只接受字符串类型,使用前用str()转换类型
2.     with open(self.metadata_path, "rb") as f:
       self.metadata = pickle.load(f) 
       从pickle文件中加载元数据，元数据包括文档ID、文本、来源、语言、块ID和修改时间等信息
3. embedding_dim = self.embedding_model.get_sentence_embedding_dimension():从方法返回该模型生成的向量维度
4. self.faiss_index = faiss.IndexFlatIP(embedding_dim)
   1. IndexFlatIP 是 FAISS 提供的最基础的索引类型之一，特点包括
      1. "Flat" 表示它执行精确搜索，不使用近似方法
      2. 直接存储所有向量，不做量化或压缩
      3. "IP" 代表 Inner Product（内积），作为相似度度量标准
   2. 在这里，我们使用内积作为相似度度量，因为内积可以很好地捕捉向量之间的相似性，适用于语义搜索任务
      1. 对于归一化向量，内积等价于余弦相似度
      2. 内积计算简单，适合大规模向量搜索
      3. 内积可以处理高维向量，适合语义搜索任务
总结:这里是根据向量模型的维度，用faiss.IndexFlatIP创建了对应维度的FAISS,并设置了内积作为相似度度量
5. FAISS 的输入格式要求
   1. FAISS 库设计为批量处理多个查询向量，因此它的搜索函数（如 search）期望输入是一个二维数组
   2. 第一维表示查询向量的数量（batch size）
   3. 第二维表示每个向量的维度

## 向量搜索
```python
distances, indices = self.faiss_index.search(query_embedding, VECTOR_TOP_K)
```
### 这行代码调用 FAISS 索引的 search 方法，执行相似度搜索并返回两个结果：
1. distances: 每个候选向量与查询向量的相似度分数
2. indices: 与每个查询向量最相似的向量的索引
### 返回值解释
1. distances: 一个二维数组，其中每个元素表示查询向量与索引中对应向量的相似度分数。在这个例子中，distances 是一个形状为 (1, VECTOR_TOP_K) 的数组，表示查询向量与索引中前 VECTOR_TOP_K 个最相似向量的相似度分数。
2. indices: 一个二维数组，其中每个元素表示查询向量与索引中对应向量的索引。在这个例子中，indices 是一个形状为 (1, VECTOR_TOP_K) 的数组，包含候选向量在FAISS索引中的位置（索引ID）,这些ID可用于从元数据中检索对应的文本和其他信息

