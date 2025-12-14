## 使用Path对象相比使用普通字符串路径有以下优势：

1. 跨平台兼容性：Path对象会自动处理不同操作系统的路径分隔符（Windows用\，Unix/Linux用/）。

2. 面向对象的API：Path对象提供了许多方便的方法，如.exists()检查路径是否存在、.mkdir()创建目录等。

3. 路径操作简化：可以使用/运算符来连接路径部分，比使用os.path.join()更直观。

## 示例用法
```
# 创建路径对象
cache_dir = Path("./data/cache")

# 检查路径是否存在
if not cache_dir.exists():
    # 创建目录
    cache_dir.mkdir(parents=True, exist_ok=True)

# 构建子路径
file_path = cache_dir / "embeddings" / "model.pkl"

# 获取路径的各个部分
print(file_path.name)      # 文件名: model.pkl
print(file_path.parent)    # 父目录: ./data/cache/embeddings
print(file_path.suffix)    # 文件扩展名: .pkl

```