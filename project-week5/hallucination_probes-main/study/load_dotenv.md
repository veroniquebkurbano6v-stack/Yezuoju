## 作用和用途

load_dotenv() 的主要作用是从项目根目录下的 .env 文件中读取环境变量，并将它们加载到当前的环境中，这样你就可以通过
os.getenv() 或 os.environ 来访问这些变量。

典型用途包括：

1. 管理敏感信息（如 API 密钥、数据库密码等）
2. 存储配置信息
3. 区分不同环境的配置（开发、测试、生产）

实现原理

1. 读取.env文件:该函数会查找项目根目录下的.env文件
2. 解析文件内容:它会解析.env文件中的每一行，将键值对加载到环境变量中
3. 使用示例

```
from dotenv import load_dotenv
import os

# 加载 .env 文件中的环境变量
load_dotenv()

# 获取环境变量
api_key = os.getenv('API_KEY')
database_url = os.getenv('DATABASE_URL')

```