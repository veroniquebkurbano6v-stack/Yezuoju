## 修复计划

### 问题分析
错误信息：`name 'query_request_json' is not defined`

在 `deepseek_agent.py` 中，第133行和第136行使用了 `query_request_json` 变量，但该变量在之前的修改中被误删了定义。

### 修复方案

**文件**: `d:\PyCharm 2025.1\project\StoryRag v2.0\src\deepseek_agent.py`

**修改位置**: 第130行（在 logger.info 之后，try 块之前）

**修改内容**:
添加 `query_request_json` 变量定义：
```python
import json
query_request_json = json.dumps({"query": query, "filter_pdf": filter_pdf}, ensure_ascii=False)
```

### 修改后代码结构
```python
logger.info(f"[DeepSeekAgent] 执行检索: query='{query}', filter_pdf={filter_pdf}")

import json
query_request_json = json.dumps({"query": query, "filter_pdf": filter_pdf}, ensure_ascii=False)

# 尝试使用 invoke 方法调用工具，如果失败则直接调用函数
try:
    tool_result = self.smart_retrieval.invoke({"query_request_json": query_request_json, "top_k": 60})
except Exception:
    tool_result = self.smart_retrieval(query_request_json, top_k=60)
```

### 预期效果
修复后，`query_request_json` 变量将被正确定义，工具调用将正常工作。