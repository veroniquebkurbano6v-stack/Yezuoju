```javascript
const api = axios.create({
  baseURL: '/api',
  timeout: 60000, // 60秒超时，因为RAG查询可能需要较长时间
  headers: {
    'Content-Type': 'application/json'
  }
})
```

## axios.create() 使用心得

`axios.create()` 是我在项目中常用的方法，它能帮我创建一个自定义的axios实例。这样做有几个好处：

1. 可以预设一些通用参数，不用每次请求都重复写
2. 项目中所有API请求都能统一管理，修改起来也方便
3. 不同模块可以使用不同的axios实例，互不干扰

### 配置参数说明

**baseURL: '/api'**
- 设置了所有请求的基础路径，这样所有请求都会自动加上/api前缀
- 比如调用 `api.get('/books')` 实际上会发送请求到 `/api/books`
- 这个设置在前后端分离的项目中特别有用，方便管理API路径

**headers: {'Content-Type': 'application/json'}**
- 设置请求头，告诉服务器请求体的数据格式是JSON
- 在发送POST或PUT请求时特别重要，确保服务器能正确解析数据
- 也可以在这里添加token等认证信息

**timeout: 60000**
- 设置请求超时时间为60秒（60000毫秒）
- 超时后会自动中断请求并抛出错误
- 在RAG查询这类可能耗时较长的场景中，适当延长超时时间很有必要



## 响应拦截器

```javascript
// 响应拦截器
api.interceptors.response.use(
  (response) => {
    return response.data
  },
  (error) => {
    if (error.response) {
      const message = error.response.data?.detail || error.response.statusText || '请求失败'
      return Promise.reject(new Error(message))
    } else if (error.request) {
      return Promise.reject(new Error('网络错误，请检查网络连接'))
    } else {
      return Promise.reject(new Error(error.message || '请求失败'))
    }
  }
)
```

响应拦截器是axios中非常强大的功能，它会在服务器返回响应后、then/catch处理前执行。这段代码实现了两个主要功能：

### 成功响应处理

```javascript
(response) => {
  return response.data
}
```

- 当服务器成功响应（状态码在2xx范围内）时，这个函数会被调用
- 它直接返回 `response.data` 而不是整个response对象
- 这样做的好处是，在使用API时可以直接获取到数据，而不需要每次都写 `.data`
- 例如：`api.get('/books').then(data => console.log(data))` 而不是 `api.get('/books').then(res => console.log(res.data))`

### 错误处理

```javascript
(error) => {
  // 错误处理逻辑
}
```

这部分处理了三种不同类型的错误情况，每种情况都有特定的处理方式：

#### 1. 服务器响应了错误状态码

```javascript
if (error.response) {
  const message = error.response.data?.detail || error.response.statusText || '请求失败'
  return Promise.reject(new Error(message))
}
```

- 当服务器返回了响应但状态码不在2xx范围内时（如404、500等），`error.response` 存在
- 错误信息获取的优先级是：
  1. `error.response.data?.detail` - 优先使用服务器返回的具体错误详情
  2. `error.response.statusText` - 如果没有detail，则使用HTTP状态文本
  3. `'请求失败'` - 如果都没有，则使用默认错误信息
- 使用可选链操作符 `?.` 安全地访问可能不存在的 `detail` 属性
- 将错误信息包装成新的Error对象并通过Promise.reject传递

#### 2. 请求已发出但未收到响应

```javascript
else if (error.request) {
  return Promise.reject(new Error('网络错误，请检查网络连接'))
}
```

- 当请求已经发出但没有收到响应时，`error.request` 存在但 `error.response` 不存在
- 这种情况通常发生在网络连接问题、服务器无响应或请求超时
- 返回一个友好的网络错误提示，建议用户检查网络连接

#### 3. 其他错误情况

```javascript
else {
  return Promise.reject(new Error(error.message || '请求失败'))
}
```

- 这种情况既没有 `error.response` 也没有 `error.request`，通常是请求配置错误或其他意外情况
- 优先使用axios提供的原始错误信息 `error.message`
- 如果没有原始错误信息，则使用默认的 `'请求失败'`

### 拦截器的实际应用

使用这个响应拦截器后，调用API的代码可以更加简洁和统一：

```javascript
// 不需要处理response.data，拦截器已经提取了数据
api.get('/books')
  .then(books => {
    // 直接使用books，而不是response.data.books
    console.log('书籍列表：', books)
  })
  .catch(err => {
    // 错误信息已经被拦截器处理过，更加友好
    console.error('错误：', err.message)
  })
```

### 错误处理最佳实践

在实际项目中，可以进一步扩展这个错误处理逻辑：

```javascript
// 可以根据状态码进行更详细的错误处理
if (error.response) {
  const status = error.response.status
  let message
  
  switch(status) {
    case 400:
      message = '请求参数错误'
      break
    case 401:
      message = '未授权，请重新登录'
      // 可以在这里触发重新登录逻辑
      break
    case 403:
      message = '拒绝访问'
      break
    case 404:
      message = '请求的资源不存在'
      break
    case 500:
      message = '服务器内部错误'
      break
    default:
      message = error.response.data?.detail || error.response.statusText || '请求失败'
  }
  
  return Promise.reject(new Error(message))
}
```

## API函数封装

项目中封装了几个常用的API函数，让调用更简单。每个API都有详细的参数说明和返回值说明：

### 1. 健康检查接口

```javascript
/**
 * 获取健康状态
 * @returns {Promise<Object>} 返回服务器健康状态信息
 */
export const getHealthAPI = () => {
  return api.get('/health')
}
```

**用途**：检查服务器是否正常运行，通常用于监控或应用启动时的健康检查。

**返回值**：
- 成功时返回服务器状态信息，可能包含服务版本、运行时间等
- 失败时抛出错误，包含错误信息

### 2. 获取书籍列表接口

```javascript
/**
 * 获取书籍列表
 * @returns {Promise<Object>} 返回按语言分类的书籍列表
 */
export const getBooksAPI = () => {
  return api.get('/books')
}
```

**用途**：获取系统中所有可用的书籍列表，通常按语言分类展示。

**返回值**：
- 成功时返回按语言分类的书籍列表，格式可能为：
  ```javascript
  {
    "zh": ["红楼梦", "西游记", ...],
    "en": ["Pride and Prejudice", ...],
    ...
  }
  ```
- 失败时抛出错误，包含错误信息

### 3. 提交查询接口

```javascript
/**
 * 提交查询
 * @param {Object} params - 查询参数
 * @param {string} params.query - 用户问题
 * @param {string|null} params.language - 语言（可选）
 * @param {string} params.mode - 模式：'library' 或 'single'
 * @param {string|null} params.book - 书籍名称（单书模式时必需）
 * @returns {Promise<Object>} 返回答案和引用
 */
export const queryAPI = (params) => {
  return api.post('/query', params)
}
```

**用途**：向RAG系统提交查询，获取基于书籍内容的回答。

**参数详解**：
- `query` (必填)：用户提出的问题或查询内容
  - 类型：string
  - 示例："红楼梦的主要人物有哪些？"
- `language` (可选)：指定查询使用的语言
  - 类型：string 或 null
  - 可选值："zh"、"en"等，不指定则使用系统默认语言
- `mode` (必填)：查询模式
  - 类型：string
  - 可选值：
    - "library"：在整个书籍库中搜索答案
    - "single"：在指定单本书籍中搜索答案
- `book` (单书模式时必填)：指定查询的书籍名称
  - 类型：string 或 null
  - 当mode为"single"时必须指定，如"红楼梦"
  - 当mode为"library"时应为null或不传

**返回值**：
- 成功时返回包含答案和引用信息的对象，格式可能为：
  ```javascript
  {
    "answer": "根据红楼梦的描述，主要人物包括贾宝玉、林黛玉、薛宝钗等...",
    "sources": [
      {
        "book": "红楼梦",
        "chapter": "第五回",
        "content": "贾宝玉...林黛玉...薛宝钗...",
        "page": 123
      },
      ...
    ]
  }
  ```
- 失败时抛出错误，包含错误信息

### 使用示例

```javascript
import { getHealthAPI, getBooksAPI, queryAPI } from './api'

// 1. 检查服务器健康状态
getHealthAPI().then(status => {
  console.log('服务器状态：', status)
}).catch(err => {
  console.error('健康检查失败：', err.message)
})

// 2. 获取书籍列表
getBooksAPI().then(books => {
  console.log('可用书籍：', books)
  // 可以在这里构建书籍选择界面
}).catch(err => {
  console.error('获取书籍列表失败：', err.message)
})

// 3. 在整个书籍库中查询
queryAPI({
  query: "中国古典文学四大名著有哪些共同特点？",
  mode: "library",
  language: "zh"
}).then(result => {
  console.log('查询结果：', result.answer)
  console.log('引用来源：', result.sources)
}).catch(err => {
  console.error('查询失败：', err.message)
})

// 4. 在单本书中查询
queryAPI({
  query: "贾宝玉的性格特点是什么？",
  mode: "single",
  book: "红楼梦",
  language: "zh"
}).then(result => {
  console.log('查询结果：', result.answer)
  console.log('引用来源：', result.sources)
}).catch(err => {
  console.error('查询失败：', err.message)
})
```

