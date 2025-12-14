import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 60000, // 60秒超时，因为RAG查询可能需要较长时间
  headers: {
    'Content-Type': 'application/json'
  }
})

// 请求拦截器
api.interceptors.request.use(
  (config) => {
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

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

/**
 * 获取健康状态
 */
export const getHealthAPI = () => {
  return api.get('/health')
}

/**
 * 获取书籍列表
 * @returns {Promise<Object>} 返回按语言分类的书籍列表
 */
export const getBooksAPI = () => {
  return api.get('/books')
}

/**
 * 提交查询
 * @param {Object} params - 查询参数
 * @param {string} params.query - 用户问题
 * @param {string|null} params.language - 语言（可选）
 * @param {string} params.mode - 模式：'library' 或 'single'
 * @param {string|null} params.book - 书籍名称（单书模式时必需）
 * @param {boolean} params.refresh_index - 是否刷新索引（可选）
 * @returns {Promise<Object>} 返回答案和引用
 */
export const queryAPI = (params) => {
  return api.post('/query', params)
}

/**
 * 更新知识库索引
 * @param {Object} params - 索引更新参数
 * @param {string|null} params.language - 语言（可选）
 * @param {string} params.mode - 模式：'library' 或 'single'
 * @param {string|null} params.book - 书籍名称（单书模式时必需）
 * @returns {Promise<Object>} 返回更新结果
 */
export const refreshIndexAPI = (params) => {
  return api.post('/refresh_index', params)
}

export default api

