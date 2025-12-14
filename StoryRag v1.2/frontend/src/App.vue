<template>
  <div class="app-container">
    <el-container class="main-container">
      <!-- 头部 -->
      <el-header class="app-header">
        <div class="header-content">
          <div class="logo-section">
            <el-icon class="logo-icon" :size="32"><Reading /></el-icon>
            <h1 class="app-title">StoryRag</h1>
            <el-tag type="info" size="small" class="version-tag">v1.1</el-tag>
          </div>
          <div class="header-subtitle">智能童话问答系统 - 支持中文、英文、日文</div>
        </div>
      </el-header>

      <!-- 主内容区 -->
      <el-main class="app-main">
        <div class="content-wrapper">
          <!-- 查询卡片 -->
          <el-card class="query-card" shadow="hover">
            <template #header>
              <div class="card-header">
                <el-icon><ChatLineRound /></el-icon>
                <span>问题查询</span>
              </div>
            </template>

            <!-- 语言选择 -->
            <el-form :model="queryForm" label-width="100px" class="query-form">
              <el-form-item label="语言选择">
                <el-select
                  v-model="queryForm.language"
                  placeholder="自动检测语言"
                  clearable
                  style="width: 200px"
                  @change="handleLanguageChange"
                >
                  <el-option label="自动检测" value="" />
                  <el-option label="中文 (Chinese)" value="Chinese" />
                  <el-option label="英文 (English)" value="English" />
                  <el-option label="日文 (Japanse)" value="Japanse" />
                </el-select>
                <el-tag v-if="detectedLanguage" type="success" class="detected-tag">
                  <el-icon><Check /></el-icon>
                  检测到: {{ detectedLanguage }}
                </el-tag>
              </el-form-item>

              <!-- 查询模式 -->
              <el-form-item label="查询模式">
                <el-radio-group v-model="queryForm.mode" @change="handleModeChange">
                  <el-radio-button label="library">
                    <el-icon><Collection /></el-icon>
                    全库查询
                  </el-radio-button>
                  <el-radio-button label="single">
                    <el-icon><Document /></el-icon>
                    单书查询
                  </el-radio-button>
                </el-radio-group>
                <el-select
                  v-if="queryForm.mode === 'single'"
                  v-model="queryForm.book"
                  placeholder="请选择书籍"
                  :disabled="!queryForm.language || bookList.length === 0"
                  style="width: 300px; margin-left: 16px"
                  filterable
                >
                  <el-option
                    v-for="book in bookList"
                    :key="book"
                    :label="book"
                    :value="book"
                  />
                </el-select>
              </el-form-item>

              <!-- 问题输入 -->
              <el-form-item label="您的问题">
                <el-input
                  v-model="queryForm.query"
                  type="textarea"
                  :rows="4"
                  placeholder="请输入您想了解的问题，例如：这个故事的主人公是谁？"
                  :disabled="loading"
                  @keydown.ctrl.enter="handleSubmit"
                />
              </el-form-item>

              <!-- 提交按钮 -->
              <el-form-item>
                <el-button
                  type="primary"
                  :loading="loading"
                  :disabled="!queryForm.query.trim()"
                  size="large"
                  @click="handleSubmit"
                >
                  <el-icon v-if="!loading"><Search /></el-icon>
                  {{ loading ? '查询中...' : '提交查询' }}
                </el-button>
                <el-button
                  v-if="answer"
                  type="info"
                  size="large"
                  @click="handleClear"
                >
                  <el-icon><RefreshLeft /></el-icon>
                  清空结果
                </el-button>
                <el-button
                  type="warning"
                  :loading="refreshing"
                  size="large"
                  @click="handleRefreshIndex"
                >
                  <el-icon v-if="!refreshing"><Refresh /></el-icon>
                  {{ refreshing ? '更新中...' : '更新知识库' }}
                </el-button>
              </el-form-item>
            </el-form>
          </el-card>

          <!-- 答案卡片 -->
          <el-card
            v-if="answer || loading"
            class="answer-card"
            shadow="hover"
            :class="{ 'loading-card': loading }"
          >
            <template #header>
              <div class="card-header">
                <el-icon><ChatDotRound /></el-icon>
                <span>模型回复</span>
                <el-tag v-if="answer" type="success" size="small" class="status-tag">
                  <el-icon><CircleCheck /></el-icon>
                  已完成
                </el-tag>
              </div>
            </template>

            <div v-if="loading" class="loading-container">
              <el-skeleton :rows="5" animated />
            </div>
            <div v-else-if="answer" class="answer-content">
              <div class="answer-text" v-html="formatAnswer(answer)"></div>
            </div>
          </el-card>

          <!-- 引用片段卡片 -->
          <el-card
            v-if="citations.length > 0"
            class="citations-card"
            shadow="hover"
          >
            <template #header>
              <div class="card-header">
                <el-icon><DocumentCopy /></el-icon>
                <span>引用片段</span>
                <el-tag type="info" size="small" class="count-tag">
                  {{ citations.length }} 条
                </el-tag>
              </div>
            </template>

            <div class="citations-list">
              <el-card
                v-for="(citation, index) in citations"
                :key="index"
                class="citation-item"
                shadow="never"
              >
                <div class="citation-header">
                  <el-tag type="primary" size="small" class="citation-tag">
                    <el-icon><Link /></el-icon>
                    {{ citation.citation }}
                  </el-tag>
                </div>
                <div class="citation-text">{{ citation.text }}</div>
              </el-card>
            </div>
          </el-card>
        </div>
      </el-main>

      <!-- 底部 -->
      <el-footer class="app-footer">
        <div class="footer-content">
          <span>© 2024 StoryRag - Powered by DeepSeek & Vue 3</span>
        </div>
      </el-footer>
    </el-container>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import {
  Reading,
  ChatLineRound,
  ChatDotRound,
  DocumentCopy,
  Search,
  RefreshLeft,
  Refresh,
  Check,
  Collection,
  Document,
  CircleCheck,
  Link
} from '@element-plus/icons-vue'
import { queryAPI, getBooksAPI, refreshIndexAPI } from './api'

const queryForm = reactive({
  query: '',
  language: '',
  mode: 'library',
  book: null
})

const loading = ref(false)
const answer = ref('')
const citations = ref([])
const detectedLanguage = ref('')
const booksMap = ref({})
const bookList = ref([])
const refreshing = ref(false)

// 加载书籍列表
const loadBooks = async () => {
  try {
    booksMap.value = await getBooksAPI()
    updateBookList()
  } catch (error) {
    console.error('加载书籍列表失败:', error)
    ElMessage.error('加载书籍列表失败')
  }
}

// 更新书籍列表
const updateBookList = () => {
  if (queryForm.language && booksMap.value[queryForm.language]) {
    bookList.value = booksMap.value[queryForm.language]
  } else {
    bookList.value = []
  }
  if (queryForm.mode === 'single' && bookList.value.length === 0 && queryForm.language) {
    ElMessage.warning('该语言暂无可用书籍')
  }
}

// 语言改变处理
const handleLanguageChange = () => {
  detectedLanguage.value = ''
  updateBookList()
  if (queryForm.mode === 'single') {
    queryForm.book = null
  }
}

// 模式改变处理
const handleModeChange = () => {
  queryForm.book = null
  if (queryForm.mode === 'single' && !queryForm.language) {
    ElMessage.warning('单书模式请先选择语言')
  }
}

// 提交查询
const handleSubmit = async () => {
  if (!queryForm.query.trim()) {
    ElMessage.warning('请输入问题')
    return
  }

  if (queryForm.mode === 'single' && !queryForm.language) {
    ElMessage.warning('单书模式请先选择语言')
    return
  }

  if (queryForm.mode === 'single' && !queryForm.book) {
    ElMessage.warning('请选择要查询的书籍')
    return
  }

  loading.value = true
  answer.value = ''
  citations.value = []
  detectedLanguage.value = ''

  try {
    const params = {
      query: queryForm.query.trim(),
      language: queryForm.language || null,
      mode: queryForm.mode,
      book: queryForm.book || null
    }

    const data = await queryAPI(params)
    answer.value = data.answer || ''
    citations.value = data.citations || []
    detectedLanguage.value = data.detected_language || ''

    if (data.detected_language && !queryForm.language) {
      ElMessage.success(`已自动检测语言: ${data.detected_language}`)
    }

    ElMessage.success('查询完成')
  } catch (error) {
    ElMessage.error(error.message || '查询失败，请稍后重试')
    answer.value = `错误: ${error.message || '查询失败'}`
  } finally {
    loading.value = false
  }
}

// 清空结果
const handleClear = () => {
  answer.value = ''
  citations.value = []
  detectedLanguage.value = ''
  queryForm.query = ''
}

// 更新知识库索引
const handleRefreshIndex = async () => {
  if (refreshing.value) return
  
  try {
    refreshing.value = true
    
    const params = {
      language: queryForm.language || null,
      mode: queryForm.mode,
      book: queryForm.book || null
    }
    
    const result = await refreshIndexAPI(params)
    ElMessage.success(result.message || '索引更新成功')
    
    // 重新加载书籍列表
    await loadBooks()
  } catch (error) {
    ElMessage.error(error.message || '索引更新失败，请稍后重试')
  } finally {
    refreshing.value = false
  }
}

// 格式化答案（支持Markdown样式）
const formatAnswer = (text) => {
  if (!text) return ''
  // 简单的Markdown转HTML
  return text
    .replace(/\n/g, '<br>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/`(.*?)`/g, '<code>$1</code>')
}

onMounted(() => {
  loadBooks()
})
</script>

<style scoped>
.app-container {
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.main-container {
  max-width: 1200px;
  margin: 0 auto;
  min-height: 100vh;
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
}

.app-header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 24px 32px;
  height: auto !important;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
}

.header-content {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.logo-section {
  display: flex;
  align-items: center;
  gap: 12px;
}

.logo-icon {
  animation: float 3s ease-in-out infinite;
}

@keyframes float {
  0%, 100% {
    transform: translateY(0px);
  }
  50% {
    transform: translateY(-10px);
  }
}

.app-title {
  font-size: 32px;
  font-weight: 700;
  margin: 0;
  text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
}

.version-tag {
  background: rgba(255, 255, 255, 0.2);
  border: 1px solid rgba(255, 255, 255, 0.3);
  color: white;
}

.header-subtitle {
  font-size: 14px;
  opacity: 0.9;
  margin-left: 44px;
}

.app-main {
  padding: 32px;
  background: #f5f7fa;
}

.content-wrapper {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.query-card,
.answer-card,
.citations-card {
  border-radius: 12px;
  transition: all 0.3s ease;
}

.query-card:hover,
.answer-card:hover,
.citations-card:hover {
  transform: translateY(-2px);
}

.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
  font-size: 16px;
}

.query-form {
  margin-top: 16px;
}

.detected-tag {
  margin-left: 12px;
}

.status-tag,
.count-tag {
  margin-left: auto;
}

.answer-content {
  padding: 16px 0;
}

.answer-text {
  line-height: 1.8;
  font-size: 15px;
  color: #303133;
  white-space: pre-wrap;
  word-break: break-word;
}

.answer-text :deep(code) {
  background: #f4f4f5;
  padding: 2px 6px;
  border-radius: 4px;
  font-family: 'Courier New', monospace;
  font-size: 14px;
}

.answer-text :deep(strong) {
  font-weight: 600;
  color: #409eff;
}

.loading-container {
  padding: 16px 0;
}

.citations-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.citation-item {
  border-left: 4px solid #409eff;
  transition: all 0.3s ease;
}

.citation-item:hover {
  border-left-color: #67c23a;
  transform: translateX(4px);
}

.citation-header {
  margin-bottom: 8px;
}

.citation-tag {
  display: inline-flex;
  align-items: center;
  gap: 4px;
}

.citation-text {
  color: #606266;
  line-height: 1.6;
  font-size: 14px;
}

.loading-card {
  opacity: 0.8;
}

.app-footer {
  background: #303133;
  color: #909399;
  text-align: center;
  padding: 16px;
  height: auto !important;
}

.footer-content {
  font-size: 14px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .app-main {
    padding: 16px;
  }

  .app-title {
    font-size: 24px;
  }

  .header-subtitle {
    font-size: 12px;
    margin-left: 0;
  }

  .query-form :deep(.el-form-item__label) {
    width: 80px !important;
  }

  .query-form :deep(.el-radio-group) {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .query-form :deep(.el-radio-button) {
    width: 100%;
  }
}
</style>

