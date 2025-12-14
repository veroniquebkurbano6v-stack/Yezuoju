# StoryRag 前端项目

基于 Vue 3 + Vite + Element Plus 构建的现代化前端界面。

## 快速启动指南

### 第一步：安装 Node.js
确保已安装 Node.js (版本 16 或更高)
下载地址：https://nodejs.org/

### 第二步：安装依赖

```bash
cd frontend
npm install
```

如果网络较慢，可以使用国内镜像：

```bash
npm install --registry=https://registry.npmmirror.com
```

### 第三步：启动后端服务
在项目根目录（StoryRag v1.1）下运行：
```bash
uvicorn api:app --reload --port 8000
```

### 第四步：启动前端开发服务器

```bash
npm run dev
```

### 第五步：访问界面
打开浏览器访问：http://localhost:3000

## 技术栈

- **Vue 3** - 渐进式 JavaScript 框架
- **Vite** - 下一代前端构建工具
- **Element Plus** - 基于 Vue 3 的组件库
- **Axios** - HTTP 客户端

## 构建生产版本

```bash
npm run build
```

构建产物将输出到 `dist` 目录。

## 预览生产构建

```bash
npm run preview
```

## 项目结构

```
frontend/
├── src/
│   ├── App.vue          # 主应用组件
│   ├── main.js          # 应用入口
│   ├── api.js           # API 服务层
│   └── style.css        # 全局样式
├── index.html           # HTML 模板
├── package.json         # 项目配置
├── vite.config.js       # Vite 配置
└── README.md           # 说明文档
```

## 功能特性

- 🎨 现代化的 UI 设计，采用渐变背景和卡片式布局
- 📱 响应式设计，支持移动端和桌面端
- 🌍 多语言支持（中文、英文、日文）
- 🔍 智能语言检测
- 📚 全库查询和单书查询两种模式
- 💬 实时查询状态反馈
- 📝 引用片段展示
- ⚡ 快速响应和流畅动画

## 常见问题

### Q: npm install 失败怎么办？
A: 尝试使用国内镜像或检查网络连接

### Q: 前端无法连接后端？
A: 确保后端服务已启动在 http://localhost:8000

### Q: 如何构建生产版本？
A: 运行 npm run build，产物在 dist 目录

## 注意事项

1. 确保后端服务已启动在 `http://localhost:8000`
2. 如果后端运行在不同端口，请修改 `vite.config.js` 中的代理配置
3. 生产环境部署时，需要配置正确的 API 基础路径

