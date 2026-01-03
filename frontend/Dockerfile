# 构建阶段
FROM node:18-alpine as build-stage

# 设置工作目录
WORKDIR /app

# 复制package.json和package-lock.json
COPY package*.json ./

# 安装依赖
RUN npm install

# 复制项目文件
COPY . .

# 构建生产版本
RUN npm run build

# 生产阶段
FROM nginx:stable-alpine as production-stage

# 复制构建好的文件到Nginx的html目录
COPY --from=build-stage /app/dist /usr/share/nginx/html

# 复制自定义Nginx配置文件（如果有）
# COPY nginx.conf /etc/nginx/conf.d/default.conf

# 暴露端口
EXPOSE 80

# 启动Nginx
CMD ["nginx", "-g", "daemon off;"]
