#!/bin/bash

echo "🚀 开始部署博客..."

# 清理缓存
echo "🧹 清理缓存..."
npx hexo clean

# 生成静态文件
echo "📦 生成静态文件..."
npx hexo generate

# 部署到 GitHub Pages
echo "🌐 部署到 GitHub Pages..."
npx hexo deploy

echo "✅ 部署完成！"
echo "🎉 访问 https://neuqzxy.github.io 查看你的博客"
