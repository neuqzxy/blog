# 快速参考卡片 🚀

## 🎯 快速命令

### 本地开发
```bash
npm run server           # 启动本地服务器 (端口 4000)
npx hexo server -p 8080  # 使用自定义端口
```

### 内容创建
```bash
npx hexo new post "标题"      # 创建文章
npx hexo new page "页面名"    # 创建页面
```

### 构建部署
```bash
npm run clean           # 清理缓存
npm run build           # 生成静态文件
./deploy.sh             # 一键部署
```

## 📝 常用语法

### LaTeX 公式
```markdown
行内: $E = mc^2$
块级: $$\int_{a}^{b} f(x)dx$$
```

### 代码块
````markdown
```python
print("Hello, World!")
```
````

### Mermaid 图表
````markdown
```mermaid
graph TD
    A --> B
```
````

## 🔗 重要链接

- **本地预览**: http://localhost:8080
- **线上地址**: https://neuqzxy.github.io
- **GitHub**: https://github.com/neuqzxy
- **Email**: zhouxinyu12138@gmail.com

## 📁 目录结构

```
blog/
├── _config.yml              # Hexo 配置
├── _config.redefine.yml     # 主题配置
├── source/
│   ├── _posts/             # 文章目录
│   ├── about/              # 关于页面
│   ├── categories/         # 分类页面
│   └── tags/               # 标签页面
├── deploy.sh               # 部署脚本
└── public/                 # 生成的静态文件
```

## ⚙️ 配置文件

- **Hexo 配置**: `_config.yml`
- **主题配置**: `_config.redefine.yml`
- **修改颜色**: `_config.redefine.yml` → `colors.primary`
- **修改社交链接**: `_config.redefine.yml` → `home_banner.social_links`

## 🎨 已启用功能

✅ LaTeX 数学公式  
✅ 代码高亮 (20+ 语言)  
✅ Mermaid 图表  
✅ 本地搜索  
✅ RSS 订阅  
✅ 字数统计  
✅ 中文优化  
✅ 明暗主题  
✅ 响应式设计  
✅ Font Awesome Pro  

## 📚 文档

- `README.md` - 项目说明
- `GUIDE.md` - 完整使用指南
- `CONFIGURATION_SUMMARY.md` - 配置总结
- `QUICK_REFERENCE.md` - 本文档

## 🆘 常见问题

**Q: 端口被占用？**
```bash
npx hexo server -p 8080
```

**Q: 更新主题？**
```bash
npm install hexo-theme-redefine@latest
```

**Q: 公式不显示？**
- 检查语法: `$公式$` 或 `$$公式$$`

**Q: 部署失败？**
- 确保 GitHub 仓库已创建
- 检查部署配置 (_config.yml)

## 💡 写作建议

1. 使用 `<!-- more -->` 分隔摘要
2. 添加合适的标签和分类
3. 图片放在 `source/images/`
4. 定期备份到 Git
5. 检查拼写和格式

## 🎉 开始写作

```bash
npx hexo new post "我的第一篇博客"
```

---

**需要帮助？** 查看 `GUIDE.md` 获取完整文档！
