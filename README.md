# 欣仔的技术小屋

个人技术博客，记录机器学习、深度学习、CUDA 编程与 Python 等方面的学习笔记和实践总结。

站点基于 [Hugo](https://gohugo.io/) 构建，主题为 [Blowfish](https://blowfish.page/)（以 Git 子模块放在 `themes/blowfish`）。文章放在 `content/posts/`，配置在 `config/_default/`。

## 安装 Hugo

Blowfish 依赖 [Hugo Extended](https://gohugo.io/installation/)。macOS 用 Homebrew 安装的就是 Extended 版本：

```bash
brew install hugo
```

确认输出里包含 `extended`：

```bash
hugo version
```

Linux、Windows 的安装方式见 [官方文档](https://gohugo.io/installation/)。

## 本地预览

```bash
git clone --recurse-submodules https://github.com/neuqzxy/blog.git
cd blog
hugo server
```

如果仓库已经克隆过，但还没有拉主题：

```bash
git submodule update --init --recursive
```

启动后在浏览器打开 `http://localhost:1313`。改文章或配置后会自动刷新。

## 部署到 GitHub Pages

仓库已配置 GitHub Actions：推送到 `master` 分支时会自动构建并发布到 GitHub Pages，也可以在 GitHub Actions 页面手动运行 `Build and deploy Hugo site`。

首次部署前，在 GitHub 仓库的 **Settings → Pages → Build and deployment** 中，将 **Source** 设为 **GitHub Actions**。工作流会递归检出 Blowfish 主题子模块，使用 Hugo Extended 构建，并自动采用 Pages 提供的站点 URL。当前仓库对应的项目站点地址为 <https://neuqzxy.github.io/blog/>。

生产环境构建（不包含草稿）：

```bash
hugo
```

静态文件输出到 `public/`。`draft: true` 的文章默认不会出现在构建结果里；本地要看草稿可以加 `-D`：

```bash
hugo server -D
```

## 写一篇文章

每篇文章是 `content/posts/<slug>/index.md`，封面图放在同一目录（例如 `featured.jpg`）。

```bash
hugo new content/posts/my-note/index.md
```

front matter 示例：

```yaml
---
title: 文章标题
date: 2026-09-24
tags:
  - ML
categories:
  - ML
draft: true
---
```

写完后把 `draft` 改成 `false`，或删掉这一行，文章才会进入正式构建。

公式使用 KaTeX。行内公式用 `$...$`，独立公式用 `$$...$$`。站点图片放在 `assets/img/`，正文里用 `/img/文件名` 引用。

## 目录

```text
config/_default/     站点、主题参数、菜单、语言
content/posts/       文章
content/authors/     关于页（作者）
layouts/             覆盖主题的布局与首页
assets/              图片、样式、脚本
themes/blowfish/     Blowfish 主题（子模块）
```

站点标题、简介和社交链接在 `config/_default/languages.en.toml`。外观、首页和文章展示在 `config/_default/params.toml`。导航菜单在 `config/_default/menus.en.toml`。
