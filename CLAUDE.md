# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

Structured Obsidian wiki of deep learning hyperparameter tuning tricks extracted from the Zhihu question 「深度学习调参有哪些技巧？」(164 answers, 30+ authors). 72 Markdown files across 14 category directories, all in Chinese, cross-linked with `[[wikilinks]]`.

## Content conventions

**Every knowledge-point page** follows this 5-section structure:
```
# <名称>
## 是什么
## 为什么重要
## 如何使用
## 常见误区
## 参见
```

**Every main category page** follows:
```
# <类别名>
## 概述
## 详细知识 (bullet index of sub-pages)
## 核心技巧 (individual tips with 来源/适用场景/原文/要点/参见)
```

**Language**: All content in Chinese. Preserve original technical terminology and colloquialisms from source authors.

**Source attribution**: Every tip must cite its original author. The raw source is `raw/tricks.md`.

## Wiki-link conventions

- Same-directory links: `[[page-name]]` (no `.md` extension)
- Cross-category links: `[[../category/page-name]]`
- Root-level links: `[[../../contradictory]]`, `[[../../README]]`
- No trailing slashes on directory references
- Architecture/model names (U-Net, MobileNet, GPT, ViT, etc.) are acceptable red links

## Adding new content

1. New knowledge-point page → place in the matching category directory, follow 5-section format
2. New main category → create directory + main page, add to README.md nav table and index.md
3. New tips → add to the relevant main page under `## 核心技巧` with source/场景/原文/要点/参见
4. After adding pages, scan for orphan `[[wikilinks]]` and resolve them

## GitHub Pages

Site hosted at `1998x-stack.github.io/ai-tricks/` via Jekyll + Cayman theme. Config in `_config.yml`. Entry point is `index.md`. Pages rebuild on push to `main`.
