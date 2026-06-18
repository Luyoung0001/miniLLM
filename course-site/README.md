# miniLLM Course Site

这是 `miniLLM` 课程的独立文档站点工程。课程源在上一级 `course/`，这里负责把课程内容同步成可浏览网站。

## 本地预览

```bash
cd course-site
python3 _scripts/sync_course.py
bundle install
bundle exec jekyll build
python3 -m http.server 4003 -d _site
```

## GitHub Actions

仓库内已经带好 GitHub Actions 工作流：`.github/workflows/deploy-course-site.yml`。
它只做一件事：校验课程站点能否成功同步并编译。

执行内容：

- `python3 _scripts/sync_course.py`
- `bundle exec jekyll build`

## Cloudflare Pages 手动配置

如果你在 Cloudflare Pages 面板里直接接 GitHub 仓库，建议这样填：

- Root directory: `course-site`
- Build command: `python3 _scripts/sync_course.py && bundle exec jekyll build`
- Build output directory: `_site`


`course-site/` 是 `miniLLM` 课程讲义的静态网站外壳。

内容真源仍然是仓库根下的 `course/`。本目录只负责：

- 把 `course/**/*.md` 同步成可发布的 Jekyll 页面
- 生成课程侧边栏导航
- 生成一页式连续阅读页
- 本地预览与静态构建

## 本地预览

1. 安装 Ruby 和 Bundler
2. 在本目录执行：

```bash
bundle config set path vendor/bundle
bundle install
python3 _scripts/sync_course.py
bundle exec jekyll serve
```

3. 打开 <http://localhost:4000>

## 目录约定

- `../course/`：课程内容真源
- `pages/_course/`：由同步脚本生成的 Jekyll collection 页面
- `_data/navigation.yml`：由同步脚本生成的侧边栏导航
- `pages/_course/continuous.md`：由同步脚本生成的一页式连续阅读页

## 更新课程内容后的发布前动作

```bash
python3 _scripts/sync_course.py
bundle exec jekyll build
```
