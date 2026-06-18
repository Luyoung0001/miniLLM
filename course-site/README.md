# miniLLM Course Site

这是 `miniLLM` 课程的独立文档站点工程。课程源在上一级 `course/`，这里负责把课程内容同步成可浏览网站，并发布到 Cloudflare Pages。

## 本地预览

```bash
cd course-site
python3 _scripts/sync_course.py
bundle install
bundle exec jekyll build
python3 -m http.server 4003 -d _site
```

## Cloudflare Pages 部署

仓库内已经带好 GitHub Actions 工作流：`.github/workflows/deploy-course-site.yml`。

默认约定：

- Pages project name: `minillm-course`
- production branch: `main`
- build source dir: `course-site/`
- deploy output dir: `course-site/_site/`

第一次接 Cloudflare 时，需要在 GitHub 仓库 secrets 中配置：

- `CLOUDFLARE_API_TOKEN`
- `CLOUDFLARE_ACCOUNT_ID`

推荐 token 权限至少覆盖：

- Cloudflare Pages: Edit
- Account: Read

如果要先在本地创建 Pages project，可用：

```bash
export CLOUDFLARE_API_TOKEN=...
export CLOUDFLARE_ACCOUNT_ID=...
./scripts/create-cloudflare-pages-project.sh minillm-course main
```

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
