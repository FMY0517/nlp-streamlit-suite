# GitHub + Streamlit 部署说明

## 你现在已经具备的前置文件

- 入口文件：`vibe_coding_launcher.py`
- Python 依赖：`requirements.txt`
- Linux 依赖：`packages.txt`
- 忽略文件：`.gitignore`
- 干净部署目录：`streamlit_deploy_bundle/`

## 推荐做法：使用独立部署目录

你当前这套代码虽然已经位于一个 Git 仓库中，但那个仓库的根目录在上一级，不利于让 `packages.txt` 作为仓库根目录文件被 Streamlit 正确识别。

因此，**最推荐的做法**是：

1. 使用已经准备好的 `streamlit_deploy_bundle` 目录
2. 把这个目录单独作为一个新的 GitHub 仓库上传
3. 在 Streamlit 里直接部署这个新仓库

这样你部署时填写的主文件路径就是最简单的：

- `vibe_coding_launcher.py`

## 第 1 步：创建 GitHub 仓库

1. 打开 https://github.com/
2. 登录你的 GitHub 账号。
3. 点击右上角 `+`。
4. 点击 `New repository`。
5. Repository name 填一个英文名，例如：`nlp-streamlit-suite`
6. 选择 `Public` 或 `Private`
7. 不要勾选 `Add a README file`、`.gitignore`、`Choose a license`
8. 点击 `Create repository`

## 第 2 步：把本地项目推送到 GitHub

1. 在资源管理器里打开：

   `C:\Users\傅牧原\OneDrive\文档\我的文档\大学\研一下\自然语言处理\作业\streamlit_deploy_bundle`

2. 在这个文件夹空白处按住 `Shift` 再点击鼠标右键
3. 点击 `在此处打开 PowerShell 窗口` 或 `Open in Terminal`
4. 在打开的终端中依次执行：

```powershell
git init
git add .
git commit -m "Prepare app for GitHub and Streamlit deployment"
git branch -M main
git remote add origin https://github.com/你的GitHub用户名/你的仓库名.git
git push -u origin main
```

如果 Git 提示你没有用户名和邮箱，先执行：

```powershell
git config --global user.name "你的GitHub用户名"
git config --global user.email "你的GitHub邮箱"
```

## 第 3 步：打开 Streamlit Community Cloud

1. 打开 https://share.streamlit.io/
2. 使用 GitHub 登录
3. 如果弹出 GitHub 授权页面，点击 `Authorize Streamlit`

## 第 4 步：创建应用

1. 进入 workspace 后，点击右上角 `Create app`
2. 当页面问 `Do you already have an app?` 时，点击 `Yup, I have an app.`
3. 在表单里填写：
   - Repository：选择你刚刚上传的仓库
   - Branch：`main`
   - Main file path：`vibe_coding_launcher.py`
4. 可选：在 `App URL` 里填一个你想要的网址前缀

## 第 5 步：设置 Python 版本

1. 点击部署表单里的 `Advanced settings`
2. 找到 `Python version`
3. 选择 `3.10`
4. 点击 `Save`

注意：你的本地运行环境是 Python 3.10，所以云端也要选 3.10。

## 第 6 步：点击部署

1. 点击 `Deploy`
2. 等待安装依赖和启动应用
3. 第一次部署可能需要几分钟

## 第 7 步：如果报错，去哪里看

1. 在部署页面右侧直接看日志
2. 或者打开你的应用页面
3. 点击右下角 `Manage app`
4. 查看 `Cloud logs`

## 这份项目当前的部署说明

- Streamlit 会自动读取 `requirements.txt`
- Streamlit 会自动读取 `packages.txt`
- 入口文件已经改成推荐使用 `vibe_coding_launcher.py`
- 部分原本写死的 Windows 路径已经改成跨平台写法
- `streamlit_deploy_bundle` 已经是适合单独上传的新仓库目录

## 后续更新代码怎么同步到网页

以后你每次改完代码，只需要执行：

```powershell
git add .
git commit -m "Update app"
git push
```

推送到 GitHub 后，Streamlit 会自动重新部署。
