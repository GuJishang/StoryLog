@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem 依次尝试：项目内虚拟环境 -> py 启动器 -> PATH 中的 python
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=py"
where py >nul 2>nul || set "PY=python"

echo 正在启动 剧情志 本地服务...
start "剧情志-本地服务" "%PY%" server.py
timeout /t 2 /nobreak >nul
start "" http://localhost:8090/index.html

echo.
echo 服务已在后台窗口运行，关闭那个窗口即停止服务。
echo 提示：「剧情问答」视图需要先构建索引：python storylog_rag.py --build
