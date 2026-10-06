#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
剧情志 - 本地服务器

作用：
  1. 托管 index.html 等静态文件
  2. 提供数据读写接口，把剧情记录自动保存为本地 JSON 文件
     （文件位置与 index.html 同目录，不会因浏览器清缓存而丢失）
  3. 每次保存时同步生成「剧情记录数据.js」桥接文件 —— 这样即使直接
     双击 index.html（file:// 协议）打开，页面也能读到全部记录

接口：
  GET  /api/data          -> 返回本地数据文件内容
  POST /api/data          -> 将请求体保存到本地数据文件（并生成滚动备份 + 桥接文件）
                             接口带 CORS 头，允许 file:// 页面跨源写入
  GET  /api/index/status  -> 剧情问答索引状态 + 重建任务进度
  POST /api/ask           -> 剧情问答（RAG：BM25 检索 + LLM 生成 + 引用核验）
                             body: {"question": "...", "game": "genshin|wuwa|pns|null", "topk": 6}
  POST /api/reindex       -> 后台重建剧情问答索引，body: {"maxPages": 10, "refresh": false}
  GET  /api/agent         -> Agent 能力信息（可用工具清单 + 活跃会话数）
  POST /api/agent         -> Agent 问答（ReAct 循环 + 工具调用 + 会话记忆）
                             body: {"question": "...", "sessionId": "可选", "game": "可选"}

用法：
  python server.py          # 默认端口 8090
  python server.py 9000     # 指定端口
然后浏览器打开 http://localhost:8090/index.html
"""
import json
import os
import sys
import time
import shutil
import threading
import http.server
import socketserver
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "剧情记录数据.json")
BRIDGE_FILE = os.path.join(BASE_DIR, "剧情记录数据.js")
BACKUP_DIR = os.path.join(BASE_DIR, ".backups")
API_PATH = "/api/data"

# ---- 剧情问答（RAG，路径 C） ----
ASK_PATH = "/api/ask"
REINDEX_PATH = "/api/reindex"
INDEX_STATUS_PATH = "/api/index/status"
# ---- Agent 问答（ReAct + 工具调用） ----
AGENT_PATH = "/api/agent"
RAG_PATHS = (ASK_PATH, REINDEX_PATH, INDEX_STATUS_PATH, AGENT_PATH)

# storylog_rag 依赖 requests / beautifulsoup4。缺失时数据接口照常工作，只是问答不可用，
# 这样「不装依赖也能先跑起来记录数据」的旧行为不被破坏。
try:
    import storylog_rag as rag
    RAG_ERROR = None
except Exception as _e:                      # pragma: no cover
    rag = None
    RAG_ERROR = f"{type(_e).__name__}: {_e}"

# Agent 层依赖 storylog_rag（索引）与 storylog_llm（模型），同样允许单独缺失
try:
    import storylog_agent as agent
    AGENT_ERROR = None
except Exception as _e:                      # pragma: no cover
    agent = None
    AGENT_ERROR = f"{type(_e).__name__}: {_e}"

# 重建索引要抓 wiki，属于耗时操作：放后台线程 + 状态轮询，避免请求超时
_index_job = {
    "running": False, "startedAt": None, "finishedAt": None,
    "ok": None, "log": [], "error": None, "stats": None,
}
_index_lock = threading.Lock()


def _run_reindex(args):
    """后台重建索引，把进度写进 _index_job 供前端轮询。"""
    def progress(msg):
        with _index_lock:
            _index_job["log"].append(msg)
            del _index_job["log"][:-40]

    try:
        index = rag.build_index(
            games=args["games"], max_pages=args["max_pages"],
            refresh=args["refresh"], progress=progress,
        )
        with _index_lock:
            _index_job.update({"ok": True, "stats": rag.index_stats(index), "finishedAt": time.time()})
    except Exception as e:
        with _index_lock:
            _index_job.update({"ok": False, "error": f"{type(e).__name__}: {e}",
                               "finishedAt": time.time()})
    finally:
        with _index_lock:
            _index_job["running"] = False

# 滚动备份：距上次备份超过该秒数才生成新快照，最多保留 MAX_BACKUPS 份
BACKUP_INTERVAL = 1800
MAX_BACKUPS = 60

EMPTY_DATA = '{"app":"storylog","reviews":{}}'


def log(msg):
    print(msg, flush=True)


def atomic_write(path, text):
    """先写临时文件再替换，避免写到一半中断损坏文件。"""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def write_bridge(data):
    """生成磁盘桥接文件，供 file:// 直接打开时读取。

    使用 ensure_ascii=True，内容全是 ASCII，
    这样无论浏览器按哪种编码解析脚本都不会乱码。
    """
    try:
        payload = json.dumps(data, ensure_ascii=True)
        atomic_write(BRIDGE_FILE, "window.STORYLOG_DATA = " + payload + ";\n")
    except Exception as e:
        log(f"[bridge] 生成失败: {e}")


def snapshot_backup():
    """把当前数据文件快照一份到 .backups 目录（节流）。"""
    if not os.path.exists(DATA_FILE):
        return
    try:
        os.makedirs(BACKUP_DIR, exist_ok=True)
        backups = sorted(
            f for f in os.listdir(BACKUP_DIR)
            if f.startswith("剧情记录数据_") and f.endswith(".json")
        )
        # 距最近一次快照不超过 BACKUP_INTERVAL 就跳过
        if backups:
            newest = os.path.join(BACKUP_DIR, backups[-1])
            if time.time() - os.path.getmtime(newest) < BACKUP_INTERVAL:
                return
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        shutil.copy2(DATA_FILE, os.path.join(BACKUP_DIR, f"剧情记录数据_{stamp}.json"))
        # 超出上限则删除最旧的
        while len(backups) + 1 > MAX_BACKUPS:
            try:
                os.remove(os.path.join(BACKUP_DIR, backups.pop(0)))
            except OSError:
                break
    except Exception as e:
        log(f"[backup] 快照失败: {e}")


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE_DIR, **kwargs)

    # ---- 数据接口 ----
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/favicon.ico":
            # 无图标时返回 204，避免浏览器控制台一直报 404 噪声
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if path == API_PATH:
            self._read_data()
            return
        if path == INDEX_STATUS_PATH:
            self._index_status()
            return
        if path == AGENT_PATH:
            self._agent_info()
            return
        super().do_GET()

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == API_PATH:
            self._write_data()
            return
        if path == ASK_PATH:
            self._ask()
            return
        if path == REINDEX_PATH:
            self._reindex()
            return
        if path == AGENT_PATH:
            self._agent_ask()
            return
        self.send_error(404, "Not found")

    def do_OPTIONS(self):
        # 跨源预检（file:// 页面 -> http://localhost）需要
        if self.path.split("?")[0] in (API_PATH,) + RAG_PATHS:
            self.send_response(204)
            self._cors_headers()
            self.send_header("Access-Control-Max-Age", "86400")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_error(404, "Not found")

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, code, obj, with_cors=True):
        raw = obj if isinstance(obj, bytes) else str(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        if with_cors:
            self._cors_headers()
        self.end_headers()
        self.wfile.write(raw)

    def _read_data(self):
        if os.path.exists(DATA_FILE):
            try:
                with open(DATA_FILE, "r", encoding="utf-8") as f:
                    body = f.read()
                if not body.strip():
                    body = EMPTY_DATA
            except Exception as e:
                self.send_error(500, f"read failed: {e}")
                return
        else:
            body = EMPTY_DATA
        self._send_json(200, body.encode("utf-8"))

    def _write_data(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            data = json.loads(body)          # 校验合法性，避免写入坏数据
        except Exception as e:
            self.send_error(400, f"invalid json: {e}")
            return

        # 过滤测试残留记录：真实章节 id 不会以 "__" 开头。
        # 这样即使某个还开着的页面把含测试数据的旧状态推回来，也不会污染数据文件。
        reviews = data.get("reviews")
        if isinstance(reviews, dict):
            cleaned = {k: v for k, v in reviews.items() if not str(k).startswith("__")}
            if len(cleaned) != len(reviews):
                log(f"[clean] 已丢弃 {len(reviews) - len(cleaned)} 条测试残留记录")
                data["reviews"] = cleaned

        snapshot_backup()                    # 覆盖前先留一份快照

        # 内容没有实质变化（只有时间戳不同）时不写盘。
        # 这能切断「页面推送 -> 文件变动 -> 预览自动刷新 -> 页面再推送」的反馈循环。
        same = False
        if os.path.exists(DATA_FILE):
            try:
                with open(DATA_FILE, "r", encoding="utf-8") as f:
                    old = json.load(f)
                same = (old.get("reviews") or {}) == (data.get("reviews") or {})
            except Exception:
                same = False

        if same:
            n = len((data.get("reviews") or {}))
            log(f"[skip] {datetime.now():%H:%M:%S}  内容无变化，跳过写盘（{n} 条）")
            self._send_json(200, json.dumps({"ok": True, "savedAt": data.get("savedAt"),
                                             "count": n, "skipped": True}))
            return

        try:
            atomic_write(DATA_FILE, json.dumps(data, ensure_ascii=False, indent=2))
            write_bridge(data)               # 同步刷新 file:// 用的桥接文件
        except Exception as e:
            self.send_error(500, f"write failed: {e}")
            return

        n = len((data.get("reviews") or {}))
        origin = self.headers.get("Origin", "-")
        referer = self.headers.get("Referer", "-")
        ua = (self.headers.get("User-Agent", "-") or "-")[:60]
        log(f"[save] {datetime.now():%H:%M:%S}  已保存 {n} 条记录 -> {os.path.basename(DATA_FILE)}"
            f"  origin={origin} referer={referer} ua={ua}")

        self._send_json(200, json.dumps({"ok": True, "savedAt": data.get("savedAt"), "count": n}))

    # ---- 剧情问答（RAG） ----
    def _read_json_body(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length).decode("utf-8") if length else ""
            return json.loads(raw or "{}")
        except Exception:
            return None

    def _index_status(self):
        with _index_lock:
            job = dict(_index_job)
        stats = None
        if rag is not None:
            index = rag.load_index()
            if index:
                stats = rag.index_stats(index)
        self._send_json(200, json.dumps({
            "ok": True, "ragAvailable": rag is not None, "ragError": RAG_ERROR,
            "job": job, "stats": stats,
        }, ensure_ascii=False))

    def _ask(self):
        if rag is None:
            self._send_json(503, json.dumps(
                {"ok": False, "error": f"问答模块不可用：{RAG_ERROR}"}, ensure_ascii=False))
            return
        payload = self._read_json_body()
        if payload is None:
            self._send_json(400, json.dumps({"ok": False, "error": "invalid json"}, ensure_ascii=False))
            return
        question = str(payload.get("question") or "").strip()
        if not question:
            self._send_json(400, json.dumps({"ok": False, "error": "question 不能为空"}, ensure_ascii=False))
            return
        game = payload.get("game") or None
        try:
            topk = int(payload.get("topk") or rag.DEFAULT_TOPK)
        except (TypeError, ValueError):
            topk = rag.DEFAULT_TOPK

        started = time.time()
        try:
            res = rag.answer_question(question, rag.load_index() or {}, topk=topk, game=game)
        except Exception as e:
            self._send_json(500, json.dumps(
                {"ok": False, "error": f"{type(e).__name__}: {e}"}, ensure_ascii=False))
            return
        log(f"[ask] {datetime.now():%H:%M:%S} mode={res['mode']} "
            f"耗时{time.time() - started:.1f}s Q: {question[:40]}")
        self._send_json(200, json.dumps({"ok": True, **res}, ensure_ascii=False))

    def _reindex(self):
        if rag is None:
            self._send_json(503, json.dumps(
                {"ok": False, "error": f"问答模块不可用：{RAG_ERROR}"}, ensure_ascii=False))
            return
        payload = self._read_json_body()
        if payload is None:
            self._send_json(400, json.dumps({"ok": False, "error": "invalid json"}, ensure_ascii=False))
            return
        with _index_lock:
            if _index_job["running"]:
                self._send_json(409, json.dumps(
                    {"ok": False, "error": "已有重建任务在进行中"}, ensure_ascii=False))
                return
            _index_job.update({
                "running": True, "startedAt": time.time(), "finishedAt": None,
                "ok": None, "log": [], "error": None, "stats": None,
            })
        try:
            max_pages = int(payload.get("maxPages") or rag.DEFAULT_MAX_PAGES)
        except (TypeError, ValueError):
            max_pages = rag.DEFAULT_MAX_PAGES
        args = {
            "games": [payload["game"]] if payload.get("game") else None,
            "max_pages": max_pages,
            "refresh": bool(payload.get("refresh")),
        }
        threading.Thread(target=_run_reindex, args=(args,), daemon=True).start()
        log(f"[reindex] {datetime.now():%H:%M:%S} 已启动后台重建 "
            f"maxPages={max_pages} refresh={args['refresh']}")
        self._send_json(202, json.dumps({"ok": True, "started": True}, ensure_ascii=False))

    # ---- Agent 问答（ReAct + 工具调用） ----
    def _agent_info(self):
        """暴露 Agent 能力，前端据此决定是否展示「Agent 模式」。"""
        if agent is None:
            self._send_json(200, json.dumps(
                {"ok": True, "agentAvailable": False, "agentError": AGENT_ERROR,
                 "tools": [], "sessions": 0}, ensure_ascii=False))
            return
        self._send_json(200, json.dumps({
            "ok": True, "agentAvailable": True, "agentError": None,
            "tools": [{"name": t["function"]["name"], "description": t["function"]["description"]}
                      for t in agent.TOOL_SCHEMAS],
            "limits": {"maxSteps": agent.AGENT_MAX_STEPS,
                       "tokenBudget": agent.AGENT_TOKEN_BUDGET,
                       "timeBudget": agent.AGENT_TIME_BUDGET},
            "sessions": agent.SESSIONS.count(),
        }, ensure_ascii=False))

    def _agent_ask(self):
        if agent is None:
            self._send_json(503, json.dumps(
                {"ok": False, "error": f"Agent 模块不可用：{AGENT_ERROR}"}, ensure_ascii=False))
            return
        payload = self._read_json_body()
        if payload is None:
            self._send_json(400, json.dumps({"ok": False, "error": "invalid json"}, ensure_ascii=False))
            return
        question = str(payload.get("question") or "").strip()
        if not question:
            self._send_json(400, json.dumps({"ok": False, "error": "question 不能为空"}, ensure_ascii=False))
            return
        game = payload.get("game") or None
        session = agent.SESSIONS.get(payload.get("sessionId"))

        started = time.time()
        try:
            res = agent.run_agent(question, session=session, game=game)
        except Exception as e:
            self._send_json(500, json.dumps(
                {"ok": False, "error": f"{type(e).__name__}: {e}"}, ensure_ascii=False))
            return
        log(f"[agent] {datetime.now():%H:%M:%S} mode={res['mode']} "
            f"tools={len(res['toolsUsed'])} 耗时{time.time() - started:.1f}s "
            f"session={session.id} Q: {question[:36]}")
        self._send_json(200, json.dumps(
            {"ok": True, "sessionId": session.id, **res}, ensure_ascii=False))

    def end_headers(self):
        # 开发用：禁止缓存静态文件，改完刷新即可见
        if self.path.endswith((".html", ".js", ".css")):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        # 静音静态资源日志，只保留接口日志
        if API_PATH in (self.path or ""):
            return
        pass


class ReusableServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    port = 8090
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass

    os.makedirs(BACKUP_DIR, exist_ok=True)

    # 启动时补齐桥接文件：已有 json 但缺 js（或 js 内容陈旧）就重新生成
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            need = True
            if os.path.exists(BRIDGE_FILE):
                with open(BRIDGE_FILE, "r", encoding="utf-8") as f:
                    cur = f.read()
                need = cur.strip() != ("window.STORYLOG_DATA = " + json.dumps(data, ensure_ascii=True) + ";").strip()
            if need:
                write_bridge(data)
                log("[bridge] 已根据数据文件重新生成 剧情记录数据.js")
        except Exception as e:
            log(f"[bridge] 初始化跳过: {e}")
    else:
        write_bridge({"app": "storylog", "savedAt": 0, "reviews": {}})

    log("=" * 56)
    log("  剧情志 本地服务器")
    log(f"  数据文件: {DATA_FILE}")
    log(f"  桥接文件: {BRIDGE_FILE}")
    log(f"  备份目录: {BACKUP_DIR}")
    log("  剧情问答: " + ("已启用 (/api/ask, /api/reindex)"
                          if rag is not None else f"未启用 —— {RAG_ERROR}"))
    log("  Agent 问答: " + (f"已启用 (/api/agent, {len(agent.TOOL_SCHEMAS)} 个工具)"
                           if agent is not None else f"未启用 —— {AGENT_ERROR}"))
    log(f"  打开页面: http://localhost:{port}/index.html")
    log("  按 Ctrl+C 退出")
    log("=" * 56)

    def run(p):
        with ReusableServer(("127.0.0.1", p), Handler) as httpd:
            httpd.serve_forever()

    try:
        run(port)
    except OSError as e:
        log(f"[warn] 端口 {port} 被占用: {e}")
        alt = port + 1
        log(f"[info] 尝试使用端口 {alt} ...")
        log(f"  打开页面: http://localhost:{alt}/index.html")
        run(alt)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\n已退出。")
