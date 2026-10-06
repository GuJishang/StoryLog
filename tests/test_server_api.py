#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""server.py 接口端到端测试：RAG 三个新接口 + 原有数据接口回归。

会临时启动一个 server 子进程（端口 8092），测完自动结束。
默认不注入 LLM_API_KEY，因此问答走「纯检索降级」路径（快且确定）。
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
PY = sys.executable
PORT = 8092
BASE = f"http://127.0.0.1:{PORT}"

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("PASS" if cond else "FAIL") + f"  {name}" + (f"  | {detail}" if detail and not cond else ""))


def call(method, path, payload=None):
    url = BASE + path
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, body


os.makedirs(os.path.join(ROOT, "tests", ".tmp"), exist_ok=True)
logfile = open(os.path.join(ROOT, "tests", ".tmp", "server_test.log"), "w", encoding="utf-8")
env = dict(os.environ)
env.pop("LLM_API_KEY", None)          # 强制走降级路径

# 关键：测试用独立索引文件，S4 的 reindex(maxPages=1) 不再污染生产索引
TEST_INDEX = os.path.join(ROOT, "tests", ".tmp", "rag_index_test.json")
PROD_INDEX = os.path.join(ROOT, ".workbuddy", "rag_index.json")
if os.path.exists(PROD_INDEX):
    shutil.copy2(PROD_INDEX, TEST_INDEX)   # 以生产索引副本为初始状态（保证 S1/S2 有内容可比）
env["STORYLOG_INDEX_FILE"] = TEST_INDEX

proc = subprocess.Popen([PY, "server.py", str(PORT)], cwd=ROOT, env=env,
                        stdout=logfile, stderr=subprocess.STDOUT)

try:
    # 等待服务就绪
    ready = False
    for _ in range(40):
        try:
            st, _b = call("GET", "/api/index/status")
            if st == 200:
                ready = True
                break
        except Exception:
            time.sleep(0.25)
    check("S0 服务启动", ready)
    if not ready:
        raise SystemExit(1)

    # S1 索引状态
    st, body = call("GET", "/api/index/status")
    check("S1 status-200", st == 200, str(st))
    check("S1 status-rag可用", body.get("ragAvailable") is True, str(body)[:120])
    check("S1 status-含索引统计", (body.get("stats") or {}).get("chunks", 0) > 0, str(body.get("stats"))[:120])
    check("S1 status-任务初始未运行", body["job"]["running"] is False)

    # S2 问答（降级路径）
    st, body = call("POST", "/api/ask", {"question": "钟离是谁", "topk": 3})
    check("S2 ask-200", st == 200, str(st))
    check("S2 ask-降级search_only", body.get("mode") == "search_only", str(body.get("mode")))
    check("S2 ask-返回引用", len(body.get("citations") or []) > 0, str(body.get("citations"))[:100])
    check("S2 ask-有答案文本", len(body.get("answer") or "") > 10, (body.get("answer") or "")[:60])

    # S3 参数校验
    st, body = call("POST", "/api/ask", {"question": "   "})
    check("S3 ask-空问题400", st == 400, str(st))

    # ---- S6~S9 Agent 接口（路径 D） ----
    # 位置说明：必须放在 S4 reindex 之前。S4 会用 maxPages=1 重建测试索引，
    # 之后索引里只剩每游戏 1 页正文，检索类断言都会因为"库里没内容"而假失败。
    # 本测试进程未注入 LLM_API_KEY，Agent 会走「降级为单次 RAG」链路——
    # 这正好覆盖「无模型可用时接口是否能优雅返回」这一关键边界。
    st, body = call("GET", "/api/agent")
    check("S6 agent-info-200", st == 200, str(st))
    check("S6 agent-模块可用", body.get("agentAvailable") is True, str(body)[:140])
    names = [t.get("name") for t in (body.get("tools") or [])]
    check("S6 agent-工具数=5", len(names) == 5, str(names))
    check("S6 agent-关键工具齐全",
          {"search_story", "get_chapters", "get_my_reviews", "get_update_status",
           "get_library_stats"} == set(names), str(names))
    check("S6 agent-返回预算信息", (body.get("limits") or {}).get("maxSteps", 0) > 0, str(body.get("limits")))

    st, body = call("POST", "/api/agent", {"question": "钟离是谁"})
    check("S7 agent-200", st == 200, str(st))
    check("S7 agent-返回sessionId", bool(body.get("sessionId")), str(body)[:110])
    check("S7 agent-无Key走降级", body.get("mode") == "rag_fallback", str(body.get("mode")))
    check("S7 agent-降级有说明", "降级" in (body.get("note") or ""), str(body.get("note"))[:110])
    check("S7 agent-有答案文本", len(body.get("answer") or "") > 10, (body.get("answer") or "")[:60])
    check("S7 agent-记录降级原因", any("degraded" in g for g in (body.get("guardrails") or [])),
          str(body.get("guardrails")))
    sid = body.get("sessionId")

    st, body2 = call("POST", "/api/agent", {"question": "那战双的剧情呢", "sessionId": sid})
    check("S8 agent-会话id可复用", body2.get("sessionId") == sid, f"{sid} vs {body2.get('sessionId')}")
    check("S8 agent-会话轮数累计", ((body2.get("session") or {}).get("turns") or 0) >= 2,
          str(body2.get("session")))

    st, _b = call("POST", "/api/agent", {"question": "   "})
    check("S9 agent-空问题400", st == 400, str(st))
    st, body = call("POST", "/api/agent", {"question": "钟离是谁", "game": "dota"})
    check("S9 agent-非法game被拒", body.get("status") == "error", str(body)[:130])
    st, body = call("POST", "/api/agent", {"question": "钟离是谁", "game": "genshin"})
    check("S9 agent-限定游戏可用",
          st == 200 and body.get("mode") == "rag_fallback" and body.get("status") == "ok",
          str(body)[:140])

    # S4 重建索引（复用缓存，不 refresh，应较快）
    st, body = call("POST", "/api/reindex", {"maxPages": 1})
    check("S4 reindex-202", st == 202, f"{st} {body}")
    running_seen = False
    for _ in range(120):
        time.sleep(1)
        _st, s = call("GET", "/api/index/status")
        if s["job"]["running"]:
            running_seen = True
        else:
            break
    check("S4 reindex-状态可轮询", running_seen)
    check("S4 reindex-成功完成", s["job"]["ok"] is True, str(s["job"])[:160])
    check("S4 reindex-产出统计", (s["job"].get("stats") or {}).get("chunks", 0) > 0, str(s["job"].get("stats"))[:120])

    # S5 原有数据接口回归
    st, body = call("GET", "/api/data")
    check("S5 data-GET正常", st == 200 and isinstance(body, dict) and "reviews" in body, str(st))
    st, body = call("GET", "/api/nope")
    check("S5 未知路径404", st == 404, str(st))

finally:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    logfile.close()
    for f in (TEST_INDEX, TEST_INDEX + ".tmp"):
        try:
            os.remove(f)
        except OSError:
            pass

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
