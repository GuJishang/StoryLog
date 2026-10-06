#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一键跑完整个项目的自动化测试（五个 Python 套件 + 一个前端端到端套件）。

用法：
    python tests/run_all.py            # 全部
    python tests/run_all.py --py       # 只跑 Python 套件（不依赖浏览器）
    python tests/run_all.py --ui       # 只跑前端端到端（需要 playwright + Chrome）

说明：
- Python 套件全部使用 mock / 纯检索路径，**不消耗任何 LLM 额度**。
- 前端套件需要 Node 与 playwright；未安装时该套件会被标记为 SKIP 而不算失败。
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

PY_SUITES = [
    ("A  路径A · LLM 抽取与匹配", "tests/test_llm_path.py"),
    ("B  路径B · AI 复盘管线", "tests/test_ai_review.py"),
    ("C  路径C · RAG 检索内核", "tests/test_rag.py"),
    ("D  路径D · Agent 层", "tests/test_agent.py"),
    ("API 服务接口端到端", "tests/test_server_api.py"),
]
UI_SUITE = ("UI 前端剧情问答端到端", "tests/test_ui_qa.cjs")

SUMMARY_RE = re.compile(r"(\d+)\s+passed,\s+(\d+)\s+failed")


def run(argv, cwd=ROOT):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("LLM_API_KEY", None)          # 保证测试不触网、不消耗额度
    p = subprocess.run(argv, cwd=cwd, env=env, capture_output=True)
    out = (p.stdout or b"").decode("utf-8", "replace") + (p.stderr or b"").decode("utf-8", "replace")
    return p.returncode, out


def main():
    ap = argparse.ArgumentParser(description="剧情志 · 全量测试")
    ap.add_argument("--py", action="store_true", help="只跑 Python 套件")
    ap.add_argument("--ui", action="store_true", help="只跑前端端到端套件")
    ap.add_argument("-v", "--verbose", action="store_true", help="失败时打印完整输出")
    args = ap.parse_args()

    suites = []
    if not args.ui:
        suites += [(n, [sys.executable, f], "py") for n, f in PY_SUITES]
    if not args.py:
        node = os.environ.get("NODE", "node")
        suites.append((UI_SUITE[0], [node, UI_SUITE[1]], "node"))

    total_pass = total_fail = 0
    rows = []
    for name, argv, kind in suites:
        if kind == "node" and not os.path.exists(os.path.join(ROOT, "node_modules", "playwright")) \
                and not os.environ.get("NODE_PATH"):
            rows.append((name, "SKIP", "未找到 playwright，设置 NODE_PATH 或 npm i -D playwright"))
            continue
        code, out = run(argv)
        m = SUMMARY_RE.search(out)
        if m:
            p, f = int(m.group(1)), int(m.group(2))
            total_pass += p
            total_fail += f
            rows.append((name, f"{p} passed / {f} failed", "" if code == 0 else "退出码非 0"))
        else:
            # 套件自身崩溃（语法错误、依赖缺失等）
            rows.append((name, "ERROR", out.strip().splitlines()[-1] if out.strip() else "无输出"))
            total_fail += 1
        if args.verbose and code != 0:
            print(f"\n===== {name} 完整输出 =====\n{out}")

    width = max(len(r[0]) for r in rows)
    print("\n" + "=" * (width + 34))
    for name, res, note in rows:
        line = f"{name.ljust(width)}   {res}"
        if note:
            line += f"   ({note})"
        print(line)
    print("=" * (width + 34))
    print(f"合计：{total_pass} passed, {total_fail} failed")
    sys.exit(1 if total_fail else 0)


if __name__ == "__main__":
    main()
