#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""剧情志 · 剧情更新检测总入口

依次运行三款游戏的检测脚本并汇总结果：
  原神        -> check_ys_wiki.py    (bilibili wiki 魔神任务页, revision 时间戳)
  鸣潮        -> check_wuwa_wiki.py  (库街区 wiki 剧情目录, 卡片列表快照比对)
  战双帕弥什  -> check_pns_wiki.py   (bilibili wiki 剧情回顾页, revision 时间戳)

Usage:
  python check_all_updates.py                  # 检测全部
  python check_all_updates.py --game genshin   # 只检测原神 (genshin|wuwa|pns)
  python check_all_updates.py --check-chapters # 强制章节比对（透传给子脚本）
"""
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

GAMES = [
    ("genshin", "原神", "check_ys_wiki.py"),
    ("wuwa", "鸣潮", "check_wuwa_wiki.py"),
    ("pns", "战双帕弥什", "check_pns_wiki.py"),
]

STATUS_MARKERS = ("FIRST_RUN", "NO_UPDATE", "UPDATE_DETECTED", "ERROR")


def run_one(script, extra_args):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    try:
        proc = subprocess.run(
            [sys.executable, script] + extra_args,
            cwd=SCRIPT_DIR,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        return proc.returncode, out, err
    except subprocess.TimeoutExpired:
        return 1, "", "TIMEOUT: 执行超过 300 秒"


def parse_status(output):
    status = "UNKNOWN"
    for line in output.splitlines():
        s = line.strip()
        if s in STATUS_MARKERS or s.startswith("ERROR"):
            status = s.split(":")[0]
            break
    new_count = 0
    for line in output.splitlines():
        if line.strip().startswith("NEW_CHAPTERS_FOUND:"):
            try:
                new_count = int(line.split(":", 1)[1].strip())
            except ValueError:
                new_count = 0
            break
    if status == "UNKNOWN":
        status = "CHAPTERS_FOUND" if new_count else "CHECKED"
    return status, new_count


def main():
    args = [a for a in sys.argv[1:]]

    # 解析 --game
    only = None
    passthrough = []
    i = 0
    while i < len(args):
        if args[i] == "--game" and i + 1 < len(args):
            only = args[i + 1]
            i += 2
            continue
        passthrough.append(args[i])
        i += 1

    selected = [g for g in GAMES if only is None or g[0] == only]
    if not selected:
        print(f"ERROR: 未知 --game {only!r}，可选: {', '.join(g[0] for g in GAMES)}")
        sys.exit(1)

    summary = []
    need_sync = []

    for gid, gname, script in selected:
        print(f"===== {gname} ({gid}) =====")
        code, out, err = run_one(script, passthrough)
        if out:
            print(out)
        if err:
            print(f"[stderr] {err}")
        if code != 0 and not out:
            print(f"EXIT_CODE: {code}")

        status, new_count = parse_status(out)
        summary.append((gname, status, new_count))
        if new_count > 0:
            need_sync.append(gname)
        print()

    print("===== SUMMARY =====")
    for gname, status, new_count in summary:
        extra = f"  ({new_count} 个新章节待同步)" if new_count else ""
        print(f"{gname}: {status}{extra}")

    if need_sync:
        print(f"NEED_SYNC: {', '.join(need_sync)}")
    else:
        print("NEED_SYNC: (无)")


if __name__ == "__main__":
    main()
