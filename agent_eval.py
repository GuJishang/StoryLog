#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""剧情志 · Agent 评测 harness

对 storylog_agent 跑一组带标注的问题，量化输出：

  效果   工具召回率（期望工具是否被调用）· 关键词命中率 · 拒答准确率 · 综合得分
  效率   平均步数 · 平均工具调用数 · 平均耗时 · 平均 token 消耗
  安全   Guardrail 触发统计（含提示注入拦截）

说明：评测集 `agent_eval_set.json` 是本项目自建的（18 题，覆盖检索/目录/个人记录/
库统计/更新状态/无需工具/库外拒答 七类），不是公开 benchmark，不宣称通用性。

用法
----
  python agent_eval.py                      # 全量评测，输出 markdown 报告
  python agent_eval.py --limit 4            # 只跑前 4 题（快速验证）
  python agent_eval.py --delay 25           # 每题之间间隔秒数（规避 RPM 限流）
  python agent_eval.py --out report.md --json results.json
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

import storylog_agent as agent
from storylog_llm import LLM_MODEL

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SET = os.path.join(ROOT, "agent_eval_set.json")

_REFUSAL_RE = re.compile(
    r"(无法(回答|说明|确定|提供)|没有(找到|查询到|相关|收录|记录)|查不到|未(能|收录|找到)"
    r"|不(足以|够)|缺少|暂无|资料不足|不在(我的|本地)|超出|知识库(中)?(没有|未))"
)


def looks_like_refusal(text):
    return bool(_REFUSAL_RE.search(text or ""))


def is_rate_limited(row):
    """判断这题是不是被限流打下去的降级结果。

    限流是环境问题不是能力问题：如果把它算进分数，测的就是 API 配额而不是 Agent。
    命中限流的题会等待后重跑。
    """
    if row.get("mode") != "rag_fallback":
        return False
    return any("429" in g or "rate_limit" in g.lower() for g in (row.get("guardrails") or []))


def score_case(case, res):
    """单题打分：工具 0.4 + 关键词 0.4 + 拒答行为 0.2。

    任何一项无法判定（如未列期望工具）按满分计入，避免用空标注拉低分数。
    """
    tools_used = res.get("toolsUsed") or []
    expect_tools = case.get("expectTools") or []
    answered = bool(res.get("answer")) and res.get("status") == "ok"

    # 工具召回：期望工具全部被调用（无期望则视为命中）
    missing = [t for t in expect_tools if t not in tools_used]
    tool_hit = not missing

    # 关键词：mustInclude 命中任一即可（宽松，避免同义改写误判）
    must = case.get("mustInclude") or []
    keyword_hit = True if not must else any(k in (res.get("answer") or "") for k in must)

    # 拒答行为：库外问题应拒答；库内问题不应拒答
    refused = looks_like_refusal(res.get("answer") or "")
    if case.get("expectRefusal"):
        refusal_ok = refused
    else:
        refusal_ok = answered and not refused

    score = 0.4 * tool_hit + 0.4 * keyword_hit + 0.2 * refusal_ok
    return {
        "toolHit": tool_hit, "missingTools": missing,
        "keywordHit": keyword_hit, "refused": refused, "refusalOk": refusal_ok,
        "answered": answered, "score": round(score, 3),
    }


def run_case(case):
    t0 = time.time()
    res = agent.run_agent(case["question"])
    detail = score_case(case, res)
    return {
        "id": case["id"],
        "category": case.get("category", ""),
        "question": case["question"],
        "expectTools": case.get("expectTools") or [],
        "toolsUsed": res.get("toolsUsed") or [],
        "mode": res.get("mode"),
        "status": res.get("status"),
        "grounding": res.get("grounding"),
        "steps": len(res.get("steps") or []),
        "elapsed": res.get("elapsed"),
        "tokens": (res.get("usage") or {}).get("total_tokens", 0),
        "guardrails": res.get("guardrails") or [],
        "answer": res.get("answer") or "",
        "sources": len(res.get("sources") or []),
        "wall": round(time.time() - t0, 1),
        **detail,
    }


def summarize(rows):
    n = len(rows) or 1
    tool_cases = [r for r in rows if r["expectTools"]]
    refusal_cases = [r for r in rows if r["category"].endswith("应拒答")]
    return {
        "cases": len(rows),
        "avgScore": round(sum(r["score"] for r in rows) / n, 3),
        "toolRecall": round(sum(1 for r in tool_cases if r["toolHit"]) / (len(tool_cases) or 1), 3),
        "toolCases": len(tool_cases),
        "refusalAcc": round(sum(1 for r in refusal_cases if r["refusalOk"]) / (len(refusal_cases) or 1), 3),
        "refusalCases": len(refusal_cases),
        "okRate": round(sum(1 for r in rows if r["status"] == "ok") / n, 3),
        "avgSteps": round(sum(r["steps"] for r in rows) / n, 2),
        "avgTools": round(sum(len(r["toolsUsed"]) for r in rows) / n, 2),
        "avgElapsed": round(sum(r["elapsed"] or 0 for r in rows) / n, 1),
        "avgTokens": round(sum(r["tokens"] for r in rows) / n),
        "totalTokens": sum(r["tokens"] for r in rows),
        "guardrailHits": sum(len(r["guardrails"]) for r in rows),
        "modes": {m: sum(1 for r in rows if r["mode"] == m)
                  for m in sorted({r["mode"] for r in rows})},
    }


def render(rows, stats, set_path):
    lines = [
        "# 剧情志 · Agent 评测报告",
        "",
        f"- 生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        f"- 被测模型：`{LLM_MODEL}`",
        f"- 评测集：`{os.path.basename(set_path)}`（自建，{stats['cases']} 题）",
        f"- Agent 预算：{agent.AGENT_MAX_STEPS} 步 / {agent.AGENT_TOKEN_BUDGET} tokens / "
        f"{agent.AGENT_TIME_BUDGET:.0f}s",
        "",
        "## 一、总体指标",
        "",
        "| 指标 | 数值 | 说明 |",
        "|---|---|---|",
        f"| 综合得分 | **{stats['avgScore']*100:.1f} / 100** | 工具 0.4 + 关键词 0.4 + 拒答 0.2 |",
        f"| 工具召回率 | **{stats['toolRecall']*100:.1f}%** | {stats['toolCases']} 道需调用工具的题 |",
        f"| 拒答准确率 | **{stats['refusalAcc']*100:.1f}%** | {stats['refusalCases']} 道库外问题 |",
        f"| 成功率 | {stats['okRate']*100:.1f}% | status=ok |",
        f"| 平均步数 | {stats['avgSteps']} | ReAct 步数 |",
        f"| 平均工具调用 | {stats['avgTools']} 次 | 含多工具并行 |",
        f"| 平均耗时 | {stats['avgElapsed']}s | 单题墙钟（含 `AGENT_RPM` 主动限速的等待，非纯模型延迟） |",
        f"| 平均 tokens | {stats['avgTokens']} | 单题消耗（合计 {stats['totalTokens']}） |",
        f"| Guardrail 触发 | {stats['guardrailHits']} 次 | 含提示注入拦截、预算熔断等 |",
        "",
        f"- 运行模式分布：{stats['modes']}",
        "",
        "## 二、逐题结果",
        "",
        "| ID | 类别 | 问题 | 模式 | 期望工具 | 实际工具 | 得分 | 耗时 | tokens |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        mark = "✅" if r["score"] >= 0.8 else ("⚠️" if r["score"] >= 0.5 else "❌")
        lines.append(
            f"| {r['id']} | {r['category']} | {r['question'][:20]} | {r['mode']} | "
            f"{','.join(r['expectTools']) or '-'} | {','.join(r['toolsUsed']) or '-'} | "
            f"{mark} {r['score']:.2f} | {r['elapsed']}s | {r['tokens']} |"
        )

    lines += ["", "## 三、未命中与异常明细", ""]
    bad = [r for r in rows if r["score"] < 0.8]
    if not bad:
        lines.append("无（全部用例得分 ≥ 0.8）。")
    for r in bad:
        lines.append(f"### {r['id']} · {r['question']}")
        lines.append(f"- 期望工具 `{r['expectTools']}` / 实际 `{r['toolsUsed']}`；"
                     f"missing={r['missingTools']}；keywordHit={r['keywordHit']}；refusalOk={r['refusalOk']}")
        if r["guardrails"]:
            lines.append(f"- Guardrails: {', '.join(r['guardrails'])}")
        lines.append(f"- 回答：{r['answer'][:220]}")
        lines.append("")

    failed = [r for r in rows if r["status"] != "ok"]
    if failed:
        lines += ["## 四、失败用例", ""]
        for r in failed:
            lines.append(f"- {r['id']} status={r['status']} answer={r['answer'][:80]!r}")
        lines.append("")

    lines += [
        "## 五、已知局限",
        "",
        "- 评测集为项目自建（18 题），覆盖面有限，不构成通用 benchmark。",
        "- 关键词命中采用「命中任一」的宽松判定，同义改写可能被误判为命中。",
        "- 拒答判定基于否定词正则，未做人工复核。",
        "- 单次运行结果受模型采样随机性影响；kimi-k2.x 被强制 temperature=1，波动更大。",
        "",
    ]
    return "\n".join(lines)


def acquire_single_instance_lock(port=8099):
    """单实例锁：用「占用本地端口」做互斥，进程退出即自动释放，不留脏锁文件。

    为什么需要：Moonshot 的 RPM 是**组织级**的，同时跑两个评测进程会各自按 3 RPM 发请求，
    合起来超限 → 大面积 429 → 评测被迫反复等待重跑，而且两个进程还会同时写同一个结果文件。
    实测踩过这个坑：一次「限速开着却持续 429」的排查，根因就是上一轮遗留的评测进程还活着。

    返回 socket 对象（需保持引用），失败则返回 None。
    """
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", port))
        s.listen(1)
        return s
    except OSError:
        s.close()
        return None


def main():
    ap = argparse.ArgumentParser(description="剧情志 · Agent 评测")
    ap.add_argument("--set", default=DEFAULT_SET, help="评测集 JSON 路径")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "剧情志-Agent评测报告.md"))
    ap.add_argument("--json", dest="json_out", help="同时输出机器可读结果")
    ap.add_argument("--limit", type=int, help="只跑前 N 题")
    ap.add_argument("--delay", type=float, default=0.0, help="每题之间的间隔秒数（规避限流）")
    ap.add_argument("--retry-rate", type=int, default=2, help="命中限流的题最多重跑几次")
    ap.add_argument("--wait-after-429", type=float, default=62.0, help="限流后等待秒数")
    ap.add_argument("--lock-port", type=int, default=8099, help="单实例锁端口（0 = 关闭）")
    args = ap.parse_args()

    lock = None
    if args.lock_port:
        lock = acquire_single_instance_lock(args.lock_port)
        if lock is None:
            print(f"REFUSE: 已有评测在运行（锁端口 {args.lock_port} 被占用）。\n"
                  f"        RPM 是组织级配额，并发跑评测会互相抢配额并大面积 429。\n"
                  f"        E02 等被限流的题会被判为环境失败。请等它结束，"
                  f"或确认无进程在跑后重试（确需绕过可加 --lock-port 0）。", flush=True)
            sys.exit(2)

    with open(args.set, "r", encoding="utf-8") as f:
        cases = json.load(f)
    if args.limit:
        cases = cases[: args.limit]

    if not agent.llm_available():
        print("NOTE: 未配置 LLM_API_KEY，Agent 会走降级链路，评测结果不代表真实效果。", flush=True)

    print(f"EVAL_START cases={len(cases)} model={LLM_MODEL} rpm={agent.AGENT_RPM}", flush=True)
    rows = []
    for i, case in enumerate(cases, 1):
        attempt = 0
        while True:
            row = run_case(case)
            attempt += 1
            if is_rate_limited(row) and attempt <= args.retry_rate:
                print(f"      ~ {row['id']} 命中限流，{args.wait_after_429:.0f}s 后重跑"
                      f"（第 {attempt} 次重试）", flush=True)
                time.sleep(args.wait_after_429)
                continue
            break
        rows.append(row)
        flag = "OK  " if row["score"] >= 0.8 else "MISS"
        print(f"  [{i}/{len(cases)}] {flag} {row['id']} score={row['score']:.2f} "
              f"mode={row['mode']} tools={row['toolsUsed']} {row['elapsed']}s "
              f"{row['tokens']}tok  Q: {row['question'][:24]}", flush=True)
        if args.delay and i < len(cases):
            time.sleep(args.delay)

    stats = summarize(rows)
    report = render(rows, stats, args.set)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"stats": stats, "rows": rows}, f, ensure_ascii=False, indent=2)

    print(f"\nEVAL_DONE 综合得分={stats['avgScore']*100:.1f} 工具召回={stats['toolRecall']*100:.1f}% "
          f"拒答准确={stats['refusalAcc']*100:.1f}% 平均步数={stats['avgSteps']} "
          f"平均耗时={stats['avgElapsed']}s 平均tokens={stats['avgTokens']}")
    print(f"报告已写入 {args.out}")


if __name__ == "__main__":
    main()
