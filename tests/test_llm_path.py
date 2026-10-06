#!/usr/bin/env python3
"""Path-A LLM 链路单测：模拟 llm_chat 响应，验证抽取校验/防幻觉/重试降级/语义匹配。

注意：LLM 层已抽到共享模块 storylog_llm，monkeypatch 必须打在该模块上；
check_ys_wiki 只是再导出，patch 它不会影响实际调用链。
这条约束本身就是「共享层抽取」的回归验证。
"""
import json
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import check_ys_wiki as m
import storylog_llm as llm

llm.LLM_API_KEY = "test-key"
llm.LLM_MAX_RETRIES = 1
llm.LLM_RETRY_BACKOFF = 0

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("PASS" if cond else "FAIL") + f"  {name}" + (f"  | {detail}" if detail and not cond else ""))


CANDS = [
    {"text": "第五章 第一幕", "href": "/ys/%E7%AC%AC%E4%BA%94%E7%AB%A0%E7%AC%AC%E4%B8%80%E5%B9%95", "url": "https://wiki.biligame.com/ys/%E7%AC%AC%E4%BA%94%E7%AB%A0%E7%AC%AC%E4%B8%80%E5%B9%95"},
    {"text": "间章 树海", "href": "/ys/%E9%97%B4%E7%AB%A0", "url": "https://wiki.biligame.com/ys/%E9%97%B4%E7%AB%A0"},
]
ALLOWED = {c["url"] for c in CANDS}

GOOD = {"chapters": [
    {"title": "第五章 第一幕", "url": CANDS[0]["url"], "kind": "主线"},
    {"title": "间章：树海", "url": CANDS[1]["url"], "kind": "间章"},
]}

# T1 抽取：正常 JSON
llm.llm_chat = lambda s, u: json.dumps(GOOD, ensure_ascii=False)
r = m.extract_chapters_llm(CANDS)
check("T1 抽取-正常响应", r == GOOD["chapters"])

# T2 抽取：```json 围栏
llm.llm_chat = lambda s, u: "```json\n" + json.dumps(GOOD, ensure_ascii=False) + "\n```"
r = m.extract_chapters_llm(CANDS)
check("T2 抽取-容忍代码围栏", r == GOOD["chapters"])

# T3 防幻觉：编造 url 被交叉核验丢弃
llm.llm_chat = lambda s, u: json.dumps({"chapters": [
    {"title": "假章节", "url": "https://wiki.biligame.com/ys/fake", "kind": "主线"}
]}, ensure_ascii=False)
r = m.extract_chapters_llm(CANDS)
check("T3 抽取-编造url被拦截(降级None)", r is None)

# T4 schema 违规：缺 title 字段 -> 重试后仍失败 -> None
calls = {"n": 0}
def bad(s, u):
    calls["n"] += 1
    return json.dumps({"chapters": [{"url": CANDS[0]["url"]}]}, ensure_ascii=False)
llm.llm_chat = bad
r = m.extract_chapters_llm(CANDS)
check("T4 抽取-schema违规重试后降级", r is None and calls["n"] == llm.LLM_MAX_RETRIES + 1, f"calls={calls['n']}")

# T5 重试：第一次坏 JSON，第二次好
seq = ["not-json{{", json.dumps(GOOD, ensure_ascii=False)]
llm.llm_chat = lambda s, u: seq.pop(0)
r = m.extract_chapters_llm(CANDS)
check("T5 抽取-重试后成功", r == GOOD["chapters"])

# T6 匹配：LLM 识别一条为改名变体
UNMATCHED = [{"title": "第五章 第二幕：新篇", "url": CANDS[0]["url"]}]
LOCAL = ["第五章 第二幕", "第五章 第一幕"]
llm.llm_chat = lambda s, u: json.dumps({"new_titles": []}, ensure_ascii=False)
r = m.match_new_chapters_llm(UNMATCHED, LOCAL)
check("T6 匹配-识别改名变体(过滤为空)", r == [])

# T7 匹配：LLM 确认为真新章节
llm.llm_chat = lambda s, u: json.dumps({"new_titles": ["第五章 第二幕：新篇"]}, ensure_ascii=False)
r = m.match_new_chapters_llm(UNMATCHED, LOCAL)
check("T7 匹配-确认真新章节", r == UNMATCHED)

# T8 匹配：LLM 编造标题 -> 整体不可信 -> None（保持规则结果）
llm.llm_chat = lambda s, u: json.dumps({"new_titles": ["编造的标题"]}, ensure_ascii=False)
r = m.match_new_chapters_llm(UNMATCHED, LOCAL)
check("T8 匹配-编造标题拦截", r is None)

# T9 无 key：纯规则模式
llm.LLM_API_KEY = ""
llm.llm_chat = lambda s, u: (_ for _ in ()).throw(AssertionError("不应调用 LLM"))
check("T9 抽取-无key直接None", m.extract_chapters_llm(CANDS) is None)
check("T9 匹配-无key直接None", m.match_new_chapters_llm(UNMATCHED, LOCAL) is None)

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
