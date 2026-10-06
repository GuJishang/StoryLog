#!/usr/bin/env python3
"""路径 B（ai_review.py）单测：mock LLM，验证目录解析/统计/校验/防幻觉/稳定性。"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import storylog_llm as llm_core
import ai_review as ar

FIXTURE = "tests/fixtures/fixture_export.json"
os.makedirs("tests/.tmp", exist_ok=True)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("PASS" if cond else "FAIL") + f"  {name}" + (f"  | {detail}" if detail and not cond else ""))


# 启用假 LLM
llm_core.LLM_API_KEY = "test-key"
llm_core.LLM_MAX_RETRIES = 1
llm_core.LLM_RETRY_BACKOFF = 0

# T1 目录解析：真实 index.html
cat = ar.build_catalog()
check("T1 目录-条目数>=100", len(cat) >= 100, f"got {len(cat)}")
p1 = cat.get("g-p-1", {})
check("T1 目录-g-p-1 归属原神", p1.get("game") == "原神", str(p1))
check("T1 目录-g-p-1 标题正确", "捕风的异乡人" in p1.get("title", ""), str(p1))

# T2 数据加载与统计（fixture，合成测试数据）
records = ar.load_records(FIXTURE, cat)
check("T2 加载-4条记录", len(records) == 4)
stats = ar.compute_stats(records)
check("T2 统计-评分3条", stats["rated_count"] == 3)
check("T2 统计-短评3条", stats["text_review_count"] == 3)
check("T2 统计-原神4条", stats["by_game"].get("原神") == 4)
check("T2 统计-活跃月份", stats["months_active"] == ["2026-08", "2026-09"], str(stats["months_active"]))

# T3 画像校验：合法通过
KNOWN = {"第一幕：捕风的异乡人", "第一幕：浮世浮生千岩间"}
GOOD = {"preference_tags": ["主题叙事", "群像塑造"], "profile": "用户偏好主题表达完整的章节，对意象与人物塑造敏感。" * 1,
        "highlights": [{"title": "第一幕：捕风的异乡人", "reason": "自由主题"}],
        "suggestions": ["补完稻妻后续"]}
check("T3 画像-合法通过", ar.validate_portrait(GOOD, KNOWN) is not None)

# T4 画像防幻觉：编造 highlight 标题 -> 整体拒绝
FAKE = dict(GOOD, highlights=[{"title": "不存在的章节", "reason": "编造"}])
check("T4 画像-编造标题拦截", ar.validate_portrait(FAKE, KNOWN) is None)
check("T4 画像-短profile拦截", ar.validate_portrait(dict(GOOD, profile="太短"), KNOWN) is None)
MIXED = dict(GOOD, highlights=[
    {"title": "第一幕：捕风的异乡人", "reason": "有效"},
    {"title": "编造的章节", "reason": "编造"}])
r_mixed = ar.validate_portrait(MIXED, KNOWN)
check("T4 画像-部分编造丢弃保留有效", r_mixed is not None and len(r_mixed["highlights"]) == 1, str(r_mixed))

# T5 情感校验：非法枚举/未知标题 -> 拒绝
check("T5 情感-合法通过", ar.validate_sentiment({"items": [
    {"title": "第一幕：捕风的异乡人", "sentiment": "positive"}]}, KNOWN) is not None)
check("T5 情感-非法枚举拦截", ar.validate_sentiment({"items": [
    {"title": "第一幕：捕风的异乡人", "sentiment": "great"}]}, KNOWN) is None)
check("T5 情感-编造标题拦截", ar.validate_sentiment({"items": [
    {"title": "编造", "sentiment": "positive"}]}, KNOWN) is None)

# T6 稳定性自检：固定响应 -> 高一致
llm_core.llm_chat = lambda s, u: json.dumps(GOOD, ensure_ascii=False)
level, a, b = ar.stability_check(records, stats)
check("T6 稳定性-固定响应判高", level.startswith("高"), level)

# T7 稳定性：两次完全不同 -> 低一致
seq = [json.dumps(dict(GOOD, preference_tags=["主题叙事"]), ensure_ascii=False),
       json.dumps(dict(GOOD, preference_tags=["战斗演出"]), ensure_ascii=False)]
llm_core.llm_chat = lambda s, u: seq.pop(0)
level, a, b = ar.stability_check(records, stats)
check("T7 稳定性-无重叠判低", level.startswith("低"), level)

# T8 降级：LLM 不可用 -> generate_portrait None，报告仍可渲染
llm_core.LLM_API_KEY = ""
p = ar.generate_portrait(records, stats)
check("T8 降级-画像None", p is None)
report = ar.render_report(records, stats, None, "failed", None, "无法评估", "fixture")
check("T8 降级-报告含降级说明", "降级" in report and "记录总览" in report)

# T9 情感偏差：高分负评被检出（mock）
llm_core.LLM_API_KEY = "test-key"
llm_core.llm_chat = lambda s, u: json.dumps({"items": [
    {"title": "第一幕：捕风的异乡人", "sentiment": "negative"},
    {"title": "第一幕：浮世浮生千岩间", "sentiment": "positive"}]}, ensure_ascii=False)
status, dev = ar.analyze_sentiment_bias(records)
check("T9 偏差-状态ok", status == "ok")
check("T9 偏差-高分负评检出", dev and any(d["title"] == "第一幕：捕风的异乡人" for d in dev), str(dev))

# T10 报告状态区分：无偏差 ≠ 语料不足
r_ok = ar.render_report(records, stats, GOOD, "ok", [], "高", "fixture")
check("T10 报告-无偏差表述正确", "未发现明显偏差" in r_ok)
r_skip = ar.render_report(records, stats, GOOD, "skipped", None, "高", "fixture")
check("T10 报告-语料不足表述正确", "无足够长文语料" in r_skip)
check("T10 报告-含模型名", llm_core.LLM_MODEL in r_ok)

# T11 main 端到端（mock LLM 按 prompt 路由）：覆盖解包顺序与状态输出
import contextlib
import io

def fake_chat(system, user):
    if "情感标注" in system:
        return json.dumps({"items": [
            {"title": "第一幕：捕风的异乡人", "sentiment": "positive"},
            {"title": "第一幕：浮世浮生千岩间", "sentiment": "neutral"},
            {"title": "第一幕：不动鸣神，恒常乐土", "sentiment": "negative"}]}, ensure_ascii=False)
    return json.dumps(GOOD, ensure_ascii=False)

llm_core.llm_chat = fake_chat
_saved_argv = sys.argv
sys.argv = ["ai_review.py", "--data", FIXTURE,
            "--out", "tests/.tmp/t11_report.md", "--no-stability"]
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ar.main()
sys.argv = _saved_argv
out = buf.getvalue()
check("T11 main-情感分析状态为完成", "情感偏差分析: 完成" in out, out.strip().replace("\n", " | "))
check("T11 main-画像为LLM生成", "画像: LLM 生成" in out, out)
t11 = open("tests/.tmp/t11_report.md", encoding="utf-8").read()
check("T11 main-报告无偏差表述", "未发现明显偏差" in t11)
check("T11 main-报告不含失败提示", "LLM 调用失败" not in t11)

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
