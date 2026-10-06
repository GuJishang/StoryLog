#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""路径 C（RAG）单测：分词 / 切分 / BM25 检索 / 上下文编号 / 引用防幻觉 / 多档降级。

LLM 层被 monkeypatch 到共享模块 storylog_llm 上。
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import storylog_llm as llm
import storylog_rag as sr

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("PASS" if cond else "FAIL") + f"  {name}" + (f"  | {detail}" if detail and not cond else ""))


# ============================================================
# T1 分词：中文 bigram + 拉丁词
# ============================================================
toks = sr.tokenize("原神第五章")
check("T1 分词-中文bigram", "原神" in toks and "神第" in toks and "五章" in toks, str(toks))
check("T1 分词-单字保留", sr.tokenize("神") == ["神"], str(sr.tokenize("神")))
check("T1 分词-拉丁词", "abc123" in sr.tokenize("测试 ABC123 结束"), str(sr.tokenize("测试 ABC123")))
check("T1 分词-空输入", sr.tokenize("") == [] and sr.tokenize(None) == [])


# ============================================================
# T2 切分
# ============================================================
text = "\n".join([f"这是第{i}段剧情内容，讲述旅行者的冒险。" for i in range(12)])
parts = sr.split_text(text, size=80, overlap=20)
check("T2 切分-产出多块", len(parts) > 1, f"n={len(parts)}")
check("T2 切分-每块不超长太多", all(len(p) <= 160 for p in parts), str([len(p) for p in parts]))
long_para = "字" * 500
hard = sr.split_text(long_para, size=100, overlap=20)
check("T2 切分-超长段落硬切", len(hard) >= 5, f"n={len(hard)}")
check("T2 切分-短输入过滤", sr.split_text("短") == [] and sr.split_text("") == [])


# ============================================================
# T3 HTML 清洗
# ============================================================
html = """
<div class="mw-parser-output">
  <script>var x=1;</script>
  <table><tr><td>导航表格</td></tr></table>
  <div class="navbox">导航盒子</div>
  <p>第四章 第一幕：真实剧情内容这里。</p>
  <p>编辑</p>
  <p>第四章 第一幕：真实剧情内容这里。</p>
</div>
"""
cleaned = sr.html_to_text(html)
check("T3 清洗-去script/table/navbox", "var x" not in cleaned and "导航" not in cleaned, cleaned[:80])
check("T3 清洗-保留正文", "真实剧情内容" in cleaned, cleaned[:80])
check("T3 清洗-去样板行", "编辑" not in cleaned.split("\n"), cleaned)
check("T3 清洗-去重复行", cleaned.count("真实剧情内容") == 1, cleaned)
check("T3 清洗-空输入", sr.html_to_text("") == "" and sr.html_to_text(None) == "")


# ============================================================
# 构造测试索引
# ============================================================
CHUNKS = [
    {"id": "c1", "source_key": "k1", "game": "genshin", "gameName": "原神",
     "version": "", "chapter": "第五章 第一幕", "url": "u1", "kind": "page",
     "text": "旅行者与派蒙来到枫丹，遇到了芙宁娜。水神芙卡洛斯的故事在此展开。"},
    {"id": "c2", "source_key": "k2", "game": "genshin", "gameName": "原神",
     "version": "", "chapter": "第五章 第二幕", "url": "u2", "kind": "page",
     "text": "钟离是璃月的岩神摩拉克斯，掌管契约。他与旅行者讨论了归终的计划。"},
    {"id": "c3", "source_key": "k3", "game": "pns", "gameName": "战双帕弥什",
     "version": "", "chapter": "41 长路归航", "url": "u3", "kind": "page",
     "text": "阿尔法与先遣队的故事，指挥官在长路归航中面对艰难抉择。"},
]
df = sr.build_bm25(CHUNKS)
lengths = [c["len"] for c in CHUNKS]
INDEX = {"chunks": CHUNKS, "df": dict(df), "nDocs": len(CHUNKS),
         "avgdl": sum(lengths) / len(lengths)}


# ============================================================
# T4 BM25 检索
# ============================================================
hits = sr.search(INDEX, "钟离是谁")
check("T4 检索-命中正确块", hits and hits[0][1]["id"] == "c2", str([(s, c["id"]) for s, c in hits]))
hits = sr.search(INDEX, "阿尔法 先遣队")
check("T4 检索-跨游戏命中", hits and hits[0][1]["id"] == "c3", str([(s, c["id"]) for s, c in hits]))
hits = sr.search(INDEX, "钟离", game="pns")
check("T4 检索-游戏过滤", all(c["game"] == "pns" for _s, c in hits), str(hits))
check("T4 检索-无命中", sr.search(INDEX, "完全不相干的量子力学内容") == [])
check("T4 检索-空索引", sr.search({"chunks": []}, "钟离") == [])
check("T4 检索-空查询", sr.search(INDEX, "") == [])
check("T4 检索-章节名加权", sr.search(INDEX, "归航")[0][1]["id"] == "c3")


# ============================================================
# T5 上下文编号
# ============================================================
ctx, cid_map = sr.build_context([(1.0, CHUNKS[0]), (0.5, CHUNKS[1])])
check("T5 上下文-含编号", ctx.startswith("[1]") and "[2]" in ctx, ctx[:60])
check("T5 上下文-编号映射", set(cid_map) == {1, 2} and cid_map[1]["id"] == "c1")
ctx2, cid2 = sr.build_context([(1.0, CHUNKS[i]) for i in range(3)], max_chars=120)
check("T5 上下文-长度截断", len(cid2) < 3, f"n={len(cid2)}")


# ============================================================
# T6 答案校验（引用防幻觉）
# ============================================================
cm = {1: CHUNKS[0], 2: CHUNKS[1]}
check("T6 校验-过短答案拒绝", sr.validate_answer({"answer": "短"}, cm) is None)
check("T6 校验-非字典拒绝", sr.validate_answer(["x"], cm) is None)
ok = sr.validate_answer({"answer": "钟离是岩神摩拉克斯，掌管契约。[2]", "citations": [2], "confidence": "high"}, cm)
check("T6 校验-正常通过", ok and ok["citations"] == [2] and ok["confidence"] == "high", str(ok))
bad = sr.validate_answer({"answer": "依据来自[1]和[99]两处。", "citations": [1, 99], "confidence": "very-high"}, cm)
check("T6 校验-越界引用被丢弃", bad and bad["citations"] == [1] and bad["dropped"] == 1, str(bad))
check("T6 校验-答案内越界标号剔除", bad and "[99]" not in bad["answer"], bad["answer"] if bad else "")
check("T6 校验-置信度归一", bad and bad["confidence"] == "medium", str(bad))


# ============================================================
# T7 answer_question：多档模式
# ============================================================
llm.LLM_API_KEY = ""
res = sr.answer_question("钟离是谁", INDEX)
check("T7 问答-无LLM降级search_only", res["status"] == "ok" and res["mode"] == "search_only", str(res["mode"]))
check("T7 问答-降级仍带引用", len(res["citations"]) > 0 and res["citations"][0]["n"] == 1, str(res["citations"])[:80])

llm.LLM_API_KEY = "test-key"
llm.LLM_MAX_RETRIES = 1
llm.LLM_RETRY_BACKOFF = 0
llm.llm_chat = lambda s, u: json.dumps(
    {"answer": "钟离即岩神摩拉克斯，掌管契约。[1]", "citations": [1], "confidence": "high"},
    ensure_ascii=False)
res = sr.answer_question("钟离是谁", INDEX)
check("T7 问答-LLM正常", res["status"] == "ok" and res["mode"] == "llm", str(res["mode"]))
check("T7 问答-引用白名单映射", res["citations"] and res["citations"][0]["id"] in ("c1", "c2"), str(res["citations"])[:100])

# LLM 编造不存在的引用 -> 被丢弃，答案仍返回
llm.llm_chat = lambda s, u: json.dumps(
    {"answer": "钟离是岩神，[7]可证。", "citations": [7], "confidence": "high"}, ensure_ascii=False)
res = sr.answer_question("钟离是谁", INDEX)
check("T7 问答-编造引用被丢弃", res["mode"] == "llm" and res["citations"] == [] and res["dropped"] >= 1, str(res)[:120])

# LLM 全失败 -> 降级 degraded（而不是抛异常）
llm.llm_chat = lambda s, u: (_ for _ in ()).throw(RuntimeError("boom"))
res = sr.answer_question("钟离是谁", INDEX)
check("T7 问答-LLM失败降级degraded", res["status"] == "ok" and res["mode"] == "degraded", str(res["mode"]))

check("T7 问答-空问题", sr.answer_question("   ", INDEX)["status"] == "no_hits")
check("T7 问答-空索引", sr.answer_question("钟离", {"chunks": []})["status"] == "empty_index")
check("T7 问答-无命中", sr.answer_question("量子纠缠退相干", INDEX)["status"] == "no_hits")


# ============================================================
# T8 索引元信息与 id 稳定性
# ============================================================
check("T8 id-稳定", sr.chunk_id("genshin", "k1", 0) == sr.chunk_id("genshin", "k1", 0))
check("T8 id-可区分", sr.chunk_id("genshin", "k1", 0) != sr.chunk_id("genshin", "k1", 1))
check("T8 hash-内容变化即变", sr.content_hash("a") != sr.content_hash("b"))
st = sr.index_stats(INDEX)
check("T8 统计-字段完整", st["chunks"] == 3 and st["pages"] == 3 and st["games"]["原神"] == 2, str(st))

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
