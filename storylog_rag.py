#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""剧情志 · 剧情问答 RAG 内核（路径 C）

三层管线
--------
  索引层  fetch_pages() -> html_to_text() -> split_text() -> build_bm25() -> .workbuddy/rag_index.json
  检索层  tokenize(中文 bigram) + BM25 打分召回 Top-K，可选 LLM 重排
  生成层  LLM 严格基于检索片段作答，引用编号必须落在检索结果内（防幻觉），
          失败自动降级为「纯检索摘要」，保证无 LLM 时依然能给出可读结果

设计取舍（为什么自己写检索）
----------------------------
实测 Moonshot /v1/embeddings 返回 permission_denied（不开放向量接口），
因此检索层采用「中文 bigram + BM25」零依赖方案：不需要分词库、不需要向量库、
不需要联网就能检索。同时保留 OpenAI 兼容向量后端（STORYLOG_EMBED_* 环境变量），
接口可用时启用「BM25 + 向量」混合检索，不可用则自动退回纯 BM25。

输出契约（与其它检测脚本风格对齐）
----------------------------------
  索引：INDEX_BUILT / INDEX_REUSED / INDEX_ERROR
  问答：ANSWER_MODE: llm | llm_rerank | search_only | degraded

Env vars
--------
  LLM_API_KEY / LLM_API_BASE / LLM_MODEL / LLM_TIMEOUT   复用 check_ys_wiki 的 LLM 层
  STORYLOG_EMBED_API_KEY  / STORYLOG_EMBED_API_BASE / STORYLOG_EMBED_MODEL
      可选向量后端；不配置则纯 BM25

Usage
-----
  python storylog_rag.py --build                      # 抓正文并构建索引
  python storylog_rag.py --build --game genshin --max-pages 6
  python storylog_rag.py --build --refresh            # 忽略缓存，全量重抓
  python storylog_rag.py --status                     # 查看索引状态
  python storylog_rag.py --search "钟离是谁"           # 纯检索
  python storylog_rag.py --ask "第五章讲了什么"         # 检索 + LLM 生成
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from urllib.parse import unquote

import requests
from bs4 import BeautifulSoup

from storylog_common import (
    BROWSER_HEADERS,
    GAME_NAMES,
    GAME_ORDER,
    make_session,
    read_catalog,
)
from storylog_llm import llm_available, llm_json_with_retry

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_DIR = os.path.join(SCRIPT_DIR, ".workbuddy")
# 索引文件路径可用环境变量覆盖：测试可用独立索引，避免污染生产索引
INDEX_FILE = os.environ.get("STORYLOG_INDEX_FILE") or os.path.join(STATE_DIR, "rag_index.json")

# 复用带浏览器请求头的 Session（部分 bwiki 子站对不完整请求头返回 567 风控页）
SESSION = make_session(referer="https://wiki.biligame.com/")

INDEX_VERSION = 1
CHUNK_SIZE = 420            # 单块目标字数
CHUNK_OVERLAP = 80          # 邻块重叠字数（保持上下文连续）
MIN_CHUNK_CHARS = 12        # 过短的块丢弃
DEFAULT_MAX_PAGES = 60      # 每次构建「每款游戏」抓取的正文页数上限（覆盖三游戏全量章节）
FETCH_DELAY = 0.5           # 页面之间间隔秒数
FETCH_TIMEOUT = 20
DEFAULT_TOPK = 6
MAX_CONTEXT_CHARS = 4200    # 送入 LLM 的片段总长度上限
BM25_K1 = 1.5
BM25_B = 0.75

# 可选向量后端（OpenAI 兼容 /embeddings）
EMBED_API_KEY = os.environ.get("STORYLOG_EMBED_API_KEY", "").strip()
EMBED_API_BASE = os.environ.get("STORYLOG_EMBED_API_BASE", "").rstrip("/")
EMBED_MODEL = os.environ.get("STORYLOG_EMBED_MODEL", "text-embedding-3-small")


# ============================================================
# 分词与文本处理
# ============================================================

_CJK_SEG_RE = re.compile(r"[\u4e00-\u9fff]+")
_LATIN_RE = re.compile(r"[a-zA-Z0-9]{2,}")


def tokenize(text):
    """零依赖中文分词：CJK 连续段做字符 bigram，拉丁字母/数字按词切。

    之所以用 bigram：中文没有空格，不引入 jieba 之类的分词库时，
    字符 bigram 的检索召回显著优于单字（单字噪声大），且实现完全确定。
    """
    text = (text or "").lower()
    tokens = []
    for seg in _CJK_SEG_RE.findall(text):
        if len(seg) == 1:
            tokens.append(seg)
        else:
            tokens.extend(seg[i:i + 2] for i in range(len(seg) - 1))
    tokens.extend(_LATIN_RE.findall(text))
    return tokens


_DROP_SELECTORS = (
    "table", "script", "style", ".navbox", ".toc", "#toc", ".mw-editsection",
    ".nomobile", ".noprint", ".mw-collapsible-toggle", ".mw-empty-elt",
    ".reference", "sup.reference", ".mw-references-wrap", ".infobox",
    ".wikitable", ".navbox-inner", ".printfooter", ".catlinks",
)
_BOILERPLATE_RE = re.compile(
    r"^(编辑|查看|讨论|历史|刷新|上一页|下一页|导航|目录|返回顶部|展开|折叠|"
    r"跳转至|页面内容|本页面|本WIKI|MediaWiki|版权|免责声明|关于我们|隐私政策)$"
)
# wiki 页面模板/运营提示语：在各章节页重复出现，属噪声而非剧情内容
_TEMPLATE_NOTICE_RE = re.compile(
    r"出于方便回顾剧情的目的|我们自己收集编写了战双的剧情回顾"
    r"|人为编写，难免出错|希望指挥官能帮忙参与编辑|反馈留言板"
    r"|可以收藏随时查看更新|觉得WIKI好玩的话|请推荐给朋友"
    r"|完整关卡数|本WIKI的全部内容|编辑本页|最后编辑"
)


def html_to_text(html):
    """把 wiki 页面 HTML 清洗成纯文本正文。失败时返回空串（由调用方记录错误）。"""
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    root = soup.find("div", class_="mw-parser-output") or soup.find("body") or soup
    for sel in _DROP_SELECTORS:
        try:
            for el in root.select(sel):
                el.decompose()
        except Exception:
            continue

    lines, seen = [], set()
    for raw in root.get_text("\n").splitlines():
        line = re.sub(r"[ \t\u00a0]+", " ", raw).strip()
        if not line or _BOILERPLATE_RE.match(line):
            continue
        if len(line) <= 2:                 # 吞掉零碎残留（单字导航等）
            continue
        if line in seen:                   # 同一页内的重复行（模板重复渲染）
            continue
        if _TEMPLATE_NOTICE_RE.search(line):   # 页面模板/运营提示语（跨页重复噪声）
            continue
        seen.add(line)
        lines.append(line)
    return "\n".join(lines)


def split_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """段落优先的切分：先按段落聚合成目标长度，超长段落硬切，邻块保留重叠。"""
    text = (text or "").strip()
    if not text:
        return []
    paras = [p.strip() for p in re.split(r"\n+", text) if p.strip()]

    chunks, buf = [], ""
    for p in paras:
        if len(p) > size:                          # 超长段落：硬切
            if buf:
                chunks.append(buf)
                buf = ""
            step = max(1, size - overlap)
            for i in range(0, len(p), step):
                seg = p[i:i + size]
                if len(seg) >= MIN_CHUNK_CHARS:
                    chunks.append(seg)
                if i + size >= len(p):
                    break
            continue
        if len(buf) + len(p) + 1 <= size:
            buf = (buf + "\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)

    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]


def content_hash(text):
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()[:12]


def chunk_id(game, source_key, idx):
    """稳定的块 id：重建索引后同一块 id 不变，便于引用与会话缓存。"""
    return hashlib.md5(f"{game}|{source_key}|{idx}".encode("utf-8")).hexdigest()[:10]


# ============================================================
# 数据源：目录（index.html）解析
# ============================================================
# 目录解析（游戏 -> 版本 -> 章节）统一由 storylog_common.read_catalog() 提供，
# 避免与检测脚本各自实现一套而出现口径不一致。


def catalog_pages():
    """把目录转成「目录型页面」，与正文页同构，供统一切分入库。"""
    pages = []
    for v in read_catalog():
        names = "、".join(e["title"] for e in v["episodes"]) or "（暂无章节）"
        text = (
            f"《{v['gameName']}》{v['versionTitle']}（{v['versionLabel']}）"
            f"收录 {len(v['episodes'])} 个章节：{names}。"
        )
        pages.append({
            "game": v["game"],
            "gameName": v["gameName"],
            "version": v["versionTitle"],
            "versionLabel": v["versionLabel"],
            "chapter": v["versionTitle"],
            "url": "",
            "pages": [],                    # 该目录下可抓正文的章节
            "kind": "catalog",
            "text": text,
            "source_key": f"{v['game']}:catalog:{v['versionId']}",
        })
    return pages


# ============================================================
# 数据源：wiki 正文抓取（尽力而为，失败记录不阻塞）
# ============================================================

def mediawiki_plaintext(api_url, page_url):
    """MediaWiki：用 action=parse 取渲染后正文，再清洗为纯文本。"""
    title = unquote(page_url.rstrip("/").split("/")[-1])
    resp = SESSION.get(
        api_url,
        params={
            "action": "parse", "page": title, "prop": "text",
            "format": "json", "redirects": 1, "disableeditsection": 1,
        },
        timeout=FETCH_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"parse error: {data['error'].get('info', '')}")
    html = (data.get("parse") or {}).get("text", {}).get("*", "")
    return html_to_text(html)


def _fetch_genshin(max_pages):
    """原神：bwiki 魔神任务目录页 -> 章节链接 -> 逐页抓正文。"""
    import check_ys_wiki as ys
    html = ys.get_wiki_html()
    seen, targets = set(), []
    for c in ys.get_wiki_chapters_rules(html):
        if c["url"] in seen:
            continue
        seen.add(c["url"])
        targets.append(c)
    pages = []
    for c in targets[:max_pages]:
        try:
            text = mediawiki_plaintext(ys.WIKI_API, c["url"])
            pages.append({
                "game": "genshin", "gameName": GAME_NAMES["genshin"],
                "version": "", "chapter": c["title"], "url": c["url"],
                "source_key": f"genshin:page:{content_hash(c['url'])}",
                "kind": "page", "text": text,
            })
        except Exception as e:
            pages.append(_error_page("genshin", c["title"], c["url"], e))
        time.sleep(FETCH_DELAY)
    return pages


def _fetch_pns(max_pages):
    """战双帕弥什：bwiki 剧情回顾页 -> 三个板块 -> 逐章节抓正文。"""
    import check_pns_wiki as pns
    raw = pns.get_wiki_html()
    sections = pns.get_sections(raw)
    tracked = ["主线剧情", "浮点纪实", "外篇剧情"]
    pages, budget = [], max_pages
    for name in tracked:
        for item in sections.get(name) or []:
            if budget <= 0:
                break
            budget -= 1
            try:
                text = mediawiki_plaintext(pns.WIKI_API, item["url"])
                pages.append({
                    "game": "pns", "gameName": GAME_NAMES["pns"],
                    "version": name, "chapter": item["title"], "url": item["url"],
                    "source_key": f"pns:page:{content_hash(item['url'])}",
                    "kind": "page", "text": text,
                })
            except Exception as e:
                pages.append(_error_page("pns", item["title"], item["url"], e))
            time.sleep(FETCH_DELAY)
        if budget <= 0:
            break
    return pages


def _fetch_wuwa(max_pages):
    """鸣潮：库街区 wiki 目录卡片 -> 逐词条取详情正文（剧情对白）。

    库街区站点是 SPA，但词条正文有公开 JSON 接口
    （见 check_wuwa_wiki.fetch_entry_detail：POST /catalogue/item/getEntryDetail，参数 id）。
    """
    import check_wuwa_wiki as wuwa
    cards = wuwa.fetch_cards()
    pages, errors = [], []
    budget = max_pages
    for c in cards:
        if budget <= 0:
            break
        eid = c.get("entryId")
        if not eid:
            continue
        budget -= 1
        try:
            entry = wuwa.fetch_entry_detail(eid)
            parts = wuwa.extract_story_html(entry)
            text = html_to_text("\n".join(parts)) if parts else ""
            if not text.strip():
                raise RuntimeError("该词条无剧情正文（资料/目录页）")
            pages.append({
                "game": "wuwa", "gameName": GAME_NAMES["wuwa"],
                "version": str(entry.get("currentVersion") or ""),
                "chapter": c["name"],
                "url": getattr(wuwa, "WIKI_PAGE_URL", ""),
                "source_key": f"wuwa:page:{content_hash(str(eid))}",
                "kind": "page", "text": text,
            })
        except Exception as e:
            errors.append({
                "game": "wuwa", "chapter": c.get("name"),
                "url": getattr(wuwa, "WIKI_PAGE_URL", ""),
                "error": f"{type(e).__name__}: {e}",
            })
        time.sleep(FETCH_DELAY)
    if not pages:
        # 全部失败时的兜底：至少保留目录卡片，避免鸣潮检索完全为空
        names = "、".join(c["name"] for c in cards)
        pages.append({
            "game": "wuwa", "gameName": GAME_NAMES["wuwa"],
            "version": "", "chapter": "剧情目录", "url": wuwa.WIKI_PAGE_URL,
            "source_key": "wuwa:catalog:cards", "kind": "catalog",
            "text": f"《鸣潮》剧情目录（库街区 wiki）共收录 {len(cards)} 个条目：{names}。",
        })
    return pages, errors


def _error_page(game, chapter, url, exc):
    return {
        "game": game, "gameName": GAME_NAMES.get(game, game),
        "version": "", "chapter": chapter, "url": url,
        "source_key": f"{game}:error:{content_hash(url)}",
        "kind": "error", "text": "",
        "error": f"{type(exc).__name__}: {exc}",
    }


def fetch_pages(games=None, max_pages=DEFAULT_MAX_PAGES):
    """抓取全部数据源，返回 (pages, errors)。任何单个源失败都不影响整体。"""
    games = games or list(GAME_ORDER)
    pages, errors = [], []

    for gid in games:
        try:
            if gid == "genshin":
                got = _fetch_genshin(max_pages)
            elif gid == "pns":
                got = _fetch_pns(max_pages)
            elif gid == "wuwa":
                got, extra_err = _fetch_wuwa(max_pages)
                errors.extend(extra_err)
            else:
                continue
            pages.extend(got)
        except Exception as e:
            errors.append({"game": gid, "error": f"{type(e).__name__}: {e}"})

    for p in pages:
        if p.get("kind") == "error":
            errors.append({"game": p["game"], "chapter": p.get("chapter"),
                           "url": p.get("url"), "error": p.get("error")})
    pages = [p for p in pages if p.get("kind") != "error"]
    return pages, errors


# ============================================================
# BM25 索引
# ============================================================

def build_bm25(chunks):
    """给每个块算词频与长度，并统计文档频率。"""
    df = Counter()
    for c in chunks:
        # 章节名参与检索（标题命中通常强相关），权重 ×2
        toks = tokenize(c["text"]) + tokenize(c.get("chapter", "")) * 2
        tf = Counter(toks)
        c["tf"] = {t: n for t, n in tf.items()}
        c["len"] = sum(tf.values()) or 1
        for t in tf:
            df[t] += 1
    return df


def bm25_score(q_tokens, chunk, df, n_docs, avgdl):
    score = 0.0
    tf = chunk.get("tf") or {}
    dl = chunk.get("len") or 1
    for t, qf in q_tokens.items():
        f = tf.get(t)
        if not f:
            continue
        n_q = df.get(t, 0)
        idf = math.log(1 + (n_docs - n_q + 0.5) / (n_q + 0.5))
        denom = f + BM25_K1 * (1 - BM25_B + BM25_B * dl / avgdl)
        score += idf * (f * (BM25_K1 + 1) / denom) * (1 + math.log(qf))
    return score


def search(index, query, topk=DEFAULT_TOPK, game=None):
    """BM25 召回，返回 [(score, chunk)]。"""
    chunks = index.get("chunks") or []
    if not chunks:
        return []
    df = Counter(index.get("df") or {})
    n_docs = index.get("nDocs") or len(chunks)
    avgdl = index.get("avgdl") or 1.0
    q = Counter(tokenize(query))
    if not q:
        return []
    scored = []
    for c in chunks:
        if game and c.get("game") != game:
            continue
        s = bm25_score(q, c, df, n_docs, avgdl)
        if s > 0:
            scored.append((s, c))
    scored.sort(key=lambda x: -x[0])
    return scored[:topk]


def index_stats(index):
    chunks = index.get("chunks") or []
    games = Counter(c.get("game") for c in chunks)
    return {
        "chunks": len(chunks),
        "pages": sum(1 for c in chunks if c.get("kind") == "page"),
        "catalog": sum(1 for c in chunks if c.get("kind") == "catalog"),
        "games": {GAME_NAMES.get(g, g): n for g, n in games.items()},
        "builtAt": index.get("builtAt"),
        "errors": len(index.get("errors") or []),
        "embed": bool(index.get("embedModel")),
    }


def save_index(index):
    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
    tmp = INDEX_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    os.replace(tmp, INDEX_FILE)


def load_index():
    if not os.path.exists(INDEX_FILE):
        return None
    try:
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def build_index(games=None, max_pages=DEFAULT_MAX_PAGES, refresh=False, progress=None):
    """构建索引：复用已抓过的页面（除非 refresh），目录内容每次重建。"""
    def say(msg):
        if progress:
            progress(msg)
        print(msg, flush=True)

    old = None if refresh else load_index()
    old_pages = {}
    if old:
        for p in old.get("_pages") or []:
            old_pages[p.get("source_key")] = p

    say("抓取数据源…")
    pages, errors = fetch_pages(games=games, max_pages=max_pages)
    pages.extend(catalog_pages())

    # 增量复用：同一 source_key 且内容未变 -> 复用旧块，省去重新切分
    reused = 0
    chunks, page_meta = [], []
    for p in pages:
        key = p["source_key"]
        h = content_hash(p["text"])
        prev = old_pages.get(key)
        if prev and prev.get("hash") == h:
            cached = [c for c in (old.get("chunks") or []) if c.get("source_key") == key]
            if cached:
                chunks.extend(cached)
                page_meta.append({"source_key": key, "hash": h, "chapters": len(cached),
                                  "kind": p["kind"], "game": p["game"], "title": p.get("chapter")})
                reused += 1
                continue
        parts = split_text(p["text"]) if p["kind"] == "page" else [p["text"]]
        for i, part in enumerate(parts):
            chunks.append({
                "id": chunk_id(p["game"], key, i),
                "source_key": key,
                "game": p["game"],
                "gameName": p["gameName"],
                "version": p.get("version", ""),
                "chapter": p.get("chapter", ""),
                "url": p.get("url", ""),
                "kind": p["kind"],
                "text": part,
            })
        page_meta.append({"source_key": key, "hash": h, "chapters": len(parts),
                          "kind": p["kind"], "game": p["game"], "title": p.get("chapter")})

    df = build_bm25(chunks)
    lengths = [c["len"] for c in chunks] or [1]
    index = {
        "version": INDEX_VERSION,
        "builtAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": "bm25-bigram",
        "nDocs": len(chunks),
        "avgdl": sum(lengths) / len(lengths),
        "df": dict(df),
        "chunks": chunks,
        "_pages": page_meta,
        "errors": errors,
    }
    save_index(index)

    st = index_stats(index)
    say(f"INDEX_BUILT chunks={st['chunks']} pages={st['pages']} catalog={st['catalog']} "
        f"reused={reused} errors={st['errors']}")
    if errors:
        for e in errors[:5]:
            say(f"NOTE: 抓取失败 {e.get('game')} {e.get('chapter') or ''} -> {e.get('error')}")
    return index


# ============================================================
# 生成层：LLM 基于片段作答（引用核验防幻觉）
# ============================================================

ANSWER_SYSTEM = (
    "你是《原神》《鸣潮》《战双帕弥什》的剧情问答助手。"
    "你只能依据给定的参考片段作答，禁止使用片段之外的知识，禁止编造人名、地名、剧情。"
    "只输出 JSON，不输出任何解释性文字。"
)

ANSWER_USER_TMPL = """参考片段（编号即引用号）：
{context}

用户问题：{question}

要求：
1. answer：依据片段作答，2-5 句，可在句中用 [编号] 标注依据，编号必须来自上面的片段。
2. citations：实际用到的片段编号数组，如 [1,3]；没有可用依据时输出 []。
3. confidence：high / medium / low 三选一；片段不足以回答时用 low，并在 answer 中说明缺少什么。
4. 只输出 JSON：{{"answer": "...", "citations": [1], "confidence": "high"}}
"""


def build_context(hits, max_chars=MAX_CONTEXT_CHARS):
    """把命中片段拼成带编号的上下文；同时返回编号 -> chunk 的映射。"""
    parts, used, cid_map = [], 0, {}
    for i, (_score, c) in enumerate(hits, 1):
        block = f"[{i}] 《{c['gameName']}》{c.get('chapter') or ''}\n{c['text']}"
        if used + len(block) > max_chars and parts:
            break
        used += len(block)
        parts.append(block)
        cid_map[i] = c
    return "\n\n".join(parts), cid_map


def validate_answer(data, cid_map):
    """校验 LLM 答案：结构、长度、引用白名单（防幻觉）。

    与路径 A/B 一致：非法引用「丢弃 + 提示」而不是全盘否决，
    只有当答案本身不可用时才返回 None 触发重试/降级。
    """
    if not isinstance(data, dict):
        return None
    answer = data.get("answer")
    if not isinstance(answer, str) or len(answer.strip()) < 8:
        return None

    cites, dropped = [], 0
    for c in data.get("citations") or []:
        try:
            n = int(c)
        except (TypeError, ValueError):
            dropped += 1
            continue
        if n in cid_map:
            if n not in cites:
                cites.append(n)
        else:
            dropped += 1

    # 答案正文里出现的 [n] 若越界则剔除，避免凭空引用不存在的片段
    def _fix(m):
        n = int(m.group(1))
        return m.group(0) if n in cid_map else ""

    answer = re.sub(r"\[(\d+)\]", _fix, answer)

    conf = str(data.get("confidence", "")).lower()
    if conf not in ("high", "medium", "low"):
        conf = "medium"

    return {
        "answer": answer.strip(),
        "citations": cites,
        "confidence": conf,
        "dropped": dropped,
    }


def answer_question(question, index, topk=DEFAULT_TOPK, game=None, rerank=False):
    """完整问答链路。任何环节失败都会降级，绝不空转。

    返回 {"status", "mode", "answer", "citations", "hits", "note"}
      status: ok | no_hits | empty_index
      mode:   llm | search_only | degraded
    """
    question = (question or "").strip()
    if not question:
        return {"status": "no_hits", "mode": "search_only", "answer": "", "citations": [],
                "hits": [], "note": "问题为空"}

    chunks = (index or {}).get("chunks") or []
    if not chunks:
        return {"status": "empty_index", "mode": "search_only", "answer": "",
                "citations": [], "hits": [],
                "note": "索引为空，请先构建索引（--build）"}

    hits = search(index, question, topk=topk, game=game)
    if not hits:
        return {"status": "no_hits", "mode": "search_only", "answer": "",
                "citations": [], "hits": [],
                "note": "本地索引中没有匹配到相关内容"}

    context, cid_map = build_context(hits)
    hit_payload = [
        {"n": n, "id": c["id"], "game": c["gameName"], "chapter": c.get("chapter"),
         "url": c.get("url"), "score": round(s, 3), "excerpt": c["text"][:160]}
        for n, (s, c) in enumerate(hits, 1) if n in cid_map
    ]

    if llm_available():
        user = ANSWER_USER_TMPL.format(context=context, question=question)
        data = llm_json_with_retry(ANSWER_SYSTEM, user, lambda d: validate_answer(d, cid_map))
        if data is not None:
            return {
                "status": "ok", "mode": "llm",
                "answer": data["answer"],
                "citations": [{"n": n, **{k: cid_map[n][k] for k in ("id", "gameName", "chapter", "url")}}
                              for n in data["citations"]],
                "confidence": data["confidence"],
                "dropped": data["dropped"],
                "hits": hit_payload,
                "note": f"已丢弃 {data['dropped']} 个越界引用" if data["dropped"] else "",
            }

    # 降级：纯检索摘要（无 LLM / LLM 失败）
    mode = "degraded" if llm_available() else "search_only"
    note = "LLM 不可用或调用失败，已降级为检索摘要" if llm_available() else "未配置 LLM，仅返回检索结果"
    lines = []
    for n, (_s, c) in enumerate(hits[:5], 1):
        snippet = re.sub(r"\s+", " ", c["text"])[:110]
        lines.append(f"[{n}] 《{c['gameName']}》{c.get('chapter') or ''}：{snippet}…")
    return {
        "status": "ok", "mode": mode,
        "answer": "根据本地剧情库检索到以下相关片段：\n" + "\n".join(lines),
        "citations": [{"n": n, **{k: c[k] for k in ("id", "gameName", "chapter", "url")}}
                      for n, (_s, c) in enumerate(hits[:5], 1)],
        "confidence": "low",
        "hits": hit_payload,
        "note": note,
    }


# ============================================================
# CLI
# ============================================================

def _print_hits(hits):
    for n, (score, c) in enumerate(hits, 1):
        head = f"[{n}] {score:.2f}  《{c['gameName']}》{c.get('chapter') or ''}"
        print(head)
        print("    " + re.sub(r"\s+", " ", c["text"])[:140] + "…")


def main():
    ap = argparse.ArgumentParser(description="剧情志 · 剧情问答 RAG")
    ap.add_argument("--build", action="store_true", help="构建/增量更新索引")
    ap.add_argument("--status", action="store_true", help="查看索引状态")
    ap.add_argument("--ask", metavar="QUESTION", help="提问（检索 + LLM 生成）")
    ap.add_argument("--search", metavar="QUERY", help="纯检索")
    ap.add_argument("--game", choices=list(GAME_ORDER), help="限定游戏")
    ap.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES, help="抓取正文页数上限")
    ap.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    ap.add_argument("--refresh", action="store_true", help="忽略缓存全量重抓")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    args = ap.parse_args()

    if args.build:
        build_index(games=[args.game] if args.game else None,
                    max_pages=args.max_pages, refresh=args.refresh)
        return

    index = load_index()

    if args.status:
        if not index:
            print("INDEX_MISSING 请先运行 --build")
            return
        st = index_stats(index)
        if args.json:
            print(json.dumps(st, ensure_ascii=False, indent=2))
        else:
            print("INDEX_STATUS")
            for k, v in st.items():
                print(f"  {k}: {v}")
        return

    if index is None:
        print("INDEX_MISSING 请先运行 python storylog_rag.py --build")
        sys.exit(1)

    if args.search:
        hits = search(index, args.search, topk=args.topk, game=args.game)
        if not hits:
            print("NO_HITS")
            return
        _print_hits(hits)
        return

    if args.ask:
        res = answer_question(args.ask, index, topk=args.topk, game=args.game)
        print(f"ANSWER_MODE: {res['mode']}")
        if args.json:
            print(json.dumps(res, ensure_ascii=False, indent=2))
            return
        if res["status"] != "ok":
            print(f"NOTE: {res['note']}")
            return
        print()
        print(res["answer"])
        if res.get("note"):
            print(f"\nNOTE: {res['note']}")
        print(f"\n置信度: {res.get('confidence')}")
        if res["citations"]:
            print("引用:")
            for c in res["citations"]:
                print(f"  [{c['n']}] 《{c['gameName']}》{c.get('chapter') or ''} {c.get('url') or ''}")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
