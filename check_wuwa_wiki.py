#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""鸣潮（库街区 wiki）剧情目录更新检测

数据源：库街区 wiki 的目录 API（该站为 SPA，无 MediaWiki revision 时间戳）
  - 目录树：POST https://api.kurobbs.com/wiki/core/catalogue/config/getActiveCatalogueListV2
  - 卡片列表：POST https://api.kurobbs.com/wiki/core/catalogue/item/getPage
              body: catalogueId=1330&page=1&limit=1000
              （1330 = 「剧情」目录，父节点 fid=1328）
  - 词条正文：POST https://api.kurobbs.com/wiki/core/catalogue/item/getEntryDetail
              参数 id=<entryId>（是 id，不是 entryId）
              正文：data.content.story.<节点>.flow.raw[].content

检测方式：由于没有 revision 时间戳，改为对「卡片名称列表」做快照比对。
  - 快照存在 .workbuddy/wuwa_wiki_state.json
  - 名称集合变化 -> UPDATE_DETECTED
  - 再与 index.html 的鸣潮条目比对，得出本地缺失的章节
  - 比对优先用封面 URL 的论坛文件 ID（精确），退化到标题归一化匹配

输出契约（与 check_ys_wiki.py 对齐）：
  FIRST_RUN | NO_UPDATE | UPDATE_DETECTED
  伴随 NEW_CHAPTERS_FOUND / NO_NEW_CHAPTERS

Usage:
  python check_wuwa_wiki.py                  # 常规快照比对
  python check_wuwa_wiki.py --check-chapters # 强制执行章节比对
"""
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime

import requests

from storylog_common import (
    BROWSER_HEADERS,
    forum_file_id,
    llm_available,
    llm_json_with_retry,
    loose_match,
    name_variants,
    read_episodes,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(SCRIPT_DIR, ".workbuddy", "wuwa_wiki_state.json")

GAME_KEY = "wuwa"
GAME_NAME = "鸣潮"

CATALOGUE_ID = 1330          # 「剧情」目录
PARENT_FID = 1328            # 父目录（版本/资料片）
API_BASE = "https://api.kurobbs.com/wiki/core"
WIKI_PAGE_URL = "https://wiki.kurobbs.com/mc/catalogue/list?fid=1328&sid=1330"

MAX_CANDIDATES = 200


def api_headers():
    h = dict(BROWSER_HEADERS)
    h.update({
        "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
        "Origin": "https://wiki.kurobbs.com",
        "Referer": WIKI_PAGE_URL,
        "wiki_type": "9",
        "source": "h5",
        "devcode": uuid.uuid4().hex,
    })
    return h


def fetch_cards():
    """返回 [{name, url, poster}]（按 wiki 展示顺序）。"""
    resp = requests.post(
        f"{API_BASE}/catalogue/item/getPage",
        data=f"catalogueId={CATALOGUE_ID}&page=1&limit=1000&_t={int(time.time() * 1000)}",
        headers=api_headers(),
        timeout=25,
    )
    resp.raise_for_status()
    data = resp.json()
    records = (
        data.get("data", {}).get("results", {}).get("records", [])
        or data.get("data", {}).get("records", [])
    )
    cards = []
    for rec in records:
        content = rec.get("content") or {}
        name = (rec.get("name") or content.get("title") or "").strip()
        if not name:
            continue
        cards.append({
            "name": name,
            "url": content.get("url") or content.get("contentUrl") or "",
            "poster": content.get("contentUrl") or "",
            # entryId 是词条详情的钥匙（getEntryDetail 接口，参数名为 id）
            "entryId": rec.get("entryId")
            or (content.get("linkConfig") or {}).get("entryId")
            or "",
        })
    return cards


def fetch_catalogues():
    """返回父目录 id -> 子目录 [{id, name}] 的映射（用于发现新增子目录）。"""
    resp = requests.post(
        f"{API_BASE}/catalogue/config/getActiveCatalogueListV2",
        data="",
        headers=api_headers(),
        timeout=25,
    )
    resp.raise_for_status()
    tree = resp.json().get("data") or []

    found = {}

    def walk(nodes):
        for node in nodes:
            if not isinstance(node, dict):
                continue
            nid = node.get("id")
            children = node.get("children") or []
            if nid is not None and children:
                found[nid] = [
                    {"id": c.get("id"), "name": c.get("name")}
                    for c in children if isinstance(c, dict)
                ]
            walk(children)

    walk(tree if isinstance(tree, list) else [])
    return found


# ---------- 词条正文（剧情对白） ----------
# 说明：库街区站点本身是 SPA，但词条正文有公开的 JSON 接口。
#   接口：POST https://api.kurobbs.com/wiki/core/catalogue/item/getEntryDetail
#   参数：id=<entryId>（注意是 id，不是 entryId）
#   正文：data.content.story.<节点ID>.flow[].raw[].content（HTML 富文本）

ENTRY_DETAIL_API = f"{API_BASE}/catalogue/item/getEntryDetail"


def fetch_entry_detail(entry_id):
    """获取词条详情（含正文）。失败抛异常。"""
    if not entry_id:
        raise ValueError("entryId 为空")
    resp = requests.post(
        ENTRY_DETAIL_API,
        data=f"id={entry_id}",
        headers=api_headers(),
        timeout=25,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("data") is None:
        raise RuntimeError(f"getEntryDetail 失败: {data.get('msg') or data.get('code')}")
    return data["data"]


def extract_story_html(entry):
    """从词条详情提取剧情正文 HTML 片段（按 flow 顺序）。

    结构：content.story.<节点ID>.flow.raw[].content（HTML 富文本）。
    兼容 flow 为 dict（{"raw": [...]}）或 list 两种形态。
    返回 HTML 片段列表；非剧情词条（纯资料/目录页）返回空列表。
    """
    story = ((entry or {}).get("content") or {}).get("story") or {}
    if not isinstance(story, dict):
        return []
    parts = []
    for node in story.values():
        if not isinstance(node, dict):
            continue
        flow = node.get("flow")
        if isinstance(flow, dict):
            raws = flow.get("raw") or []
        elif isinstance(flow, list):
            raws = flow
        else:
            raws = []
        for raw in raws:
            if isinstance(raw, dict):
                html = raw.get("content") or ""
                if html.strip():
                    parts.append(html)
    return parts


# ---------- 新章节匹配 ----------

def match_rule(cards, local_eps):
    """规则匹配：先按封面文件 ID，再按标题归一化（含上/下篇变体）。返回未匹配到的卡片。"""
    local_ids = {forum_file_id(e["poster"]) for e in local_eps if e["poster"]}
    local_ids.discard("")
    local_titles = [e["title"] for e in local_eps]

    unmatched = []
    for c in cards:
        cid = forum_file_id(c.get("poster", ""))
        if cid and cid in local_ids:
            continue
        names = name_variants(c["name"])
        if any(loose_match(n, t) for n in names for t in local_titles):
            continue
        unmatched.append(c)
    return unmatched


MATCH_SYSTEM = (
    "你是版本条目比对助手。只输出 JSON，不输出任何解释。"
    "任务：判断 wiki 新章节标题是否只是本地已有条目的改名、合并或等价变体。"
)

MATCH_USER_TMPL = """wiki 疑似新章节（JSON）：
{unmatched}

本地已有条目（JSON）：
{local}

判断规则：
- 语义相同（仅措辞、序号格式、标点、"上/下"拆分差异）视为已有，不输出。
- 本地完全没有对应剧情内容的才是新章节。
- new_titles 只能包含输入 wiki 疑似新章节中出现过的 title，禁止改写。

输出格式：{{"new_titles": ["...", "..."]}}"""


def validate_matched(data, allowed):
    if not isinstance(data, dict) or not isinstance(data.get("new_titles"), list):
        return None
    out = []
    for t in data["new_titles"]:
        if isinstance(t, str) and t in allowed:
            out.append(t)
        else:
            return None       # 出现编造标题，整体判为不可信
    return out


def match_llm(unmatched, local_titles):
    """LLM 语义复核。失败返回 None（保持规则结果）。"""
    if not llm_available() or not unmatched:
        return None
    names = [c["name"] for c in unmatched][:MAX_CANDIDATES]
    user = MATCH_USER_TMPL.format(
        unmatched=json.dumps(names, ensure_ascii=False),
        local=json.dumps(local_titles, ensure_ascii=False),
    )
    confirmed = llm_json_with_retry(
        MATCH_SYSTEM, user, lambda d: validate_matched(d, set(names))
    )
    if confirmed is None:
        return None
    keep = set(confirmed)
    return [c for c in unmatched if c["name"] in keep]


def run_chapter_check(cards=None):
    """返回 (新章节列表, 使用的方法)。"""
    if cards is None:
        cards = fetch_cards()
    local_eps = read_episodes(GAME_KEY)
    local_titles = [e["title"] for e in local_eps]

    unmatched = match_rule(cards, local_eps)
    method = "rules"

    refined = match_llm(unmatched, local_titles)
    if refined is not None:
        unmatched = refined
        method = "llm"

    new_chapters = [
        {"title": c["name"], "url": c["url"], "poster": c["poster"], "section": ""}
        for c in unmatched
    ]
    return new_chapters, method


# ---------- 状态 ----------

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    state["checkedAt"] = datetime.now().isoformat(timespec="seconds")
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def print_chapters(new_chapters, method):
    print(f"METHOD: {method}")
    if new_chapters:
        print(f"NEW_CHAPTERS_FOUND: {len(new_chapters)}")
        for ch in new_chapters:
            print(f"  - {ch['title']} -> {ch['url']}")
            if ch.get("poster"):
                print(f"      poster: {ch['poster']}")
    else:
        print("NO_NEW_CHAPTERS")


def main():
    force = "--check-chapters" in sys.argv

    cards = fetch_cards()
    names = [c["name"] for c in cards]

    if force:
        new_chapters, method = run_chapter_check(cards)
        print_chapters(new_chapters, method)
        return

    state = load_state()
    prev_names = state.get("names")

    # 顺带记录目录结构（发现新增子目录）
    try:
        catalogues = fetch_catalogues()
        children = catalogues.get(PARENT_FID, [])
        state["catalogue_children"] = children
    except Exception as e:
        children = None
        print(f"NOTE: 目录树获取失败（不影响主检测）: {e}")

    if prev_names is None:
        save_state({"names": names, "count": len(names), "catalogue_children": state.get("catalogue_children", [])})
        print("FIRST_RUN")
        print(f"Recorded {len(names)} cards.")
        # 首次运行也做一次章节比对，便于补齐本地缺失
        new_chapters, method = run_chapter_check(cards)
        print_chapters(new_chapters, method)
        return

    if prev_names == names:
        print("NO_UPDATE")
        print(f"Cards: {len(names)} (unchanged)")
        save_state(state)
        return

    added = [n for n in names if n not in set(prev_names)]
    removed = [n for n in prev_names if n not in set(names)]

    print("UPDATE_DETECTED")
    print(f"Previous: {len(prev_names)} cards")
    print(f"Current:  {len(names)} cards")
    if added:
        print(f"Wiki 新增: {', '.join(added)}")
    if removed:
        print(f"Wiki 移除: {', '.join(removed)}")
    if children is not None:
        print(f"目录子节点: {', '.join(str(c.get('name')) for c in children)}")

    try:
        new_chapters, method = run_chapter_check(cards)
        print_chapters(new_chapters, method)
    except Exception as e:
        print(f"CHAPTER_CHECK_ERROR: {e}")
        print("Manual review needed.")

    save_state({"names": names, "count": len(names), "catalogue_children": state.get("catalogue_children", [])})


if __name__ == "__main__":
    main()
