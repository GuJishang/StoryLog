#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""战双帕弥什（bilibili wiki）剧情回顾页更新检测

数据源：MediaWiki
  - 页面 revision：https://wiki.biligame.com/zspms/api.php
  - 页面正文：https://wiki.biligame.com/zspms/剧情回顾

页面结构（实测）：
  - 板块标题形如 <p><br />板块名 <br /> </p>
  - 首个板块「主线剧情」用 <div class="zs-DialogTab"><div style="display:none">主线剧情
  - 章节条目为 <div class="zs-btnDialogCBox"><a href="/zspms/xxx">标题</a> + CG 图

只跟踪 index.html 中已有的三个板块：
  主线剧情 -> p-main / 浮点纪实 -> p-fd / 外篇剧情 -> p-ex
其他板块的变化仅作提示，不自动同步。

输出契约（与 check_ys_wiki.py 对齐）：
  FIRST_RUN | NO_UPDATE | UPDATE_DETECTED
  伴随 NEW_CHAPTERS_FOUND / NO_NEW_CHAPTERS

Usage:
  python check_pns_wiki.py                  # 常规 revision 检查
  python check_pns_wiki.py --check-chapters # 强制执行章节比对
"""
import json
import os
import re
import sys
from html import unescape

import requests
from bs4 import BeautifulSoup

from storylog_common import (
    BROWSER_HEADERS,
    llm_available,
    llm_json_with_retry,
    loose_match,
    read_episodes,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(SCRIPT_DIR, ".workbuddy", "pns_wiki_state.json")

WIKI_API = "https://wiki.biligame.com/zspms/api.php"
PAGE_TITLE = "剧情回顾"
WIKI_PAGE_URL = "https://wiki.biligame.com/zspms/%E5%89%A7%E6%83%85%E5%9B%9E%E9%A1%BE"
WIKI_BASE = "https://wiki.biligame.com"
GAME_KEY = "pns"
GAME_NAME = "战双帕弥什"

# 跟踪的板块 -> index.html 中的版本组 id
SECTION_TO_VERSION = {
    "主线剧情": "p-main",
    "浮点纪实": "p-fd",
    "外篇剧情": "p-ex",
}
TRACKED_SECTIONS = list(SECTION_TO_VERSION.keys())

MAX_CANDIDATES = 200

# 统一浏览器化请求头（见 storylog_common）—— 本站点缺 Accept 等头会返回 567 风控页
HEADERS = BROWSER_HEADERS

# 板块标题：<p><br />板块名 <br /> </p>
_SECTION_MARK_RE = re.compile(r"<p>\s*<br\s*/?>\s*([^<>\n]{1,12}?)\s*<br\s*/?>\s*</p>")
# 首个板块标题：<div class="zs-DialogTab"><div style="display:none">主线剧情
_DIALOG_MARK_RE = re.compile(
    r'class="zs-DialogTab"><div style="display:none">([^<>\n]{1,12})'
)
# 章节链接
_LINK_RE = re.compile(r'<a\s+href="(/zspms/[^"#?]+)"[^>]*>([^<]{1,60})</a>')


# ---------- 页面抓取与解析 ----------

def get_wiki_html():
    resp = requests.get(WIKI_PAGE_URL, headers=HEADERS, timeout=25)
    resp.raise_for_status()
    return resp.text


def get_poster_map(raw):
    """标题 -> CG 封面 URL（来自 zs-btnDialogCBox 中的 CG 图）。
    页面给出的 src 已经是可直接引用的完整 URL（形如 .../thumb/x/xx/<hash>.png/280px-CG_xxx.png），
    与 index.html 中现有封面的格式一致，无需再做转换。"""
    soup = BeautifulSoup(raw, "lxml")
    mapping = {}
    for box in soup.find_all(class_="zs-btnDialogCBox"):
        a = box.find("a", href=True)
        if not a or not a["href"].startswith("/zspms/"):
            continue
        title = a.get_text(strip=True)
        img = box.find("img")
        if not title or not img:
            continue
        src = (img.get("data-src") or img.get("src") or "").strip()
        if src.startswith("//"):
            src = "https:" + src
        elif src.startswith("/"):
            src = WIKI_BASE + src
        if title not in mapping:
            mapping[title] = src
    return mapping


def get_sections(raw):
    """按板块切分，返回 {板块名: [{'title','url','poster'}]}（保持文档顺序）。"""
    marks = []
    for m in _SECTION_MARK_RE.finditer(raw):
        marks.append((m.start(), m.group(1).strip()))
    for m in _DIALOG_MARK_RE.finditer(raw):
        marks.append((m.start(), m.group(1).strip()))
    marks.sort(key=lambda x: x[0])

    # 去重（同一位置只保留一个）
    dedup = []
    for pos, name in marks:
        if dedup and dedup[-1][0] == pos:
            continue
        dedup.append((pos, name))
    marks = dedup

    links = [(m.start(), WIKI_BASE + m.group(1), unescape(m.group(2)).strip())
             for m in _LINK_RE.finditer(raw)]

    poster_map = get_poster_map(raw)

    bounds = [p for p, _ in marks] + [len(raw)]
    sections = {}
    for i, (_, name) in enumerate(marks):
        lo, hi = bounds[i], bounds[i + 1]
        seen, items = set(), []
        for pos, url, title in links:
            if not (lo <= pos < hi) or not title or title in seen:
                continue
            seen.add(title)
            items.append({
                "title": title,
                "url": url,
                "poster": poster_map.get(title, ""),
                "section": name,
            })
        if items and name not in sections:
            sections[name] = items
    return sections


# ---------- 新章节匹配 ----------

def local_titles_by_version():
    """按版本组返回本地已有条目标题。"""
    groups = {"p-main": [], "p-fd": [], "p-ex": []}
    for ep in read_episodes(GAME_KEY):
        eid = ep["id"]
        if eid.startswith("p-fd-"):
            groups["p-fd"].append(ep["title"])
        elif eid.startswith("p-ex-"):
            groups["p-ex"].append(ep["title"])
        else:
            groups["p-main"].append(ep["title"])
    return groups


MATCH_SYSTEM = (
    "你是版本条目比对助手。只输出 JSON，不输出任何解释。"
    "任务：判断 wiki 新章节标题是否只是本地已有条目的改名、合并或等价变体。"
)

MATCH_USER_TMPL = """wiki 疑似新章节（JSON）：
{unmatched}

本地已有条目（JSON）：
{local}

判断规则：
- 语义相同（仅措辞、序号格式、标点差异）视为已有，不输出。
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
            return None
    return out


def match_llm(unmatched, local_titles):
    if not llm_available() or not unmatched:
        return None
    names = [c["title"] for c in unmatched][:MAX_CANDIDATES]
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
    return [c for c in unmatched if c["title"] in keep]


def run_chapter_check(raw=None):
    """返回 (新章节列表, 方法说明, 各板块条目数)。"""
    if raw is None:
        raw = get_wiki_html()
    sections = get_sections(raw)
    local = local_titles_by_version()

    new_chapters = []
    methods = []
    section_counts = {}

    for section, items in sections.items():
        section_counts[section] = len(items)
        if section in TRACKED_SECTIONS:
            version_id = SECTION_TO_VERSION[section]
            titles = local.get(version_id, [])
            unmatched = [
                it for it in items
                if not any(loose_match(it["title"], t) for t in titles)
            ]
            method = "rules"
            refined = match_llm(unmatched, titles)
            if refined is not None:
                unmatched = refined
                method = "llm"
            if method == "llm":
                methods.append(f"{section}={method}")
            for it in unmatched:
                it = dict(it)
                it["version_id"] = version_id
                new_chapters.append(it)

    method_desc = "rules" if not methods else ", ".join(methods)
    return new_chapters, method_desc, section_counts


def diff_untracked(baseline, current):
    """未跟踪板块的条目数变化（只在有基线且确实变化时返回）。"""
    if not baseline:
        return []
    out = []
    for name, count in current.items():
        if name in TRACKED_SECTIONS:
            continue
        old = baseline.get(name)
        if old is not None and old != count:
            out.append((name, old, count))
    for name, old in baseline.items():
        if name not in current and name not in TRACKED_SECTIONS:
            out.append((name, old, 0))
    return out


# ---------- revision ----------

def get_last_revision():
    params = {
        "action": "query",
        "prop": "revisions",
        "titles": PAGE_TITLE,
        "rvlimit": "1",
        "rvprop": "timestamp|user|comment|size",
        "format": "json",
    }
    resp = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", {})
    for _, page in pages.items():
        revs = page.get("revisions", [])
        if revs:
            return revs[0]
    return None


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
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def print_chapters(new_chapters, method, untracked_changed=None):
    print(f"METHOD: {method}")
    if new_chapters:
        print(f"NEW_CHAPTERS_FOUND: {len(new_chapters)}")
        for ch in new_chapters:
            print(f"  - [{ch['section']} -> {ch.get('version_id', '?')}] {ch['title']} -> {ch['url']}")
            if ch.get("poster"):
                print(f"      poster: {ch['poster']}")
    else:
        print("NO_NEW_CHAPTERS")
    if untracked_changed:
        memo = "; ".join(f"{n} {o}->{c}" for n, o, c in untracked_changed)
        print(f"NOTE_UNTRACKED_SECTIONS_CHANGED: {memo}")


def main():
    force = "--check-chapters" in sys.argv

    if force:
        new_chapters, method, _counts = run_chapter_check()
        print_chapters(new_chapters, method, None)
        return

    rev = get_last_revision()
    if not rev:
        print("ERROR: 无法获取 revision 信息")
        sys.exit(1)

    current_ts = rev.get("timestamp")
    current_user = rev.get("user", "unknown")
    current_comment = rev.get("comment", "")
    current_size = rev.get("size", 0)

    state = load_state()
    stored_ts = state.get("timestamp")
    baseline_counts = state.get("section_counts") or {}

    def persist(counts):
        save_state({
            "timestamp": current_ts,
            "user": current_user,
            "comment": current_comment,
            "size": current_size,
            "section_counts": counts,
        })

    if stored_ts is None:
        print("FIRST_RUN")
        print(f"Recorded current revision: {current_ts}")
        print(f"Editor: {current_user}")
        try:
            new_chapters, method, counts = run_chapter_check()
            print_chapters(new_chapters, method, None)
        except Exception as e:
            counts = {}
            print(f"CHAPTER_CHECK_ERROR: {e}")
        persist(counts)
        return

    if stored_ts == current_ts:
        print("NO_UPDATE")
        print(f"Last edit: {current_ts} by {current_user}")
        return

    print("UPDATE_DETECTED")
    print(f"Previous: {stored_ts}")
    print(f"Current:  {current_ts}")
    print(f"Editor:   {current_user}")
    print(f"Comment:  {current_comment}")

    try:
        new_chapters, method, counts = run_chapter_check()
        untracked = diff_untracked(baseline_counts, counts)
        print_chapters(new_chapters, method, untracked)
    except Exception as e:
        counts = baseline_counts
        print(f"CHAPTER_CHECK_ERROR: {e}")
        print("Manual review needed.")

    persist(counts)


if __name__ == "__main__":
    main()
