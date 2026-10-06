#!/usr/bin/env python3
"""
Check if the Genshin Impact wiki page (魔神任务) has been updated.

v2 (Path-A / LLM 内核):
- 可选 LLM 结构化抽取：规则预筛候选链接 -> LLM 分类归一章节 -> schema 校验
  + 候选集交叉核验（防幻觉）-> 重试 -> 失败自动降级回规则解析。
- 可选 LLM 语义匹配：替代原「字符串包含」匹配，识别改名/合并章节，
  失败自动降级回规则匹配。
- LLM 走 OpenAI 兼容接口（requests 直调，无 SDK 依赖），密钥从环境变量读取。

Env vars:
  LLM_API_KEY    API 密钥；留空则纯规则模式
  LLM_API_BASE   默认 https://api.moonshot.cn/v1
  LLM_MODEL      默认 kimi-k2.6
  LLM_TIMEOUT    单次请求超时秒数，默认 60

Usage:
  python check_ys_wiki.py                    # 常规 revision 检查
  python check_ys_wiki.py --check-chapters   # 强制执行章节抽取与比对
"""
import requests
import json
import os
import re
import sys
import time
from urllib.parse import quote
from bs4 import BeautifulSoup

# LLM 通用层已抽到共享模块（原先本文件同时兼任「共享 LLM 库」，职责混淆）；
# 这里保持同名再导出，历史调用方（storylog_common / ai_review 等）无需改动。
from storylog_llm import (  # noqa: F401
    LLM_MODEL,
    llm_available,
    llm_chat,
    llm_json_with_retry,
    parse_llm_json,
)
from storylog_common import BROWSER_HEADERS

WIKI_API = "https://wiki.biligame.com/ys/api.php"
PAGE_TITLE = "魔神任务"
WIKI_PAGE_URL = "https://wiki.biligame.com/ys/%E9%AD%94%E7%A5%9E%E4%BB%BB%E5%8A%A1"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(SCRIPT_DIR, ".workbuddy", "ys_wiki_state.json")
HTML_FILE = os.path.join(SCRIPT_DIR, "index.html")

# 统一浏览器化请求头（见 storylog_common）
HEADERS = BROWSER_HEADERS

# LLM 配置与实现见 storylog_llm.py（环境变量：LLM_API_KEY / LLM_API_BASE / LLM_MODEL ...）
MAX_CANDIDATES = 200         # 送入 LLM 的候选链接上限
MAX_TEXT_LEN = 50            # 单个候选标题截断长度

WIKI_BASE = "https://wiki.biligame.com"


# ============================================================
# 章节抽取：规则路径（原逻辑）与 LLM 路径
# ============================================================

def get_wiki_html():
    resp = requests.get(WIKI_PAGE_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.text


def _normalize_anchor(a):
    """归一化一个 <a> 为 (text, href)，不合法返回 None。

    兼容魔神任务页的 taskIcon 幕级卡片结构（2026-10-05 修复的漏报根因）：
    a) 卡片内 <a> 正文只有 <img>，页面名在 title 属性里 -> 用 title 兜底；
    b) taskIconTitle/Subhead 的 href 是相对页名（如「白夜似梦初醒」，
       不带 /ys/ 前缀）-> 补全为 /ys/<urlencode>。
    """
    href = (a.get("href") or "").strip()
    text = a.get_text(strip=True) or (a.get("title") or "").strip()
    if not href or not text or "edit" in href or "action" in href:
        return None
    if href.startswith("#"):
        return None  # 页内锚点，不是独立词条
    if href.startswith("http://") or href.startswith("https://"):
        if not href.startswith(WIKI_BASE + "/ys/"):
            return None
        return text, href[len(WIKI_BASE):]
    if not href.startswith("/"):
        href = "/ys/" + quote(href, safe="")
    if not href.startswith("/ys/"):
        return None
    return text, href


def get_candidate_links(html):
    """规则预筛：所有指向 /ys/ 的正文链接，作为 LLM 的候选集（超集）。"""
    soup = BeautifulSoup(html, "lxml")
    content = soup.find("div", class_="mw-parser-output") or soup
    candidates = []
    seen = set()
    for a in content.find_all("a"):
        norm = _normalize_anchor(a)
        if not norm:
            continue
        text, href = norm
        if len(text) > MAX_TEXT_LEN:
            continue
        full_url = WIKI_BASE + href
        if full_url in seen:
            continue
        seen.add(full_url)
        candidates.append({"text": text, "href": href, "url": full_url})
        if len(candidates) >= MAX_CANDIDATES:
            break
    return candidates


def get_wiki_chapters_rules(html):
    """规则路径（原逻辑扩展）：按关键词抽取疑似章节链接。

    两类结构都覆盖：
    1) taskIcon 幕级卡片（魔神任务页主体）：taskIconTitle 给出「第X章 第Y幕」，
       taskIconSubhead 给出幕名，拼成「第Y幕：幕名」——与 index.html 条目
       命名一致，便于双向子串匹配；
    2) 其余含 幕/章/序/间章 关键词的文本链接（章级导航等，原逻辑）。
    """
    soup = BeautifulSoup(html, "lxml")
    content = soup.find("div", class_="mw-parser-output") or soup
    chapters = []
    seen = set()

    def add(title, href):
        key = (title, href)
        if title and href and key not in seen:
            seen.add(key)
            chapters.append({"title": title, "url": WIKI_BASE + href})

    # 1) taskIcon 幕级卡片
    for a in content.find_all("a", class_="taskIconTitle"):
        block = a.find_parent("div", class_="taskIcon")
        if not block:
            continue
        norm = _normalize_anchor(a)
        if not norm:
            continue
        head, href = norm
        sub = block.find("a", class_="taskIconSubhead")
        name = sub.get_text(strip=True) if sub else ""
        # 优先取「第X幕」；头部为「序章 第二幕」这类双前缀时，
        # 取幕号而非「序章」，才能与 index.html 的「第二幕：xxx」子串匹配
        m = re.search(r"第[一二三四五六七八九十百\d]+幕", head)
        if not m:
            m = re.search(r"幕间|序幕|序奏|序章", head)
        label = m.group(0) if m else head
        add(f"{label}：{name}" if name else label, href)

    # 2) 其余文本链接（原逻辑 + 归一化）
    for a in content.find_all("a"):
        cls = a.get("class") or []
        if "taskIconTitle" in cls or "taskIconSubhead" in cls:
            continue
        norm = _normalize_anchor(a)
        if not norm:
            continue
        text, href = norm
        if CHAPTER_HINT.search(text):
            add(text, href)
    return chapters


EXTRACT_SYSTEM = (
    "你是剧情章节结构化助手。只输出 JSON，不输出任何解释。"
    "任务：从候选链接中筛选出属于《原神》魔神任务剧情章节的条目"
    "（主线各章各幕、序章、间章），排除导航、活动、其他任务类型和无关链接。"
)

EXTRACT_USER_TMPL = """候选链接列表（JSON）：
{candidates}

要求：
1. title：章节名，保留幕/章/序号，可做轻度归一化（去修饰、统一全半角）。
2. url：必须严格等于 "https://wiki.biligame.com" + 该候选的 href，禁止编造或修改。
3. kind：从 主线/序章/间章 中选一个。
4. 没有符合的条目时输出 {{"chapters": []}}。

输出格式：{{"chapters": [{{"title": "...", "url": "...", "kind": "..."}}]}}"""


def validate_extracted(data, allowed_urls):
    """schema 校验 + 候选集交叉核验（url 必须来自候选集，防幻觉）。"""
    if not isinstance(data, dict) or not isinstance(data.get("chapters"), list):
        return None
    out, dropped = [], 0
    for c in data["chapters"]:
        if not isinstance(c, dict):
            dropped += 1
            continue
        title, url = c.get("title"), c.get("url")
        if not isinstance(title, str) or not title.strip() or len(title) > MAX_TEXT_LEN:
            dropped += 1
            continue
        if not isinstance(url, str) or url not in allowed_urls:
            dropped += 1
            continue
        kind = c.get("kind", "其他")
        out.append({
            "title": title.strip(),
            "url": url,
            "kind": kind if isinstance(kind, str) else "其他",
        })
    if dropped:
        print(f"NOTE: LLM 输出中 {dropped} 条未通过交叉核验，已丢弃")
    return out or None


CHAPTER_HINT = re.compile(r"幕|章|序|间章")


def filter_candidates_for_llm(candidates):
    """性能预筛：只把疑似章节的候选送 LLM（关键词超集），大幅压缩 token 与延迟。
    命中过少时回退全量候选，避免漏抽。"""
    hits = [c for c in candidates if CHAPTER_HINT.search(c["text"])]
    return hits if len(hits) >= 5 else candidates[:MAX_CANDIDATES]


def extract_chapters_llm(candidates):
    """LLM 结构化抽取。成功返回章节列表，失败返回 None（调用方降级）。"""
    if not llm_available() or not candidates:
        return None
    candidates = filter_candidates_for_llm(candidates)
    if not candidates:
        return None
    allowed_urls = {c["url"] for c in candidates}
    user = EXTRACT_USER_TMPL.format(
        candidates=json.dumps(
            [{"text": c["text"], "href": c["href"]} for c in candidates],
            ensure_ascii=False,
        )
    )
    return llm_json_with_retry(
        EXTRACT_SYSTEM, user, lambda d: validate_extracted(d, allowed_urls)
    )


# ============================================================
# 新章节匹配：规则路径（原逻辑）与 LLM 路径
# ============================================================

def match_new_chapters_rule(wiki_chapters, html_titles):
    """规则路径（原逻辑）：双向子串包含匹配。"""
    new_chapters = []
    for ch in wiki_chapters:
        if not any(ch["title"] in ep or ep in ch["title"] for ep in html_titles):
            new_chapters.append(ch)
    return new_chapters


MATCH_SYSTEM = (
    "你是版本条目比对助手。只输出 JSON，不输出任何解释。"
    "任务：判断 wiki 新章节标题是否只是本地已有条目的改名、合并或等价变体。"
)

MATCH_USER_TMPL = """wiki 疑似新章节（JSON）：
{unmatched}

本地已有条目（JSON）：
{local}

判断规则：
- 语义相同（仅措辞/序号格式/标点差异）视为已有，不输出。
- 本地完全没有对应剧情内容的才是新章节。
- new_titles 只能包含输入 wiki 疑似新章节中出现过的 title，禁止改写。

输出格式：{{"new_titles": ["...", "..."]}}"""


def validate_matched(data, unmatched_titles):
    if not isinstance(data, dict) or not isinstance(data.get("new_titles"), list):
        return None
    out = []
    for t in data["new_titles"]:
        if isinstance(t, str) and t in unmatched_titles:
            out.append(t)
        else:
            return None  # 出现编造标题，整体判为不可信
    return out


def match_new_chapters_llm(rule_unmatched, html_titles):
    """对规则匹配的剩余项做 LLM 语义复核。失败返回 None（保持规则结果）。"""
    if not llm_available() or not rule_unmatched:
        return None
    unmatched_titles = [ch["title"] for ch in rule_unmatched]
    user = MATCH_USER_TMPL.format(
        unmatched=json.dumps(unmatched_titles, ensure_ascii=False),
        local=json.dumps(html_titles, ensure_ascii=False),
    )
    confirmed = llm_json_with_retry(
        MATCH_SYSTEM, user, lambda d: validate_matched(d, unmatched_titles)
    )
    if confirmed is None:
        return None
    title_set = set(confirmed)
    return [ch for ch in rule_unmatched if ch["title"] in title_set]


# ============================================================
# 前端条目读取
# ============================================================

def get_html_episodes():
    """读取 index.html，抽取原神板块已有条目标题。"""
    with open(HTML_FILE, "r", encoding="utf-8") as f:
        html = f.read()
    genshin_match = re.search(r"id: 'genshir.*?(?=\bid: 'wuwa'|$)", html, re.DOTALL)
    if not genshin_match:
        genshin_match = re.search(
            r"id: 'genshin'.*?(?=\bid: 'wuwa'|\bid: 'pns'|\Z)", html, re.DOTALL
        )
    section = genshin_match.group(0) if genshin_match else html
    return re.findall(r"title:\s*'([^']+)'", section)


# ============================================================
# 主流程
# ============================================================

def run_chapter_check():
    """抽取 wiki 章节 -> 匹配本地条目 -> 返回 (新章节列表, 使用的方法)。"""
    html = get_wiki_html()
    candidates = get_candidate_links(html)
    html_titles = get_html_episodes()
    extract_method = "rules"

    wiki_chapters = extract_chapters_llm(candidates)
    if wiki_chapters is None:
        if llm_available():
            print("NOTE: LLM 抽取失败，已降级到规则解析")
        wiki_chapters = get_wiki_chapters_rules(html)
    else:
        extract_method = "llm"

    rule_unmatched = match_new_chapters_rule(wiki_chapters, html_titles)
    match_method = "rules"
    new_chapters = rule_unmatched

    if rule_unmatched and llm_available():
        llm_new = match_new_chapters_llm(rule_unmatched, html_titles)
        if llm_new is not None:
            new_chapters = llm_new
            match_method = "llm"

    method = f"extract={extract_method}, match={match_method}"
    return new_chapters, method


def get_last_revision():
    """从 MediaWiki API 获取页面最新 revision。"""
    params = {
        "action": "query",
        "prop": "revisions",
        "titles": PAGE_TITLE,
        "rvlimit": "1",
        "rvprop": "timestamp|user|comment|size",
        "format": "json",
    }
    resp = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=15)
    data = resp.json()
    pages = data.get("query", {}).get("pages", {})
    for page_id, page_data in pages.items():
        revisions = page_data.get("revisions", [])
        if revisions:
            return revisions[0]
    return None


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def main():
    force_check = "--check-chapters" in sys.argv

    if force_check:
        new_chapters, method = run_chapter_check()
        print(f"METHOD: {method}")
        if new_chapters:
            print(f"NEW_CHAPTERS_FOUND: {len(new_chapters)}")
            for ch in new_chapters:
                print(f"  - [{ch.get('kind', '?')}] {ch['title']} -> {ch['url']}")
        else:
            print("NO_NEW_CHAPTERS")
        return

    rev = get_last_revision()
    if not rev:
        print("ERROR: Could not fetch revision info from wiki API")
        sys.exit(1)

    current_ts = rev["timestamp"]
    current_user = rev.get("user", "unknown")
    current_comment = rev.get("comment", "")
    current_size = rev.get("size", 0)

    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    stored_ts = None
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
            stored_ts = state.get("timestamp")
    else:
        state = {"first_run": True}

    if stored_ts is None:
        state = {
            "timestamp": current_ts,
            "user": current_user,
            "comment": current_comment,
            "size": current_size,
        }
        save_state(state)
        print("FIRST_RUN")
        print(f"Recorded current revision: {current_ts}")
        print(f"Editor: {current_user}")
        print(f"Comment: {current_comment}")
        print("No comparison possible on first run. Subsequent runs will detect changes.")
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
        new_chapters, method = run_chapter_check()
        print(f"METHOD: {method}")
        if new_chapters:
            print(f"NEW_CHAPTERS_FOUND: {len(new_chapters)}")
            for ch in new_chapters:
                print(f"  - [{ch.get('kind', '?')}] {ch['title']} -> {ch['url']}")
        else:
            print("NO_NEW_CHAPTERS")
            print("Page was edited but no new chapter links found.")
    except Exception as e:
        print(f"\nCHAPTER_CHECK_ERROR: {e}")
        print("Manual review needed.")

    state["timestamp"] = current_ts
    state["user"] = current_user
    state["comment"] = current_comment
    state["size"] = current_size
    save_state(state)


if __name__ == "__main__":
    main()
