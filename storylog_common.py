# -*- coding: utf-8 -*-
"""剧情志 · 检测脚本共用工具层

职责：
  1. 从 index.html 读取游戏 / 版本 / 剧集目录（四级模型：游戏->版本->章节->单集）
  2. 文本归一化、名称变体、封面 URL 特征提取
  3. 统一的浏览器化请求头与 Session（规避部分 bwiki 子站的 567 风控）
  4. 再导出 LLM 通用层（实现在 storylog_llm.py），便于各脚本单一入口导入

注意：本模块不反向依赖任何检测脚本（check_*.py），避免循环导入。
"""
import os
import re

import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = ROOT                      # 兼容既有脚本的命名习惯
HTML_FILE = os.path.join(ROOT, "index.html")

# GAMES 数组中的游戏顺序，用于切分 index.html 区块
GAME_ORDER = ["genshin", "wuwa", "pns"]
GAME_NAMES = {"genshin": "原神", "wuwa": "鸣潮", "pns": "战双帕弥什"}

# LLM 通用层（独立模块，见 storylog_llm.py）
from storylog_llm import llm_available, llm_json_with_retry  # noqa: E402,F401


# ============================================================
# 网络请求：统一请求头
# ============================================================
# 实测：wiki.biligame.com/zspms 对「请求头不完整」的客户端返回 567 风控页，
# 而原神子站则正常。补全 Accept / Accept-Language 后即返回 200。
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9",
}


def make_session(referer=""):
    """带浏览器请求头的 Session；跨请求复用连接，降低被风控概率。"""
    s = requests.Session()
    s.headers.update(BROWSER_HEADERS)
    if referer:
        s.headers["Referer"] = referer
    return s


# ============================================================
# 目录解析：index.html
# ============================================================

def read_html():
    with open(HTML_FILE, "r", encoding="utf-8") as f:
        return f.read()


def read_game_section(game_id):
    """切出某个游戏在 GAMES 数组中的文本区块。"""
    html = read_html()
    start = html.find(f"    id: '{game_id}',")
    if start < 0:
        return ""
    idx = GAME_ORDER.index(game_id)
    if idx + 1 < len(GAME_ORDER):
        end = html.find(f"    id: '{GAME_ORDER[idx + 1]}',", start)
    else:
        end = html.find("\n];", start)
    return html[start:end if end > 0 else len(html)]


_EP_RE = re.compile(
    r"\{\s*id:\s*'([^']+)',\s*title:\s*'([^']*)'(?:,\s*poster:\s*'([^']*)')?"
)


def read_episodes(game_id):
    """返回该游戏全部剧集条目 [{'id','title','poster'}]。"""
    section = read_game_section(game_id)
    out = []
    for m in _EP_RE.finditer(section):
        out.append({"id": m.group(1), "title": m.group(2), "poster": m.group(3) or ""})
    return out


_VER_RE = re.compile(
    r"\{\s*id:\s*'([^']+)',\s*title:\s*'([^']*)',\s*version:\s*'([^']*)'"
)


def read_catalog():
    """按版本解析目录，返回：
    [{game, gameName, versionId, versionTitle, versionLabel, episodes:[{id,title}]}]

    与 read_episodes 的区别：这里保留「版本 -> 章节」的归属关系，
    版本条目自身不会被误当成剧集（用 id 排除）。
    """
    out = []
    for gid in GAME_ORDER:
        section = read_game_section(gid)
        if not section:
            continue
        marks = list(_VER_RE.finditer(section))
        for i, m in enumerate(marks):
            vid, vtitle, vlabel = m.group(1), m.group(2), m.group(3)
            lo = m.end()
            hi = marks[i + 1].start() if i + 1 < len(marks) else len(section)
            eps = []
            for em in _EP_RE.finditer(section[lo:hi]):
                if em.group(1) == vid:
                    continue
                eps.append({"id": em.group(1), "title": em.group(2)})
            out.append({
                "game": gid,
                "gameName": GAME_NAMES.get(gid, gid),
                "versionId": vid,
                "versionTitle": vtitle,
                "versionLabel": vlabel,
                "episodes": eps,
            })
    return out


# ---------- 文本与封面工具 ----------

_NOISE_RE = re.compile(r"[\s·、，,。.：:；;！!？?\-—－_~～（）()【】\[\]「」『』《》<>\"'’“”]+")


def norm(text):
    """归一化：去掉空白、标点、连接符，用于宽松比较。"""
    return _NOISE_RE.sub("", text or "")


_FORUM_RE = re.compile(r"/forum/([0-9a-fA-F]{32})")


def forum_file_id(url):
    """库街区封面 URL 的论坛文件 ID（32 位十六进制），用于精确匹配同一张图。"""
    if not url:
        return ""
    m = _FORUM_RE.search(url)
    return m.group(1).lower() if m else ""


def loose_match(a, b):
    """归一化后的双向子串匹配。"""
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return False
    return na in nb or nb in na


_SPLIT_SUFFIX_RE = re.compile(r"[\s·・]*(上|中|下|前篇|后篇|续)$")


def name_variants(name):
    """生成名称变体：原文 + 去掉结尾「上/中/下/前篇/后篇/续」后的主体。
    用于处理 wiki 拆成上下篇、而本地合并为一条的情况。"""
    name = (name or "").strip()
    out = []
    if name:
        out.append(name)
    base = _SPLIT_SUFFIX_RE.sub("", name).strip()
    if base and base != name:
        out.append(base)
    return out
