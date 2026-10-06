#!/usr/bin/env python3
"""
剧情志 · AI 复盘（路径 B）
基于应用「导出数据」生成的 JSON（reviews 语料），生成剧情偏好画像报告。

管线：规则统计底座（确定性，永远可出）
      -> LLM 偏好画像（schema 校验 + 引用核验防幻觉 + 重试 + 降级）
      -> 评分 vs 文本情感偏差分析（LLM，可选）
      -> 稳定性自检（画像生成两遍，输出偏好标签一致性实测值）

复用 check_ys_wiki 的 LLM 通用层（环境变量配置、重试、降级，见其文件头注释）。

Usage:
  python ai_review.py                       # 自动搜索 剧情志_导出_*.json
  python ai_review.py --data 导出.json      # 指定数据文件
  python ai_review.py --no-stability        # 跳过稳定性自检（省一次调用）
"""
import argparse
import datetime
import glob
import json
import os
import re
import sys

import storylog_llm as llm_core
from storylog_common import HTML_FILE, SCRIPT_DIR
from storylog_llm import (
    llm_available,
    llm_json_with_retry,
)

STATUS_LABELS = {
    "unset": "未开始", "planned": "计划中", "playing": "游玩中",
    "completed": "已完成", "dropped": "搁置",
}
MAX_REVIEWS_TO_LLM = 60      # 送入 LLM 的文本记录上限
MAX_REVIEW_CHARS = 200       # 单条长文截断长度


# ============================================================
# 目录解析：index.html -> episodeId -> {game, version, title}
# ============================================================

GAME_HEADER = re.compile(r"^    id: '(\w+)',\s*\n\s*name: '([^']+)'", re.M)
VERSION_HEADER = re.compile(
    r"id: '([\w-]+)',\s*\n\s*title: '([^']*)',\s*\n\s*version: '([^']*)'"
)
EPISODE_ITEM = re.compile(r"\{ id: '([\w-]+)', title: '((?:[^'\\]|\\.)*)'")


def build_catalog():
    """扫描 index.html，按位置顺序关联 游戏 -> 版本 -> 单集。返回 {eid: {...}}。"""
    with open(HTML_FILE, "r", encoding="utf-8") as f:
        html = f.read()
    catalog = {}
    games = [(mo.start(), mo.group(1), mo.group(2)) for mo in GAME_HEADER.finditer(html)]
    if not games:
        return catalog
    bounds = [g[0] for g in games] + [len(html)]
    for gi, (start, gid, gname) in enumerate(games):
        section = html[start:bounds[gi + 1]]
        markers = []
        for mo in VERSION_HEADER.finditer(section):
            markers.append((mo.start(), "v", mo.groups()))
        for mo in EPISODE_ITEM.finditer(section):
            markers.append((mo.start(), "e", mo.groups()))
        markers.sort(key=lambda x: x[0])
        cur_ver = "未知版本"
        for _, kind, groups in markers:
            if kind == "v":
                cur_ver = groups[2] or groups[1]
            else:
                eid, title = groups
                if eid not in catalog:
                    catalog[eid] = {
                        "game": gname, "game_id": gid,
                        "version": cur_ver, "title": title,
                    }
    return catalog


# ============================================================
# 数据加载与规则统计
# ============================================================

def find_data_file(explicit):
    if explicit:
        return explicit if os.path.exists(explicit) else None
    hits = sorted(glob.glob(os.path.join(SCRIPT_DIR, "剧情志_导出_*.json")))
    return hits[-1] if hits else None


def load_records(path, catalog):
    """读取导出 JSON，关联目录元数据。返回记录列表。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    reviews = data.get("reviews", {})
    records = []
    for eid, r in reviews.items():
        if not isinstance(r, dict):
            continue
        meta = catalog.get(eid, {})
        records.append({
            "episode_id": eid,
            "game": meta.get("game", "未知游戏"),
            "version": meta.get("version", "?"),
            "title": meta.get("title", eid),
            "rating": int(r.get("rating") or 0),
            "status": r.get("status", "unset"),
            "review": str(r.get("review") or "").strip(),
            "date": str(r.get("date") or ""),
        })
    return records


def compute_stats(records):
    """规则统计底座：确定性输出，不依赖 LLM。"""
    stats = {
        "total_records": len(records),
        "by_game": {},
        "by_status": {},
        "rating_dist": {str(i): 0 for i in range(1, 6)},
        "rated_count": 0,
        "text_review_count": 0,
        "avg_rating_by_game": {},
        "months_active": [],
    }
    rating_sum = {}
    rating_cnt = {}
    for r in records:
        g = stats["by_game"].setdefault(r["game"], 0)
        stats["by_game"][r["game"]] = g + 1
        s = STATUS_LABELS.get(r["status"], r["status"])
        stats["by_status"][s] = stats["by_status"].get(s, 0) + 1
        if 1 <= r["rating"] <= 5:
            stats["rating_dist"][str(r["rating"])] += 1
            stats["rated_count"] += 1
            rating_sum[r["game"]] = rating_sum.get(r["game"], 0) + r["rating"]
            rating_cnt[r["game"]] = rating_cnt.get(r["game"], 0) + 1
        if r["review"]:
            stats["text_review_count"] += 1
        if len(r["date"]) >= 7:
            stats["months_active"].append(r["date"][:7])
    for game, total in rating_sum.items():
        stats["avg_rating_by_game"][game] = round(total / rating_cnt[game], 2)
    stats["months_active"] = sorted(set(stats["months_active"]))
    return stats


# ============================================================
# LLM 偏好画像
# ============================================================

PORTRAIT_SYSTEM = (
    "你是游戏剧情体验分析助手。只输出 JSON，不输出任何解释。"
    "任务：基于用户的剧情记录统计与原文短评，生成个人剧情偏好画像。"
    "所有 highlights 的 title 必须逐字来自输入记录中出现的标题，禁止编造。"
)

PORTRAIT_USER_TMPL = """记录统计（JSON）：
{stats}

带文字短评的记录（JSON，review 为用户原文）：
{text_reviews}

要求：
1. preference_tags：3-6 个偏好标签（如剧情类型、叙事风格、情感倾向维度）。
2. profile：3-5 句画像，引用具体记录支撑，不说空话。
3. highlights：最有代表性的记录，title 逐字来自上面输入；记录不足 3 条时输出全部即可，禁止编造或使用未出现的标题。
4. suggestions：1-3 条具体可执行的后续体验建议。

输出格式：{{"preference_tags": ["..."], "profile": "...",
"highlights": [{{"title": "...", "reason": "..."}}], "suggestions": ["..."]}}"""


def validate_portrait(data, known_titles):
    if not isinstance(data, dict):
        return None
    tags = data.get("preference_tags")
    profile = data.get("profile")
    highlights = data.get("highlights")
    suggestions = data.get("suggestions")
    if not (isinstance(tags, list) and 1 <= len(tags) <= 8
            and all(isinstance(t, str) and t.strip() for t in tags)):
        return None
    if not isinstance(profile, str) or len(profile.strip()) < 20:
        return None
    if not (isinstance(highlights, list) and isinstance(suggestions, list)):
        return None
    clean_hl, dropped = [], 0
    for h in highlights:
        if not isinstance(h, dict):
            dropped += 1
            continue
        t = h.get("title")
        # 引用核验：title 必须来自输入记录（防幻觉），不匹配则丢弃该条
        if not isinstance(t, str) or not any(t in kt or kt in t for kt in known_titles):
            dropped += 1
            continue
        clean_hl.append({"title": t, "reason": str(h.get("reason", "")).strip()})
    if highlights and not clean_hl:
        return None  # 全部编造，整体判不可信
    if dropped:
        print(f"NOTE: 画像中 {dropped} 条代表性记录未通过引用核验，已丢弃")
    clean_sug = [s for s in suggestions if isinstance(s, str) and s.strip()]
    if len(clean_sug) != len(suggestions):
        return None
    return {
        "preference_tags": [t.strip() for t in tags],
        "profile": profile.strip(),
        "highlights": clean_hl,
        "suggestions": clean_sug,
    }


def generate_portrait(records, stats):
    """LLM 画像。失败返回 None（报告降级为纯统计）。"""
    if not llm_available():
        return None
    text_records = [r for r in records if r["review"]][:MAX_REVIEWS_TO_LLM]
    text_reviews = [
        {"game": r["game"], "title": r["title"], "rating": r["rating"],
         "review": r["review"][:MAX_REVIEW_CHARS]}
        for r in text_records
    ]
    if not text_reviews:
        return None
    user = PORTRAIT_USER_TMPL.format(
        stats=json.dumps(stats, ensure_ascii=False),
        text_reviews=json.dumps(text_reviews, ensure_ascii=False),
    )
    known_titles = {r["title"] for r in records} | {r["title"] for r in text_records}
    return llm_json_with_retry(
        PORTRAIT_SYSTEM, user, lambda d: validate_portrait(d, known_titles)
    )


# ============================================================
# 评分 vs 文本情感 偏差分析
# ============================================================

SENTIMENT_SYSTEM = (
    "你是文本情感标注助手。只输出 JSON，不输出任何解释。"
    "对每条剧情短评标注整体情感倾向，忽略评分本身，只看文本。"
)

SENTIMENT_USER_TMPL = """记录列表（JSON）：
{items}

对每条输出 sentiment，取值仅限 positive / neutral / negative。
title 必须逐字来自输入。

输出格式：{{"items": [{{"title": "...", "sentiment": "..."}}]}}"""


def validate_sentiment(data, known_titles):
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        return None
    out = []
    for it in data["items"]:
        if not isinstance(it, dict):
            return None
        t, s = it.get("title"), it.get("sentiment")
        if not isinstance(t, str) or t not in known_titles or s not in ("positive", "neutral", "negative"):
            return None
        out.append({"title": t, "sentiment": s})
    return out


def analyze_sentiment_bias(records):
    """LLM 情感标注 -> 与评分对比。返回 (status, 偏差列表)。
    status: skipped(语料不足) / failed(LLM失败) / ok(有结果，偏差可能为空)。"""
    if not llm_available():
        return "skipped", None
    items = [
        {"title": r["title"], "rating": r["rating"], "review": r["review"][:MAX_REVIEW_CHARS]}
        for r in records if len(r["review"]) >= 10
    ][:MAX_REVIEWS_TO_LLM]
    if not items:
        return "skipped", None
    user = SENTIMENT_USER_TMPL.format(items=json.dumps(items, ensure_ascii=False))
    known = {it["title"] for it in items}
    labeled = llm_json_with_retry(
        SENTIMENT_SYSTEM, user, lambda d: validate_sentiment(d, known)
    )
    if labeled is None:
        return "failed", None
    by_title = {it["title"]: it["rating"] for it in items}
    deviations = []
    for it in labeled:
        rating = by_title[it["title"]]
        if rating >= 4 and it["sentiment"] == "negative":
            deviations.append({**it, "rating": rating,
                               "note": "高分但文字偏负面，期待与体验可能存在落差"})
        elif rating <= 2 and it["sentiment"] == "positive":
            deviations.append({**it, "rating": rating,
                               "note": "低分但文字偏正面，可能在克制表达不满以外的情绪"})
    return "ok", deviations


# ============================================================
# 稳定性自检：画像两遍，偏好标签一致性
# ============================================================

def stability_check(records, stats):
    """生成两遍画像，计算偏好标签 Jaccard 重叠。返回 (一致性等级, 标签集合对)。"""
    a = generate_portrait(records, stats)
    b = generate_portrait(records, stats)
    if not a or not b:
        return "无法评估（LLM 不可用或失败）", a, b
    ta, tb = {t.lower() for t in a["preference_tags"]}, {t.lower() for t in b["preference_tags"]}
    union = ta | tb
    overlap = len(ta & tb) / len(union) if union else 1.0
    level = "高" if overlap >= 0.5 else ("中" if overlap > 0 else "低")
    return f"{level}（标签重叠 {len(ta & tb)}/{len(union)}）", a, b


# ============================================================
# 报告生成
# ============================================================

def render_report(records, stats, portrait, sentiment_status, deviations, stability, data_path):
    today = datetime.date.today().isoformat()
    lines = [
        "# 剧情志 · AI 复盘报告", "",
        f"- 生成日期：{today}",
        f"- 数据来源：`{os.path.basename(data_path)}`",
        f"- 画像模式：{'LLM + 规则统计' if portrait else '仅规则统计（LLM 不可用或失败）'}",
        "",
        "## 一、记录总览（规则统计，确定性）", "",
        f"- 总记录 **{stats['total_records']}** 条，其中评分 {stats['rated_count']} 条、"
        f"带文字短评 {stats['text_review_count']} 条",
        f"- 各游戏记录数：{'、'.join(f'{k} {v}' for k, v in stats['by_game'].items()) or '无'}",
        f"- 状态分布：{'、'.join(f'{k} {v}' for k, v in stats['by_status'].items()) or '无'}",
        f"- 评分分布（1-5 星）：{'、'.join(f'{k}★×{v}' for k, v in stats['rating_dist'].items())}",
        f"- 各游戏均分：{'、'.join(f'{k} {v}' for k, v in stats['avg_rating_by_game'].items()) or '无'}",
        f"- 活跃月份：{'、'.join(stats['months_active']) or '无日期记录'}",
        "",
    ]
    if portrait:
        lines += [
            "## 二、剧情偏好画像（LLM）", "",
            f"**偏好标签**：{'、'.join(portrait['preference_tags'])}", "",
            portrait["profile"], "",
            "### 代表性记录", "",
        ]
        lines += [f"- 「{h['title']}」— {h['reason']}" for h in portrait["highlights"]]
        lines += ["", "### 后续建议", ""]
        lines += [f"- {s}" for s in portrait["suggestions"]]
        lines.append("")
    else:
        lines += ["## 二、剧情偏好画像（LLM）", "", "LLM 不可用或调用失败，本节降级跳过。", ""]
    if sentiment_status == "ok":
        lines += ["## 三、评分 vs 文字情感 偏差", ""]
        if deviations:
            lines += [f"- 「{d['title']}」（{d['rating']}★）：{d['note']}" for d in deviations]
        else:
            lines += ["所有长文短评的情感倾向与评分一致，未发现明显偏差。"]
        lines.append("")
    elif sentiment_status == "skipped":
        lines += ["## 三、评分 vs 文字情感 偏差", "", "无足够长文语料，跳过。", ""]
    else:
        lines += ["## 三、评分 vs 文字情感 偏差", "", "LLM 调用失败，本节跳过。", ""]
    lines += [
        "## 四、输出稳定性自检", "",
        f"- 模型：`{llm_core.LLM_MODEL}`",
        f"- 两遍生成的偏好标签一致性：{stability}",
        "",
    ]
    return "\n".join(lines)


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="剧情志 AI 复盘")
    parser.add_argument("--data", help="导出 JSON 路径（默认自动搜索 剧情志_导出_*.json）")
    parser.add_argument("--out", help="报告输出路径")
    parser.add_argument("--no-stability", action="store_true", help="跳过稳定性自检")
    args = parser.parse_args()

    data_path = find_data_file(args.data)
    if not data_path:
        print("ERROR: 未找到导出数据。请先在剧情志应用里点「导出数据」按钮，")
        print("       把生成的 剧情志_导出_*.json 放到脚本目录，或用 --data 指定路径。")
        sys.exit(1)
    print(f"数据文件: {data_path}")

    catalog = build_catalog()
    print(f"目录解析: {len(catalog)} 个单集条目")
    records = load_records(data_path, catalog)
    if not records:
        print("ERROR: 导出文件里没有有效记录（reviews 为空）。先在应用里记录几条剧情体验再导出。")
        sys.exit(1)
    print(f"有效记录: {len(records)} 条")

    stats = compute_stats(records)

    portrait = generate_portrait(records, stats)
    if portrait is None and llm_available():
        print("NOTE: LLM 画像失败，报告降级为纯统计")
    print(f"画像: {'LLM 生成' if portrait else '降级为纯统计'}")

    sentiment_status, deviations = analyze_sentiment_bias(records)
    print(f"情感偏差分析: {'完成' if sentiment_status == 'ok' else ('跳过' if sentiment_status == 'skipped' else '失败降级')}")

    if args.no_stability:
        stability = "已跳过（--no-stability）"
    else:
        stability, a, b = stability_check(records, stats)
        print(f"稳定性自检: {stability}")

    out_path = args.out or os.path.join(SCRIPT_DIR, f"剧情志-AI复盘-{datetime.date.today().isoformat()}.md")
    report = render_report(records, stats, portrait, sentiment_status, deviations, stability, data_path)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"报告已生成: {out_path}")


if __name__ == "__main__":
    main()
