#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""剧情志 · LLM 通用层（全工程共用）

从 check_ys_wiki.py 中抽出：原先「原神检测脚本」同时兼任「共享 LLM 库」，
职责混淆且与 storylog_common 存在循环导入隐患。抽成独立模块后：

    storylog_llm.py      纯 LLM 能力（无本地依赖）
        ↑
        ├── check_ys_wiki.py / check_pns_wiki.py / check_wuwa_wiki.py
        ├── storylog_common.py （工具层再导出）
        ├── ai_review.py      （路径 B：AI 复盘）
        └── storylog_rag.py   （路径 C：剧情问答 RAG）

对外接口
--------
  llm_available()                     是否配置了 Key
  llm_chat(system, user)              调 OpenAI 兼容 /chat/completions，返回文本
  parse_llm_json(text)                解析 JSON（容忍 ```json 围栏）
  llm_json_with_retry(system, user, validator)
                                      带 schema 校验与重试，全部失败返回 None（调用方降级）

Env vars
--------
  LLM_API_KEY      密钥；留空则进入纯规则/纯检索模式
  LLM_API_BASE     默认 https://api.moonshot.cn/v1
  LLM_MODEL        默认 kimi-k2.6
  LLM_TIMEOUT      单次请求超时秒数，默认 60
  LLM_TEMPERATURE  默认 0；部分模型（kimi-k2.x）只接受 1，遇 400 自动切换
"""
import json
import os
import re
import time

import requests

# ---------- 配置（环境变量，无硬编码密钥） ----------
LLM_API_KEY = os.environ.get("LLM_API_KEY", "").strip()
LLM_API_BASE = os.environ.get("LLM_API_BASE", "https://api.moonshot.cn/v1").rstrip("/")
LLM_MODEL = os.environ.get("LLM_MODEL", "kimi-k2.6")
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "60"))
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0"))
LLM_MAX_RETRIES = 2          # 总尝试次数 = 1 + LLM_MAX_RETRIES
LLM_RETRY_BACKOFF = 2        # 秒
LLM_RATE_BACKOFF = 35        # 秒，429/限流时的退避时长


def llm_available():
    return bool(LLM_API_KEY)


def _post_chat(payload, timeout=None):
    """底层 POST /chat/completions，返回解析后的 JSON。失败抛异常。

    温度自适应：部分模型（如 kimi-k2.x）仅允许 temperature=1，遇 400 自动切换重试一次。
    """
    temp = payload.get("temperature", LLM_TEMPERATURE)
    for _ in range(2):
        payload["temperature"] = temp
        resp = requests.post(
            f"{LLM_API_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            json=payload,
            timeout=timeout or LLM_TIMEOUT,
        )
        try:
            resp.raise_for_status()
        except requests.HTTPError:
            body = getattr(resp, "text", "") or ""
            if temp != 1 and "temperature" in body.lower():
                temp = 1
                continue
            raise
        return resp.json()


def llm_chat(system, user):
    """调用 OpenAI 兼容 chat/completions，返回文本内容。失败抛异常。

    强制 JSON 输出（response_format=json_object），供各模块的结构化抽取使用。
    """
    data = _post_chat({
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": LLM_TEMPERATURE,
        "response_format": {"type": "json_object"},
    })
    return data["choices"][0]["message"]["content"]


def llm_chat_full(messages, tools=None, response_format=None, temperature=None, timeout=None):
    """多轮 + 工具调用对话（Agent 层使用）。

    返回 {"message": {...}, "usage": {...}}；message 可能含 tool_calls。
    与 llm_chat 的区别：带上 tools 时不发 response_format（多数服务端不允许两者同时指定）。
    """
    payload = {"model": LLM_MODEL, "messages": messages,
               "temperature": LLM_TEMPERATURE if temperature is None else temperature}
    if tools:
        payload["tools"] = tools
        payload.setdefault("tool_choice", "auto")
    elif response_format:
        payload["response_format"] = response_format

    data = _post_chat(payload, timeout=timeout)
    choice = (data.get("choices") or [{}])[0]
    return {"message": choice.get("message") or {}, "usage": data.get("usage") or {}}


def parse_llm_json(text):
    """解析 LLM 输出 JSON，容忍 ```json 代码围栏。失败抛异常。"""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", cleaned, flags=re.DOTALL)
    return json.loads(cleaned)


def llm_json_with_retry(system, user, validator):
    """带重试的 LLM JSON 调用。每次输出过 validator，全部失败返回 None。"""
    last_err = None
    for attempt in range(1 + LLM_MAX_RETRIES):
        try:
            data = parse_llm_json(llm_chat(system, user))
            validated = validator(data)
            if validated is not None:
                return validated
            last_err = "输出未通过校验"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            body = getattr(e, "response", None)
            if body is not None:
                last_err += f" | body: {body.text[:200]}"
            # 限流退避：429/rate_limit 时等更久再重试
            if "429" in last_err or "rate_limit" in last_err.lower():
                time.sleep(LLM_RATE_BACKOFF)
                continue
        if attempt < LLM_MAX_RETRIES:
            time.sleep(LLM_RETRY_BACKOFF)
    hint = "（API 账户额度不足，请充值或更换 Key）" if "insufficient_user_quota" in str(last_err) else ""
    print(f"NOTE: LLM 调用失败，原因: {last_err}{hint}")
    return None
