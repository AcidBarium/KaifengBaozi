"""edge-tts 接入点兼容 + 重试 + 缓存层。

背景（2026-10 起）
------------------
Microsoft 把 Edge Read Aloud 的**语音合成** WebSocket 从旧地址迁走了：

    旧: wss://speech.platform.bing.com/consumer/speech/synthesize/readaloud/edge/v1
    新: wss://api.msedgeservices.com/tts/cognitiveservices/websocket/v1

旧地址现在在大部分网络直接返回 **403 Invalid response status**（rany2/edge-tts#490），
而 edge-tts 7.2.8（当前 PyPI 最新版）里 WSS_URL 仍写死旧地址，
所以 `/api/tts` 会 100% 走到 except 分支、统一返回 502 Bad Gateway。

新地址需要同样的 TrustedClientToken / Sec-MS-GEC，并把 token 再作为
`Ocp-Apim-Subscription-Key` 传一次。实测新地址可用，但会偶发
`ClientConnectorError`（连不上）和 `429 Too Many Requests`（限流），
因此这里同时实现：接入点切换 + 指数退避重试 + 串行化 + 磁盘缓存。

用法
----
    from tts_endpoint import synthesize_speech
    audio: bytes = await synthesize_speech("订单 66 完成")

可调环境变量
------------
    TTS_VOICE        默认发音人（默认 zh-CN-XiaoxiaoNeural）
    TTS_RATE         语速（默认 +0%）
    TTS_VOLUME       音量（默认 +0%）
    TTS_PITCH        音调（默认 +0Hz）
    TTS_ATTEMPTS     每个接入点的尝试次数（默认 3）
    TTS_BACKOFF      首次退避秒数，指数增长（默认 0.6）
    TTS_TIMEOUT      单次合成总超时秒数（默认 20）
    TTS_CACHE_DIR    音频缓存目录（默认 <项目根>/tts-cache）
    TTS_CACHE_TTL    缓存有效秒数（默认 604800 = 7 天，设 0 关闭缓存）
    TTS_ENDPOINT     强制只用某个接入点：new / old（默认 auto：先 new 再 old）
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import edge_tts
import edge_tts.communicate as _communicate

def log() -> logging.Logger:
    """复用 uvicorn 已配置的 handler，日志才能出现在 journalctl 里。

    直接 getLogger("tts") 打出的内容不会进 uvicorn 的 handler（root logger 没配），
    会被静默丢弃，所以这里优先走 uvicorn.error。
    """
    uv = logging.getLogger("uvicorn.error")
    # uvicorn.error 自身没有 handler，是靠向上传播给 uvicorn 的，
    # 所以必须用 hasHandlers()（沿祖先链找），不能只看 .handlers。
    return uv if uv.hasHandlers() else logging.getLogger(__name__)


# Edge 浏览器内置朗读功能所用的公开客户端标识，非私密凭证。
TRUSTED_CLIENT_TOKEN = "6A5AA1D4EAFF4E9FB37E23D68491D6F4"

_NEW_URL = (
    "wss://api.msedgeservices.com/tts/cognitiveservices/websocket/v1"
    f"?TrustedClientToken={TRUSTED_CLIENT_TOKEN}"
    f"&Ocp-Apim-Subscription-Key={TRUSTED_CLIENT_TOKEN}"
)
_OLD_URL = (
    "wss://speech.platform.bing.com/consumer/speech/synthesize/readaloud/edge/v1"
    f"?TrustedClientToken={TRUSTED_CLIENT_TOKEN}"
)

# (名称, URL)。auto 模式下按此顺序尝试：新地址优先，旧地址仅作兜底
# （个别区域旧地址可能仍然可用）。
_ENDPOINTS: Dict[str, Tuple[str, str]] = {
    "new": ("api.msedgeservices.com", _NEW_URL),
    "old": ("speech.platform.bing.com", _OLD_URL),
}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


DEFAULT_VOICE = os.getenv("TTS_VOICE", "zh-CN-XiaoxiaoNeural")
DEFAULT_RATE = os.getenv("TTS_RATE", "+0%")
DEFAULT_VOLUME = os.getenv("TTS_VOLUME", "+0%")
DEFAULT_PITCH = os.getenv("TTS_PITCH", "+0Hz")

ATTEMPTS = max(1, _env_int("TTS_ATTEMPTS", 3))
BACKOFF = max(0.0, _env_float("TTS_BACKOFF", 0.6))
# 遇到 429 时额外多等的时间：限流状态下死磕只会把尝试次数烧光
THROTTLE_BACKOFF = max(0.0, _env_float("TTS_THROTTLE_BACKOFF", 4.0))
TIMEOUT = max(3.0, _env_float("TTS_TIMEOUT", 20))
CONNECT_TIMEOUT = min(10, int(TIMEOUT))

CACHE_DIR = Path(os.getenv("TTS_CACHE_DIR") or (Path(__file__).resolve().parent.parent / "tts-cache"))
CACHE_TTL = _env_int("TTS_CACHE_TTL", 7 * 24 * 3600)

# 串行化合成：既避免并发下切换模块级 WSS_URL 的竞态，
# 也天然压低请求频率，减少 429。
_synth_lock = asyncio.Lock()


def endpoint_order() -> List[Tuple[str, str]]:
    """返回本次要尝试的 (名称, URL) 列表。"""
    forced = (os.getenv("TTS_ENDPOINT") or "").strip().lower()
    if forced in _ENDPOINTS:
        return [_ENDPOINTS[forced]]
    return [_ENDPOINTS["new"], _ENDPOINTS["old"]]


def _build_communicate(text: str, voice: str, rate: str, volume: str, pitch: str):
    """按当前安装的 edge-tts 版本可接受的参数构造 Communicate（兼容旧版本）。"""
    kwargs = {"rate": rate, "volume": volume, "pitch": pitch}
    params = inspect.signature(edge_tts.Communicate.__init__).parameters
    if "connect_timeout" in params:
        kwargs["connect_timeout"] = CONNECT_TIMEOUT
    if "receive_timeout" in params:
        kwargs["receive_timeout"] = int(TIMEOUT)
    return edge_tts.Communicate(text, voice, **kwargs)


async def _synth_once(url: str, text: str, voice: str, rate: str, volume: str, pitch: str) -> bytes:
    # edge-tts 在建立连接时读取本模块全局 WSS_URL，因此这里是切换接入点的唯一开关。
    _communicate.WSS_URL = url
    communicate = _build_communicate(text, voice, rate, volume, pitch)
    buffer = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buffer.extend(chunk["data"])
    if not buffer:
        raise RuntimeError("空音频")
    return bytes(buffer)


async def _synthesize_uncached(text: str, voice: str, rate: str, volume: str, pitch: str) -> bytes:
    if not hasattr(_communicate, "WSS_URL"):
        raise RuntimeError(
            "当前 edge-tts 版本里没有 communicate.WSS_URL，无法切换接入点；"
            "请先升级: pip install -U edge-tts"
        )
    endpoints = endpoint_order()
    last_exc: Optional[BaseException] = None

    for name, url in endpoints:
        for attempt in range(1, ATTEMPTS + 1):
            try:
                audio = await asyncio.wait_for(
                    _synth_once(url, text, voice, rate, volume, pitch), timeout=TIMEOUT + 5
                )
                log().info("TTS 成功: %s (第 %d 次尝试, %d 字节)", name, attempt, len(audio))
                return audio
            except Exception as exc:  # noqa: BLE001 - 逐个接入点降级
                last_exc = exc
                log().warning(
                    "TTS 失败: %s 第 %d/%d 次尝试: %s: %s",
                    name, attempt, ATTEMPTS, type(exc).__name__, exc,
                )
                if attempt < ATTEMPTS:
                    delay = BACKOFF * (2 ** (attempt - 1)) + random.uniform(0, 0.3)
                    if "429" in str(exc):
                        # 被限流了：退避久一点，否则下一发还是 429，白白烧掉尝试次数
                        delay += THROTTLE_BACKOFF
                    await asyncio.sleep(delay)

    raise RuntimeError(f"所有 TTS 接入点均失败，最后一次: {type(last_exc).__name__}: {last_exc}")


# ── 磁盘缓存 ────────────────────────────────────────────────────────────


def cache_path(key: str, namespace: str = "") -> Path:
    """缓存文件路径。

    namespace 用来隔离不同用途的音频：整句合成走根目录，裁过静音的
    拼接片段走 `segments/`，两者即使原文相同也不会互相覆盖。
    """
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    base = CACHE_DIR / namespace if namespace else CACHE_DIR
    return base / f"{digest}.mp3"


def load_cached(path: Path) -> Optional[bytes]:
    if CACHE_TTL <= 0:
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    if time.time() - stat.st_mtime > CACHE_TTL:
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return data or None


def store_cached(path: Path, audio: bytes) -> None:
    if CACHE_TTL <= 0:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(audio)
        tmp.replace(path)  # 原子落盘，避免半截文件被读到
    except OSError as exc:
        log().warning("TTS 缓存写入失败 %s: %s", path, exc)


def utterance_key(text: str, voice: str, rate: str, volume: str, pitch: str) -> str:
    return "\x1f".join([text, voice, rate, volume, pitch])


# ── 对外接口 ────────────────────────────────────────────────────────────


async def synthesize_speech(
    text: str,
    voice: Optional[str] = None,
    rate: Optional[str] = None,
    volume: Optional[str] = None,
    pitch: Optional[str] = None,
    use_cache: bool = True,
) -> bytes:
    """合成语音，返回 mp3 字节。失败抛异常（调用方转 502）。"""
    text = (text or "").strip()
    if not text:
        raise ValueError("缺少文本")

    voice = voice or DEFAULT_VOICE
    rate = rate or DEFAULT_RATE
    volume = volume or DEFAULT_VOLUME
    pitch = pitch or DEFAULT_PITCH

    key = cache_path(utterance_key(text, voice, rate, volume, pitch))
    if use_cache:
        hit = load_cached(key)
        if hit:
            log().debug("TTS 命中缓存: %s", text)
            return hit

    async with _synth_lock:
        if use_cache:  # 等锁期间可能已被别的请求合成好
            hit = load_cached(key)
            if hit:
                return hit
        audio = await _synthesize_uncached(text, voice, rate, volume, pitch)
        store_cached(key, audio)
        return audio


async def probe_endpoints(text: str = "播报测试，欢迎光临") -> List[Tuple[str, str, str]]:
    """逐个接入点实测，返回 (名称, 结果, 详情)。给自检脚本用。"""
    results: List[Tuple[str, str, str]] = []
    for name, url in _ENDPOINTS.values():
        try:
            audio = await asyncio.wait_for(
                _synth_once(url, text, DEFAULT_VOICE, DEFAULT_RATE, DEFAULT_VOLUME, DEFAULT_PITCH),
                timeout=TIMEOUT + 5,
            )
            results.append((name, "OK", f"{len(audio)} 字节"))
        except Exception as exc:  # noqa: BLE001
            results.append((name, "FAIL", f"{type(exc).__name__}: {exc}"))
    return results
