"""播报文案的分段合成与拼接：前缀预生成 + 备注/菜品片段缓存。

为什么需要这一层
----------------
播报文案里带单号（`订单 67 完成` / `订单 68 完成`），每换一单整句指纹就变，
所以只按整句缓存命中率≈0。这一层把文案拆成：

    新订单 {单号}，  +  打包  +  鸡蛋汤1份  +  豆浆1份  +  菜饼2份

其中**前缀**按当日单号范围（1..`TTS_MAX_ORDER_NO`，默认 200）提前批量生成，
**备注**和**菜名+份数**这类片段按需生成后缓存。之后绝大多数播报
（每天的单号就那几个，菜品也就那十来样）都是本地拼接、完全不联网。

拆解规则与 `static/kitchen.html` 的 `buildSpeech()` 严格对应：

    created  新订单 {号}，[{备注}，]{菜名}{份数}，…
    announce 订单 {号} 更新，[{备注}，]{菜名}{份数}，…
    status   订单 {号} 完成 / 订单 {号} 取消
    固定句   播报测试，欢迎光临

拆不出来的文案一律回退到整句合成，**正确性不依赖这里的规则**。

实测依据（2026-10）
-------------------
* 所有片段格式完全一致：MPEG2 Layer III / 48kbps / 24kHz / 单声道，
  无 ID3 头尾、帧连续无残留 → 字节级拼接出来是合法 MP3；
* 但 Edge 会把短句**补齐到约 1.87s**，尾部静音长达 1.1~1.4s。裸拼
  「订单+六十七+完成」要 5.6s，整句只要 2.1s（多 166%），所以必须裁静音；
* 两端各保留一小段静音，让拼接点落在静音里 —— 这样 Layer III 位池
  （bit reservoir）跨帧引用造成的不连续、以及编码器延迟，都听不出来。

依赖
----
裁静音需要 MP3 解码器，用 `miniaudio`（纯 pip 安装，自带 C 库，不需要系统 ffmpeg）。
装不上时自动降级为「不裁剪、直接拼接」：能用，只是句间停顿偏长。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:  # 以 `uvicorn backend.main:app` 方式导入
    from . import tts_endpoint as _tts
except ImportError:  # 直接 `python backend/tts_phrases.py` 运行
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import tts_endpoint as _tts  # type: ignore[no-redef]

try:
    import miniaudio  # 仅裁静音时用到，缺了也能跑
except Exception:  # noqa: BLE001 - 缺依赖时降级，不影响服务启动
    miniaudio = None  # type: ignore[assignment]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


# 当日单号上限：超过这个范围的前缀不预生成，但仍然按需合成并缓存。
MAX_ORDER_NO = max(1, _env_int("TTS_MAX_ORDER_NO", 200))
# 是否启用分段拼接（关掉就完全回到整句合成）。
COMPOSE = (os.getenv("TTS_COMPOSE", "1") or "1") not in {"0", "false", "False"}
# 片段末尾保留的停顿（毫秒）。既是逗号停顿，也用来盖住拼接点。
PAUSE_MS = max(0, _env_int("TTS_PAUSE_MS", 170))
# 预生成时的间隔（秒），避免触发 429。
PREWARM_INTERVAL = float(os.getenv("TTS_PREWARM_INTERVAL", "0.3") or 0.3)

_SEG_NAMESPACE = "segments"

FIXED_PHRASES: Tuple[str, ...] = ("播报测试，欢迎光临",)

# kitchen.html 里的三种前缀
_RE_HEAD_CREATED = re.compile(r"^新订单 (\d+)$")
_RE_HEAD_UPDATED = re.compile(r"^订单 (\d+) 更新$")
_RE_HEAD_STATUS = re.compile(r"^订单 (\d+) (?:完成|取消)$")


def split_segments(text: str) -> Optional[List[str]]:
    """把播报文案拆成可独立合成/缓存的片段；拆不出来返回 None。"""
    text = (text or "").strip()
    if not text:
        return None
    if text in FIXED_PHRASES or _RE_HEAD_STATUS.match(text):
        return [text]

    parts = [p.strip() for p in text.split("，") if p.strip()]
    if len(parts) < 2:
        return None
    head, tail = parts[0], parts[1:]
    if _RE_HEAD_CREATED.match(head) or _RE_HEAD_UPDATED.match(head):
        # 前缀带上逗号一起合成，逗号后的停顿才是自然的
        return [head + "，"] + tail
    return None


# ── MP3 帧解析与静音裁剪 ────────────────────────────────────────────────
#
# Edge TTS 固定输出 MPEG2 Layer III / 24kHz，1 帧 = 576 采样 = 24ms。
# 按帧边界裁剪，不需要重新编码，因此运行时不需要编码器。

_SR = 24000
_FRAME_SEC = 0.024
_SILENCE_RMS = 0.01  # 静音判定阈值（归一化幅度）
_BR_V1L3 = [None, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, None]
_BR_V2L3 = [None, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, None]
_SR_TABLE = {3: [44100, 48000, 32000, None], 2: [22050, 24000, 16000, None], 0: [11025, 12000, 8000, None]}


def frame_offsets(data: bytes) -> List[Tuple[int, int]]:
    """返回 [(帧起始偏移, 帧长)]。非帧字节会被跳过。"""
    frames: List[Tuple[int, int]] = []
    i, n = 0, len(data)
    while i + 4 <= n:
        if data[i] == 0xFF and (data[i + 1] & 0xE0) == 0xE0:
            h = int.from_bytes(data[i:i + 4], "big")
            version = (h >> 19) & 0b11
            layer = (h >> 17) & 0b11
            bitrate_index = (h >> 12) & 0b1111
            sr_index = (h >> 10) & 0b11
            padding = (h >> 9) & 1
            if version != 1 and layer == 1 and sr_index != 3:
                table = _BR_V1L3 if version == 3 else _BR_V2L3
                bitrate = table[bitrate_index]
                sample_rate = _SR_TABLE[version][sr_index]
                if bitrate and sample_rate:
                    length = (144 if version == 3 else 72) * bitrate * 1000 // sample_rate + padding
                    if i + length <= n:
                        frames.append((i, length))
                        i += length
                        continue
        i += 1
    return frames


def _speech_seconds(mp3: bytes) -> Optional[Tuple[float, float]]:
    """解码后返回 (第一处非静音的时刻, 最后一处非静音的结束时刻)。"""
    if miniaudio is None:
        return None
    try:
        decoded = miniaudio.decode(
            mp3,
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=1,
            sample_rate=_SR,
        )
    except Exception:  # noqa: BLE001 - 解码失败就当裁不了
        return None
    pcm = decoded.samples
    window = _SR // 100  # 10ms
    if len(pcm) < window:
        return None
    first = last = None
    for start in range(0, len(pcm) - window + 1, window):
        block = pcm[start:start + window]
        rms = (sum(float(x) * x for x in block) / len(block)) ** 0.5 / 32768.0
        if rms > _SILENCE_RMS:
            if first is None:
                first = start
            last = start + window
    if first is None or last is None:
        return None
    return first / _SR, last / _SR


def trim_silence(mp3: bytes, pause_ms: Optional[int] = None) -> bytes:
    """按帧边界裁掉首尾多余静音。

    头部留 2 帧、尾部留 `pause_ms`（默认 TTS_PAUSE_MS），让拼接点落在静音里。
    无法解码或解析时原样返回 —— 宁可不裁，也不能把音频弄坏。
    """
    frames = frame_offsets(mp3)
    if len(frames) < 4:
        return mp3
    bounds = _speech_seconds(mp3)
    if bounds is None:
        return mp3
    first_s, last_s = bounds

    lead_frames = 2
    tail_frames = max(0, round((pause_ms if pause_ms is not None else PAUSE_MS) / 1000 / _FRAME_SEC))
    start_frame = max(0, int(first_s / _FRAME_SEC) - lead_frames)
    end_frame = min(len(frames), int(last_s / _FRAME_SEC) + 1 + tail_frames)
    if end_frame <= start_frame:
        return mp3
    start = frames[start_frame][0]
    end_offset, end_length = frames[end_frame - 1]
    return mp3[start:end_offset + end_length]


# ── 片段缓存与拼接 ──────────────────────────────────────────────────────


def _segment_path(text: str, voice: str) -> Path:
    return _tts.cache_path(_tts.utterance_key(text, voice, *_tts_params()), namespace=_SEG_NAMESPACE)


def _tts_params() -> Tuple[str, str, str]:
    return _tts.DEFAULT_RATE, _tts.DEFAULT_VOLUME, _tts.DEFAULT_PITCH


async def segment(text: str, voice: Optional[str] = None, force: bool = False) -> bytes:
    """取一个片段：命中缓存直接用，否则合成 + 裁静音 + 落盘。"""
    voice = voice or _tts.DEFAULT_VOICE
    path = _segment_path(text, voice)
    if not force:
        hit = _tts.load_cached(path)
        if hit:
            return hit
    # use_cache=False：片段缓存存的是裁过的，没必要再存一份没裁的整句
    raw = await _tts.synthesize_speech(text, voice, use_cache=False)
    audio = trim_silence(raw)
    _tts.store_cached(path, audio)
    return audio


async def synthesize(text: str, voice: Optional[str] = None) -> bytes:
    """播报入口：能拆就分段拼接，拆不开或失败就整句合成。"""
    text = (text or "").strip()
    if not text:
        raise ValueError("缺少文本")

    if COMPOSE:
        segments = split_segments(text)
        if segments:
            try:
                parts = [await segment(s, voice) for s in segments]
                joined = b"".join(parts)
                if len(segments) > 1:
                    _tts.log().info(
                        "TTS 分段拼接: %d 段 %d 字节 -> %s", len(segments), len(joined), text
                    )
                return joined
            except Exception as exc:  # noqa: BLE001 - 分段失败不该让播报整体失败
                _tts.log().warning("TTS 分段拼接失败，回退整句: %s: %s", type(exc).__name__, exc)

    return await _tts.synthesize_speech(text, voice)


# ── 预生成 ──────────────────────────────────────────────────────────────


# 预生成顺序 = 使用频率从高到低。预生成可能要跑很久，中途被打断时
# 先做的是最常用的模板，而不是"单号 1 的四种模板"。
_PREFIX_TEMPLATES: Tuple[str, ...] = (
    "订单 {n} 完成",   # 出餐，最高频
    "新订单 {n}，",    # 来单
    "订单 {n} 更新，",  # 前台点"提交修改"
    "订单 {n} 取消",   # 最少
)


def planned_prefixes(max_no: Optional[int] = None) -> List[str]:
    """需要预生成的全部前缀，顺序即生成顺序。"""
    limit = max(1, max_no if max_no is not None else MAX_ORDER_NO)
    texts: List[str] = []
    for template in _PREFIX_TEMPLATES:
        texts.extend(template.format(n=n) for n in range(1, limit + 1))
    texts.extend(FIXED_PHRASES)
    return texts


async def prewarm(
    max_no: Optional[int] = None,
    interval: Optional[float] = None,
    force: bool = False,
    voice: Optional[str] = None,
) -> Dict[str, int]:
    """批量生成前缀。已存在的默认跳过（可重复执行/断点续跑）。

    失败项在末尾再补一轮 —— 429 限流是成阵的，隔一会儿重试通常就过了。
    """
    gap = PREWARM_INTERVAL if interval is None else interval
    texts = planned_prefixes(max_no)
    voice = voice or _tts.DEFAULT_VOICE
    done = skipped = failed = 0
    pending: List[str] = []
    total = len(texts)
    log = _tts.log()
    for i, text in enumerate(texts, 1):
        if not force and _tts.load_cached(_segment_path(text, voice)):
            skipped += 1
            continue
        try:
            await segment(text, voice)
            done += 1
        except Exception as exc:  # noqa: BLE001 - 单条失败不该中断整批
            failed += 1
            pending.append(text)
            log.warning("预生成失败 %r: %s: %s", text, type(exc).__name__, exc)
        if i % 50 == 0 or i == total:
            log.info("预生成进度 %d/%d（新增 %d，跳过 %d，失败 %d）", i, total, done, skipped, failed)
        if gap:
            await asyncio.sleep(gap)

    for text in list(pending):
        if gap:
            await asyncio.sleep(gap)
        try:
            await segment(text, voice)
            done += 1
            failed -= 1
            pending.remove(text)
            log.info("补生成成功 %r", text)
        except Exception as exc:  # noqa: BLE001
            log.warning("补生成仍失败 %r: %s: %s", text, type(exc).__name__, exc)

    if pending:
        log.warning("预生成结束，仍有 %d 条失败（重新执行本命令即可续传）: %s", len(pending), "、".join(pending[:5]))
    return {"total": total, "done": done, "skipped": skipped, "failed": failed}


def stats() -> Dict[str, object]:
    """看一眼片段缓存现状。"""
    directory = _tts.CACHE_DIR / _SEG_NAMESPACE
    files = list(directory.glob("*.mp3")) if directory.exists() else []
    size = sum(f.stat().st_size for f in files)
    planned = planned_prefixes()
    ready = sum(1 for t in planned if _tts.load_cached(_segment_path(t, _tts.DEFAULT_VOICE)))
    return {
        "dir": str(directory),
        "files": len(files),
        "mb": round(size / 1024 / 1024, 2),
        "planned": len(planned),
        "ready": ready,
        "max_order_no": MAX_ORDER_NO,
        "compose": COMPOSE,
        "pause_ms": PAUSE_MS,
        "decoder": "miniaudio" if miniaudio is not None else "缺失(降级为不裁剪)",
    }


# ── 命令行 ──────────────────────────────────────────────────────────────


def _main() -> int:
    parser = argparse.ArgumentParser(description="播报前缀预生成 / 自检")
    parser.add_argument("action", nargs="?", default="status", choices=["status", "prewarm", "demo"])
    parser.add_argument("--max", type=int, default=None, help="单号上限（默认 TTS_MAX_ORDER_NO）")
    parser.add_argument("--interval", type=float, default=None, help="每条间隔秒数")
    parser.add_argument("--force", action="store_true", help="已存在的也重新生成")
    args = parser.parse_args()

    if args.action == "status":
        for k, v in stats().items():
            print(f"  {k:14} {v}")
        return 0

    if args.action == "prewarm":
        print(f"开始预生成前缀 1..{args.max or MAX_ORDER_NO}（间隔 {args.interval if args.interval is not None else PREWARM_INTERVAL}s）")
        result = asyncio.run(prewarm(args.max, args.interval, args.force))
        print(f"完成: {result}")
        return 0 if not result["failed"] else 1

    # demo：拼两条真实播报，落盘到 /tmp 供试听
    import tempfile
    import time

    cases = [
        ("订单 68 完成", "整句预生成"),
        ("新订单 68，打包，鸡蛋汤1份，豆浆1份，菜饼2份", "分段拼接"),
        ("新订单 69，灌汤包1份", "前缀+单品"),
    ]
    out_dir = Path(tempfile.gettempdir())
    for text, kind in cases:
        started = time.time()
        audio = asyncio.run(synthesize(text))
        elapsed = time.time() - started
        path = out_dir / f"demo_{abs(hash(text)) % 10000}_{kind}.mp3"
        path.write_bytes(audio)
        print(f"  [{kind}] {len(audio):6}B {elapsed:5.2f}s  {text}\n      -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
