#!/usr/bin/env python3
"""在服务器上跑这个脚本，确认 TTS 到底哪一步坏了、哪个接入点可用。

    python3 check_tts.py

它会：
  1. 打印 edge-tts 版本与旧/新接入点地址；
  2. 分别实测旧地址和新地址（不写缓存）；
  3. 用 synthesize_speech() 走完整流程（重试+缓存）合成一句真实播报。

不需要启动 web 服务，不依赖 FastAPI。
"""

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import importlib.metadata as metadata  # noqa: E402

import tts_endpoint  # noqa: E402


def main() -> int:
    try:
        print("edge-tts 版本:", metadata.version("edge-tts"))
    except metadata.PackageNotFoundError:
        print("edge-tts 未安装")
        return 1

    print("python:", sys.version.split()[0])
    print("尝试顺序:", " -> ".join(n for n, _ in tts_endpoint.endpoint_order()))
    ttl = "永久" if tts_endpoint.CACHE_TTL <= 0 else f"{tts_endpoint.CACHE_TTL}s"
    enabled = "开启" if tts_endpoint.CACHE_ENABLED else "已停用"
    print("缓存目录:", tts_endpoint.CACHE_DIR, f"(TTL {ttl}, {enabled})")
    print()
    print("── 逐接入点实测 ──")
    results = asyncio.run(tts_endpoint.probe_endpoints())
    for name, status, detail in results:
        print(f"  [{status:4}] {name:26} {detail}")

    print()
    print("── 走完整流程（重试+缓存）──")
    import time

    try:
        started = time.time()
        audio = asyncio.run(tts_endpoint.synthesize_speech("播报测试，欢迎光临，订单 66 完成"))
        elapsed = time.time() - started
        out = Path(tempfile.gettempdir()) / "check_tts_sample.mp3"
        out.write_bytes(audio)
        print(f"  [OK]   {len(audio)} 字节, {elapsed:.2f}s -> {out}")
        print("  试听: scp 下来用播放器打开，或 ffprobe/浏览器直接播")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] {type(exc).__name__}: {exc}")
        return 1

    print()
    print("结论: TTS 链路可用。" if any(s == "OK" for _, s, _ in results) else "结论: 两个接入点都不通，看上面报错。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
