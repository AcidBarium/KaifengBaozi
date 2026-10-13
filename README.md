# 包子店前台/后厨点餐 Demo

Python + FastAPI + SQLite，小屏设备（平板/树莓派显示器）可直接浏览器使用。

## 快速运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

打开浏览器：
- 前台：`http://<树莓派IP>:8000/`
- 后厨：`http://<树莓派IP>:8000/kitchen`

默认密码（可用环境变量覆盖）：
- 前台：`front123` 环境变量 `FRONT_PASSWORD`
- 后厨：`kitchen123` 环境变量 `KITCHEN_PASSWORD`

## 功能
- 前台：
  - 登录后创建订单、添加/减少菜品、备注、完成/取消。
  - 菜品：包子、油饼、菜饼、煎鸡蛋（价格可在 `backend/main.py` 的 `MENU` 修改）。
  - 实时同步，7天收入汇总。支持平板触控，大按钮。
- 后厨：
  - 登录后自动拉取订单并通过 WebSocket 实时更新。
  - 新订单或修改自动语音播报：后端用 Microsoft Edge TTS（`edge-tts`，无需 API Key）合成，浏览器 SpeechSynthesis 作为兜底。
  - 无需额外操作，仅查看。

## 目录
- `backend/main.py`：FastAPI 服务，SQLite 持久化，WebSocket 广播。
- `backend/tts_endpoint.py`：Edge TTS 合成封装（接入点切换、重试、缓存）。
- `backend/check_tts.py`：TTS 自检脚本，排障时先跑它。
- `static/index.html`：前台页面。
- `static/kitchen.html`：后厨出餐屏。
- `static/styles.css`：统一样式。

## 语音播报（TTS）

合成走 Microsoft Edge TTS，无需 API Key。2026 年 10 月微软迁移了合成接入点：

| 接入点 | 状态 |
| --- | --- |
| `speech.platform.bing.com`（旧） | 返回 403，已不可用（发音人列表仍在旧地址） |
| `api.msedgeservices.com`（新） | 可用，但偶发连不上 / 429 限流 |

`edge-tts` 库（含最新版）里地址仍是写死的旧地址，所以必须由 `backend/tts_endpoint.py`
把 `WSS_URL` 切到新地址；该模块同时提供**指数退避重试**、**旧地址兜底**、
**磁盘缓存**和**串行化**（降低 429 概率）。

排障时先跑自检（不用起 web 服务）：

```bash
python backend/check_tts.py
```

可调环境变量（都有默认值，不配也能跑）：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `TTS_VOICE` | `zh-CN-XiaoxiaoNeural` | 发音人 |
| `TTS_RATE` / `TTS_VOLUME` / `TTS_PITCH` | `+0%` / `+0%` / `+0Hz` | 语速/音量/音调 |
| `TTS_ATTEMPTS` | `3` | 每个接入点的重试次数 |
| `TTS_BACKOFF` | `0.6` | 首次退避秒数，指数增长 |
| `TTS_TIMEOUT` | `20` | 单次合成超时（秒） |
| `TTS_CACHE_DIR` | `<项目根>/tts-cache` | 音频缓存目录（需可写，已 gitignore） |
| `TTS_CACHE_TTL` | `604800` | 缓存有效期（秒），设 `0` 关闭缓存 |
| `TTS_ENDPOINT` | 空（auto） | 强制只用 `new` 或 `old` |

若 systemd 里开了 `ProtectSystem=strict` / `ReadOnlyPaths`，记得给缓存目录放行，
或把 `TTS_CACHE_DIR` 指到 `/var/cache/kaifengbaozi`。

## 数据库位置
- SQLite 文件：`data.db`（与项目根目录同级）。删除可清空数据。

## 部署提示
- 树莓派建议使用 `--host 0.0.0.0` 监听局域网。
- 可用 `pm2`/`supervisor` 或 `systemd` 保持进程常驻。
- 若需 HTTPS，自行在网关/路由器或 Nginx 终端处理 TLS。
