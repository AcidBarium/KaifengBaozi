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
- `backend/tts_phrases.py`：播报文案的分段拼接与前缀预生成。
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

### 分段拼接 + 前缀预生成

播报文案里带单号（`订单 67 完成` / `订单 68 完成`），只按整句缓存命中率≈0。
`backend/tts_phrases.py` 把文案拆成片段：

```
新订单 68，打包，鸡蛋汤1份，豆浆1份
└─前缀─┘  └备注┘ └─菜品片段─┘ └片段┘
```

* **前缀**（`订单 {号} 完成` / `新订单 {号}，` / `订单 {号} 更新，` / `订单 {号} 取消`）
  按当日单号范围**提前批量生成**；
* **备注**、**菜名+份数**这类片段按需生成后缓存；
* 之后绝大多数播报都是本地拼接、完全不联网。

拆解规则与 `static/kitchen.html` 的 `buildSpeech()` 一一对应；**拆不出来的文案自动回退整句合成**，
所以改前端文案不会把播报弄坏，最多是享受不到拼接优化。

**Edge 会把短句补齐到约 1.87 秒（尾部 1.1~1.4 秒静音）**，所以片段必须按 MP3 帧边界裁掉多余静音，
否则拼出来全是空拍（实测裸拼「订单+六十七+完成」要 5.6s，整句只要 2.1s）。裁剪两端各留一小段静音，
让拼接点落在静音里。实测拼好的真实播报 5.47s，反而比整句的 5.78s 略短。

先跑一次预生成（可重复执行，已存在的会跳过，中断了再跑就是续传）：

```bash
python backend/tts_phrases.py status              # 看缓存现状
python backend/tts_phrases.py prewarm             # 生成 1..TTS_MAX_ORDER_NO 的前缀
python backend/tts_phrases.py prewarm --max 25    # 只生成 1..25
python backend/tts_phrases.py demo                # 拼两条真实播报，落盘到 /tmp 试听
```

预生成按**使用频率从高到低**排序（先 `订单 N 完成`，最后 `订单 N 取消`），
默认范围 200 共 801 条，会持续一段时间（受 429 限流影响可能更久），建议打烊后跑或挂后台。

### 环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `TTS_VOICE` | `zh-CN-XiaoxiaoNeural` | 发音人 |
| `TTS_RATE` / `TTS_VOLUME` / `TTS_PITCH` | `+0%` / `+0%` / `+0Hz` | 语速/音量/音调 |
| `TTS_ATTEMPTS` | `3` | 每个接入点的重试次数 |
| `TTS_BACKOFF` | `0.6` | 首次退避秒数，指数增长 |
| `TTS_TIMEOUT` | `20` | 单次合成超时（秒） |
| `TTS_CACHE_DIR` | `<项目根>/tts-cache` | 音频缓存目录（需可写，已 gitignore） |
| `TTS_CACHE_TTL` | `0`（永久） | 缓存有效期（秒）。默认 `0` = **永久保留**；给秒数才按修改时间过期 |
| `TTS_CACHE` | `1` | 设 `0` 完全停用缓存（读写都不走缓存） |
| `TTS_THROTTLE_BACKOFF` | `4.0` | 遇到 429 时额外多等的秒数 |
| `TTS_ENDPOINT` | 空（auto） | 强制只用 `new` 或 `old` |
| `TTS_COMPOSE` | `1` | 设 `0` 完全关闭分段拼接，回到整句合成 |
| `TTS_MAX_ORDER_NO` | `200` | 预生成的单号上限（超过的仍按需合成并缓存） |
| `TTS_PAUSE_MS` | `170` | 片段末尾保留的停顿（毫秒） |
| `TTS_PREWARM` | `0` | 设 `1` 则开机在后台自动补齐前缀 |
| `TTS_PREWARM_INTERVAL` | `0.3` | 预生成每条之间的间隔（秒），避免 429 |

若 systemd 里开了 `ProtectSystem=strict` / `ReadOnlyPaths`，记得给缓存目录放行，
或把 `TTS_CACHE_DIR` 指到 `/var/cache/kaifengbaozi`。

### 注意

* **缓存默认永久保留**，不会被自动清理：预生成的前缀 801 条约 8 MB，菜品片段
  有界（11 个菜 × 份数），备注片段是前台手打的短语，几百条也才几 MB。
  按 100 单/天估计，一年也就几十 MB，不需要靠过期来省空间。
  真要限制就设 `TTS_CACHE_TTL`（秒），想彻底停用设 `TTS_CACHE=0`。
  缓存目录可以随时手删，删了就是冷启动，不会坏数据。
* 缓存键里含发音人/语速/音量/音调，所以**改这些参数会生成一套全新文件**，
  旧文件不会再被读到（也是因此不需要靠 TTL 来实现"换发音人生效"）。
* 分段拼接的音质**不如整句**：每个片段是独立朗读的，缺少整句的语调衔接。
  想对比就 `python backend/tts_phrases.py demo` 拼一条，再和整句合成比一比。
* `miniaudio` 只用于裁静音。装不上也能跑，只是片段会带上句间长静音（`TTS_COMPOSE=0` 可彻底关掉这一层）。

## 数据库位置
- SQLite 文件：`data.db`（与项目根目录同级）。删除可清空数据。

## 部署提示
- 树莓派建议使用 `--host 0.0.0.0` 监听局域网。
- 可用 `pm2`/`supervisor` 或 `systemd` 保持进程常驻。
- 若需 HTTPS，自行在网关/路由器或 Nginx 终端处理 TLS。
