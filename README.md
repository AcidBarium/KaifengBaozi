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
  - 新订单或修改自动语音播报（浏览器需支持 SpeechSynthesis）。
  - 无需额外操作，仅查看。

## 目录
- `backend/main.py`：FastAPI 服务，SQLite 持久化，WebSocket 广播。
- `static/index.html`：前台页面。
- `static/kitchen.html`：后厨出餐屏。
- `static/styles.css`：统一样式。

## 数据库位置
- SQLite 文件：`data.db`（与项目根目录同级）。删除可清空数据。

## 部署提示
- 树莓派建议使用 `--host 0.0.0.0` 监听局域网。
- 可用 `pm2`/`supervisor` 或 `systemd` 保持进程常驻。
- 若需 HTTPS，自行在网关/路由器或 Nginx 终端处理 TLS。
