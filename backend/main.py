import asyncio
import json
import os
import secrets
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, create_engine, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship, sessionmaker

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data.db"
MENU_PATH = BASE_DIR / "menu.json"

TTS_API_URL = os.getenv("TTS_API_URL", "https://tts.wangwangit.com/v1/audio/speech")
TTS_DEFAULT_VOICE = os.getenv("TTS_VOICE", "zh-CN-XiaoxiaoNeural")
TTS_SPEED = float(os.getenv("TTS_SPEED", "1.0"))
TTS_PITCH = os.getenv("TTS_PITCH", "0")
TTS_STYLE = os.getenv("TTS_STYLE", "general")

DATABASE_URL = f"sqlite:///{DB_PATH}"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, index=True)
    status = Column(String, default="open")  # open, completed, cancelled
    note = Column(String, nullable=True)
    total = Column(Float, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    items = relationship("OrderItem", cascade="all, delete-orphan", back_populates="order")


class OrderItem(Base):
    __tablename__ = "order_items"

    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(Integer, ForeignKey("orders.id"))
    key = Column(String)
    name = Column(String)
    price = Column(Float)
    quantity = Column(Integer, default=1)

    order = relationship("Order", back_populates="items")


Base.metadata.create_all(bind=engine)


# Simple in-memory auth tokens per role
ACTIVE_TOKENS: Dict[str, str] = {}
TOKEN_LOCK = asyncio.Lock()
FRONT_PASSWORD = os.getenv("FRONT_PASSWORD", "lcx050409")
KITCHEN_PASSWORD = os.getenv("KITCHEN_PASSWORD", "kitchen123")


def load_menu_config() -> List[Dict]:
    if not MENU_PATH.exists():
        raise RuntimeError(f"menu config not found: {MENU_PATH}")
    try:
        data = json.loads(MENU_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - startup guard
        raise RuntimeError(f"failed to read menu: {exc}")
    if not isinstance(data, list):
        raise RuntimeError("menu config must be a list")
    menu: List[Dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        name = item.get("name")
        price = item.get("price")
        if not key or not name or price is None:
            continue
        menu.append({"key": str(key), "name": str(name), "price": float(price)})
    if not menu:
        raise RuntimeError("menu config is empty")
    return menu


# Menu definition (key is used internally)
MENU = load_menu_config()
MENU_PRICES = {item["key"]: item for item in MENU}


class LoginRequest(BaseModel):
    role: str
    password: str


class ItemInput(BaseModel):
    key: str
    quantity: int = 1


class OrderCreate(BaseModel):
    note: Optional[str] = None
    items: List[ItemInput] = []


class OrderUpdate(BaseModel):
    status: Optional[str] = None
    note: Optional[str] = None


class ItemUpdate(BaseModel):
    quantity: int


class StatsResponse(BaseModel):
    total: float
    orders: int


app = FastAPI(title="Baozhidian Quick Order")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

websocket_connections: List[WebSocket] = []


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def calc_display_no(order: Order, db: Session) -> int:
    # Daily sequence starting from 1 based on UTC date
    day_start = order.created_at.replace(hour=0, minute=0, second=0, microsecond=0)
    min_id = (
        db.query(func.min(Order.id))
        .filter(Order.created_at >= day_start)
        .filter(Order.created_at < day_start + timedelta(days=1))
        .scalar()
    )
    if not min_id:
        return order.id
    return max(1, order.id - int(min_id) + 1)


def serialize_order(order: Order, db: Session):
    return {
        "id": order.id,
        "display_no": calc_display_no(order, db),
        "status": order.status,
        "note": order.note or "",
        "total": round(order.total, 2),
        "created_at": order.created_at.isoformat(),
        "updated_at": order.updated_at.isoformat() if order.updated_at else None,
        "items": [
            {
                "id": item.id,
                "key": item.key,
                "name": item.name,
                "price": item.price,
                "quantity": item.quantity,
                "line_total": round(item.price * item.quantity, 2),
            }
            for item in order.items
        ],
    }


def recalc_total(order: Order):
    order.total = sum(item.price * item.quantity for item in order.items)


async def broadcast(message: Dict):
    stale: List[WebSocket] = []
    for ws in websocket_connections:
        try:
            await ws.send_json(message)
        except Exception:
            stale.append(ws)
    for ws in stale:
        websocket_connections.remove(ws)


async def auth_role(required_roles: List[str], authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="缺少凭证")
    token = authorization.split(" ", 1)[1]
    async with TOKEN_LOCK:
        role = ACTIVE_TOKENS.get(token)
    if not role or role not in required_roles:
        raise HTTPException(status_code=401, detail="无效凭证")
    return role


def require_roles(roles: List[str]):
    async def _dep(authorization: Optional[str] = Header(None)):
        return await auth_role(roles, authorization)

    return _dep


@app.post("/api/login")
async def login(payload: LoginRequest):
    role = payload.role.lower()
    if role not in {"front", "kitchen"}:
        raise HTTPException(status_code=400, detail="角色错误")
    if role == "front":
        expected = FRONT_PASSWORD
        if payload.password != expected:
            raise HTTPException(status_code=401, detail="密码错误")
    # kitchen 角色不再校验密码，便于后厨免登录展示
    token = secrets.token_hex(16)
    async with TOKEN_LOCK:
        ACTIVE_TOKENS[token] = role
    return {"token": token, "role": role}


@app.get("/api/menu")
async def get_menu():
    return MENU


@app.get("/api/orders")
async def list_orders(
    status: Optional[str] = None,
    db: Session = Depends(get_db),
    role: str = Depends(require_roles(["front", "kitchen"])),
):
    query = db.query(Order)
    if status:
        query = query.filter(Order.status == status)
    orders = query.order_by(Order.created_at.desc()).all()
    return [serialize_order(o, db) for o in orders]


@app.post("/api/orders")
async def create_order(
    payload: OrderCreate,
    db: Session = Depends(get_db),
    role: str = Depends(require_roles(["front"])),
):
    order = Order(status="open", note=payload.note or "")
    for item in payload.items:
        menu_item = MENU_PRICES.get(item.key)
        if not menu_item:
            raise HTTPException(status_code=400, detail="未知菜品")
        db_item = OrderItem(
            key=item.key,
            name=menu_item["name"],
            price=menu_item["price"],
            quantity=item.quantity,
        )
        order.items.append(db_item)
    recalc_total(order)
    db.add(order)
    db.commit()
    db.refresh(order)
    data = serialize_order(order, db)
    await broadcast({"type": "order_updated", "action": "created", "order": data})
    return data


@app.post("/api/orders/{order_id}/announce")
async def announce_order(
    order_id: int,
    db: Session = Depends(get_db),
    role: str = Depends(require_roles(["front"])),
):
    order = db.query(Order).filter(Order.id == order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="订单不存在")
    data = serialize_order(order, db)
    await broadcast({"type": "order_updated", "action": "announce", "order": data})
    return data


@app.post("/api/orders/{order_id}/items")
async def add_item(
    order_id: int,
    payload: ItemInput,
    db: Session = Depends(get_db),
    role: str = Depends(require_roles(["front"])),
):
    order = db.query(Order).filter(Order.id == order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="订单不存在")
    if order.status != "open":
        raise HTTPException(status_code=400, detail="订单已完成或取消")
    menu_item = MENU_PRICES.get(payload.key)
    if not menu_item:
        raise HTTPException(status_code=400, detail="未知菜品")
    existing = next((i for i in order.items if i.key == payload.key), None)
    if existing:
        existing.quantity += payload.quantity
    else:
        order.items.append(
            OrderItem(
                key=payload.key,
                name=menu_item["name"],
                price=menu_item["price"],
                quantity=payload.quantity,
            )
        )
    recalc_total(order)
    db.commit()
    db.refresh(order)
    data = serialize_order(order, db)
    await broadcast({"type": "order_updated", "action": "item_added", "order": data})
    return data


@app.patch("/api/orders/{order_id}/items/{item_id}")
async def update_item(
    order_id: int,
    item_id: int,
    payload: ItemUpdate,
    db: Session = Depends(get_db),
    role: str = Depends(require_roles(["front"])),
):
    order = db.query(Order).filter(Order.id == order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="订单不存在")
    item = db.query(OrderItem).filter(OrderItem.id == item_id, OrderItem.order_id == order_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="菜品不存在")
    if order.status != "open":
        raise HTTPException(status_code=400, detail="订单已完成或取消")
    if payload.quantity <= 0:
        db.delete(item)
    else:
        item.quantity = payload.quantity
    recalc_total(order)
    db.commit()
    db.refresh(order)
    data = serialize_order(order, db)
    await broadcast({"type": "order_updated", "action": "item_changed", "order": data})
    return data


@app.patch("/api/orders/{order_id}")
async def update_order(
    order_id: int,
    payload: OrderUpdate,
    db: Session = Depends(get_db),
    role: str = Depends(require_roles(["front"])),
):
    order = db.query(Order).filter(Order.id == order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="订单不存在")
    if payload.status:
        if payload.status not in {"open", "completed", "cancelled"}:
            raise HTTPException(status_code=400, detail="状态错误")
        order.status = payload.status
    if payload.note is not None:
        order.note = payload.note
    recalc_total(order)
    db.commit()
    db.refresh(order)
    data = serialize_order(order, db)
    await broadcast({"type": "order_updated", "action": "status_changed", "order": data})
    return data


@app.get("/api/stats/summary", response_model=StatsResponse)
async def stats(
    days: int = 1,
    db: Session = Depends(get_db),
    role: str = Depends(require_roles(["front"])),
):
    since = datetime.utcnow() - timedelta(days=days)
    q = db.query(func.sum(Order.total), func.count(Order.id)).filter(
        Order.status == "completed", Order.created_at >= since
    )
    total, count = q.first()
    return StatsResponse(total=round(total or 0, 2), orders=count or 0)


@app.get("/api/stats/overview")
async def stats_overview(
    db: Session = Depends(get_db),
    role: str = Depends(require_roles(["front"])),
):
    now = datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=6)

    def sum_and_count(since: datetime):
        q = db.query(func.sum(Order.total), func.count(Order.id)).filter(
            Order.status == "completed", Order.created_at >= since
        )
        total, cnt = q.first()
        return {"total": round(total or 0, 2), "orders": cnt or 0}

    # per day totals for the last 7 days including today
    daily_rows = (
        db.query(func.date(Order.created_at), func.sum(Order.total), func.count(Order.id))
        .filter(Order.status == "completed", Order.created_at >= week_start)
        .group_by(func.date(Order.created_at))
        .order_by(func.date(Order.created_at))
        .all()
    )
    daily_map = {row[0]: {"total": float(row[1] or 0), "orders": int(row[2] or 0)} for row in daily_rows}
    days = []
    for i in range(7):
        day = week_start + timedelta(days=i)
        key = day.date().isoformat()
        row = daily_map.get(key, {"total": 0.0, "orders": 0})
        days.append({"date": key, "total": round(row["total"], 2), "orders": row["orders"]})

    items_rows = (
        db.query(
            OrderItem.key,
            OrderItem.name,
            func.sum(OrderItem.quantity),
            func.sum(OrderItem.quantity * OrderItem.price),
        )
        .join(Order, OrderItem.order_id == Order.id)
        .filter(Order.status == "completed")
        .group_by(OrderItem.key, OrderItem.name)
        .order_by(OrderItem.name)
        .all()
    )
    items = [
        {
            "key": row[0],
            "name": row[1],
            "quantity": int(row[2] or 0),
            "revenue": round(float(row[3] or 0), 2),
        }
        for row in items_rows
    ]

    return {
        "today": sum_and_count(today_start),
        "week": sum_and_count(week_start),
        "days": days,
        "items": items,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=4401)
        return
    async with TOKEN_LOCK:
        role = ACTIVE_TOKENS.get(token)
    if role not in {"front", "kitchen"}:
        await websocket.close(code=4401)
        return
    await websocket.accept()
    websocket_connections.append(websocket)
    try:
        while True:
            # Keep the connection open; no incoming messages expected
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in websocket_connections:
            websocket_connections.remove(websocket)


@app.get("/")
async def root():
    return FileResponse(str(BASE_DIR / "static" / "index.html"))


@app.get("/kitchen")
async def kitchen_page():
    return FileResponse(str(BASE_DIR / "static" / "kitchen.html"))


@app.get("/test")
async def test_page():
    return FileResponse(str(BASE_DIR / "static" / "test.html"))


@app.get("/stats")
async def stats_page():
    return FileResponse(str(BASE_DIR / "static" / "stats.html"))



@app.get("/api/tts")
async def tts(text: str, voice: Optional[str] = None):
    text = (text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="缺少文本")
    if len(text) > 240:
        raise HTTPException(status_code=400, detail="文本过长")

    payload = {
        "input": text,
        "voice": voice or TTS_DEFAULT_VOICE,
        "speed": TTS_SPEED,
        "pitch": TTS_PITCH,
        "style": TTS_STYLE,
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(TTS_API_URL, json=payload)
    except httpx.RequestError:
        raise HTTPException(status_code=502, detail="TTS 服务不可用")

    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"TTS 生成失败: {resp.status_code}")

    media_type = resp.headers.get("content-type", "audio/mpeg")
    return StreamingResponse(BytesIO(resp.content), media_type=media_type, headers={"Cache-Control": "no-store"})

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
