"""将私人 Web/PWA 聊天入口挂载到现有 FastAPI 应用。

新增简单登录：注册/登录接口签发会话令牌，WebSocket 凭令牌鉴权。
每个账号拥有独立身份（``web:{username}`` / ``conversation_id=username``），
因此不同用户的数据不再混在一起。注册时采集设备标识符与客户端 IP，
每台设备仅允许注册一次，并受全局账号上限约束。
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from config.platforms import WebSettings
from core.image.service import ImageService
from core.platform.dispatcher import MessageDispatcher
from core.platform.models import IncomingMessage
from core.qq.interactions import StickerCatalog
from core.web.auth import AccountStore
from core.web.history import WebHistoryStore
from core.web.sender import WebSocketSender

logger = logging.getLogger(__name__)


def _client_ip(request: Request) -> str:
    """尽量还原真实客户端 IP（兼容反向代理）。"""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    return request.client.host if request.client else ""


def _get_store(app: FastAPI, settings: WebSettings) -> AccountStore:
    """在 app.state 上惰性构建并复用账号存储（install 幂等）。"""
    store: AccountStore | None = getattr(app.state, "chat_web_accounts", None)
    if store is None:
        store = AccountStore(
            settings.accounts_file,
            secret=settings.token or "insecure-default-secret",
            max_accounts=settings.max_accounts,
        )
        app.state.chat_web_accounts = store
    return store


def _get_history(app: FastAPI, settings: WebSettings) -> WebHistoryStore:
    """在 app.state 上复用 Web 实际投递历史存储。"""
    history: WebHistoryStore | None = getattr(app.state, "chat_web_history", None)
    if history is None:
        history = WebHistoryStore(settings.history_db)
        app.state.chat_web_history = history
    return history


def install_web_routes(
    app: FastAPI,
    settings: WebSettings,
    dispatcher: MessageDispatcher,
    image_service: ImageService,
    stickers: StickerCatalog,
    *,
    stream_pause: float = 0.0,
) -> None:
    """注册 Web 页面、静态资源和 WebSocket；可重复调用但只安装一次。"""
    if not settings.enabled:
        logger.info("Web 入口：关闭")
        return
    if not settings.token:
        logger.error("Web 入口已启用但 WEB_TOKEN 为空，为安全起见不启动")
        return
    if getattr(app.state, "chat_web_installed", False):
        return

    static_dir = Path(settings.static_dir)
    required = ("index.html", "app.js", "styles.css", "manifest.webmanifest", "icon.svg")
    missing = [name for name in required if not (static_dir / name).is_file()]
    if missing:
        logger.error("Web 入口静态文件缺失：%s", ", ".join(missing))
        return

    root = settings.path

    @app.get(root, include_in_schema=False)
    @app.get(f"{root}/", include_in_schema=False)
    async def chat_page():
        # 同时兼容内部挂载路径的有/无尾斜杠；公网由 Caddy 统一为 /bot/。
        return FileResponse(static_dir / "index.html")

    @app.get(f"{root}/app.js", include_in_schema=False)
    async def chat_script():
        return FileResponse(static_dir / "app.js", media_type="text/javascript")

    @app.get(f"{root}/styles.css", include_in_schema=False)
    async def chat_styles():
        return FileResponse(static_dir / "styles.css", media_type="text/css")

    @app.get(f"{root}/manifest.webmanifest", include_in_schema=False)
    async def chat_manifest():
        return FileResponse(
            static_dir / "manifest.webmanifest",
            media_type="application/manifest+json",
        )

    @app.get(f"{root}/icon.svg", include_in_schema=False)
    async def chat_icon():
        return FileResponse(static_dir / "icon.svg", media_type="image/svg+xml")

    @app.get(f"{root}/health", include_in_schema=False)
    async def chat_health():
        return JSONResponse({"ok": True, "platform": "web"})

    @app.post(f"{root}/register", include_in_schema=False)
    async def chat_register(request: Request):
        store = _get_store(app, settings)
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"ok": False, "error": "请求格式错误"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"ok": False, "error": "请求格式错误"}, status_code=400)
        username = str(body.get("username", "")).strip()
        password = str(body.get("password", ""))
        device_id = str(body.get("device_id", "")).strip()
        ok, error, account = store.register(
            username, password, device_id, _client_ip(request)
        )
        if not ok:
            return JSONResponse({"ok": False, "error": error}, status_code=409)
        token = store.create_token(account.username)
        return JSONResponse(
            {"ok": True, "token": token, "username": account.username}
        )

    @app.post(f"{root}/login", include_in_schema=False)
    async def chat_login(request: Request):
        store = _get_store(app, settings)
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"ok": False, "error": "请求格式错误"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"ok": False, "error": "请求格式错误"}, status_code=400)
        username = str(body.get("username", "")).strip()
        password = str(body.get("password", ""))
        device_id = str(body.get("device_id", "")).strip()
        ok, error, account = store.login(
            username, password, _client_ip(request), device_id
        )
        if not ok:
            return JSONResponse({"ok": False, "error": error}, status_code=401)
        token = store.create_token(account.username)
        return JSONResponse(
            {"ok": True, "token": token, "username": account.username}
        )

    @app.get(f"{root}/me", include_in_schema=False)
    async def chat_me(request: Request):
        store = _get_store(app, settings)
        header = request.headers.get("authorization", "")
        token = header.replace("Bearer ", "", 1) if header.startswith("Bearer ") else ""
        username = store.verify_token(token)
        if not username:
            return JSONResponse({"ok": False}, status_code=401)
        return JSONResponse({"ok": True, "username": username})

    @app.websocket(f"{root}/ws")
    async def chat_socket(websocket: WebSocket):
        await websocket.accept()
        store = _get_store(app, settings)
        try:
            auth = await websocket.receive_json()
            token = str(auth.get("token", "")) if isinstance(auth, dict) else ""
            username = store.verify_token(token) if auth.get("type") == "auth" else None
            if not username:
                await websocket.send_json({"type": "auth_error"})
                await websocket.close(code=4401)
                return
            await websocket.send_json({"type": "ready", "username": username})
            history_store = _get_history(app, settings)

            def recent_history() -> list[dict]:
                try:
                    events = history_store.recent_rounds(username, rounds=2)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Web 最近投递历史读取失败：user=%s error=%s", username, exc
                    )
                    events = []
                if events:
                    return events
                # 新存储首次上线时保留旧 runtime 文本窗口作为一次性兼容回退；
                # 用户产生首条新 Web 轮次后即切换到包含图片的实际投递历史。
                try:
                    return dispatcher.conversation_service.recent_message_window(
                        "web:main", f"web:{username}", rounds=2
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Web 旧会话窗口兼容读取失败：user=%s error=%s", username, exc
                    )
                    return []

            await websocket.send_json(
                {"type": "history", "messages": recent_history()}
            )
            while True:
                payload = await websocket.receive_json()
                if isinstance(payload, dict) and payload.get("type") == "history_sync":
                    await websocket.send_json(
                        {"type": "history", "messages": recent_history()}
                    )
                    continue
                if not isinstance(payload, dict) or payload.get("type") != "message":
                    await websocket.send_json(
                        {"type": "notice", "text": "不支持的消息格式。"}
                    )
                    continue
                text = str(payload.get("text", "")).strip()
                raw_images = payload.get("images", [])
                images = tuple(
                    str(item) for item in raw_images
                    if isinstance(item, str) and item.startswith("data:image/")
                )[:4]
                raw_id = str(payload.get("id", "")).strip()
                turn_id = raw_id or f"server-{id(payload)}"
                try:
                    history_store.append_user_message(username, turn_id, text, images)
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "Web 用户消息持久化失败：user=%s turn=%s error=%s",
                        username,
                        turn_id,
                        exc,
                    )
                    await websocket.send_json(
                        {"type": "notice", "text": "消息保存失败，请稍后再试。"}
                    )
                    continue
                native_event = {"message_id": raw_id} if raw_id.isdigit() else None
                message = IncomingMessage(
                    platform="web",
                    self_id="web:main",
                    user_id=f"web:{username}",
                    text=text,
                    images=images,
                    message_id=raw_id or None,
                    conversation_id=username,
                    native_event=native_event,
                )
                sender = WebSocketSender(
                    websocket,
                    image_service,
                    stickers,
                    history_store,
                    username,
                    turn_id,
                    stream_pause=stream_pause,
                )
                if settings.stream:
                    await dispatcher.dispatch_stream(message, sender)
                else:
                    await dispatcher.dispatch(message, sender)
        except WebSocketDisconnect:
            logger.info("Web 客户端已断开：%s", "-")
        except Exception as exc:  # noqa: BLE001
            logger.warning("WebSocket 会话异常：%s", exc)
            try:
                await websocket.close(code=1011)
            except Exception:  # noqa: BLE001
                pass

    app.state.chat_web_installed = True
    logger.info("Web 入口已启用：%s", root)
