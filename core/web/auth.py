"""Web 入口的简单账号与登录。

设计要点：
- 每个用户在注册时获得独立身份，聊天数据按用户名隔离（不再共用一个身份）。
- 注册时采集设备标识符（device_id）与客户端 IP，并落库。
- 每台设备只能注册一个账号（防滥用）；另受全局账号上限 ``max_accounts`` 约束，
  个人 BOT 场景默认上限为 1（即全实例仅允许一个账号）。
- 登录成功后签发 HMAC 签名的无状态会话令牌，WebSocket 据此鉴权。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_MAX_ACCOUNTS = 1
SESSION_TTL = 60 * 60 * 24 * 30  # 30 天


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + pad)


def hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000).hex()


def verify_password(password: str, salt: bytes, expected: str) -> bool:
    return hmac.compare_digest(hash_password(password, salt), expected)


@dataclass
class Account:
    username: str
    salt: str
    password_hash: str
    device_id: str
    register_ip: str
    register_at: float
    last_login_ip: str = ""
    last_login_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "username": self.username,
            "salt": self.salt,
            "password_hash": self.password_hash,
            "device_id": self.device_id,
            "register_ip": self.register_ip,
            "register_at": self.register_at,
            "last_login_ip": self.last_login_ip,
            "last_login_at": self.last_login_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Account":
        return cls(
            username=str(data.get("username", "")),
            salt=str(data.get("salt", "")),
            password_hash=str(data.get("password_hash", "")),
            device_id=str(data.get("device_id", "")),
            register_ip=str(data.get("register_ip", "")),
            register_at=float(data.get("register_at", 0.0) or 0.0),
            last_login_ip=str(data.get("last_login_ip", "")),
            last_login_at=float(data.get("last_login_at", 0.0) or 0.0),
        )


class AccountStore:
    """基于 JSON 文件的轻量账号存储，线程安全。"""

    def __init__(
        self,
        path: Path,
        secret: str,
        max_accounts: int = DEFAULT_MAX_ACCOUNTS,
    ) -> None:
        self.path = Path(path)
        self.secret = secret.encode("utf-8")
        self.max_accounts = max_accounts
        self._lock = threading.Lock()
        self._accounts: dict[str, Account] = {}
        self._device_index: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            raw = {}
        for item in raw.get("accounts", []):
            account = Account.from_dict(item)
            self._accounts[account.username] = account
            if account.device_id:
                self._device_index[account.device_id] = account.username

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"accounts": [acc.to_dict() for acc in self._accounts.values()]}
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(self.path)

    # ---- 注册 / 登录 ----

    def register(
        self, username: str, password: str, device_id: str, ip: str
    ) -> tuple[bool, str, Optional[Account]]:
        username = (username or "").strip()
        if not (3 <= len(username) <= 32 and username.replace("_", "").isalnum()):
            return False, "用户名需为 3-32 位字母、数字或下划线", None
        if len(password or "") < 6:
            return False, "密码至少 6 位", None
        with self._lock:
            if device_id and device_id in self._device_index:
                return False, "该设备已注册过账号，无法重复注册", None
            if username in self._accounts:
                return False, "该用户名已被占用", None
            if self.max_accounts and len(self._accounts) >= self.max_accounts:
                return False, "已达到最大账号数，停止注册", None
            salt = secrets.token_bytes(16)
            account = Account(
                username=username,
                salt=salt.hex(),
                password_hash=hash_password(password, salt),
                device_id=device_id or "",
                register_ip=ip or "",
                register_at=time.time(),
            )
            self._accounts[username] = account
            if device_id:
                self._device_index[device_id] = username
            self._save()
            return True, "", account

    def login(
        self, username: str, password: str, ip: str, device_id: str = ""
    ) -> tuple[bool, str, Optional[Account]]:
        """校验账号密码并记录登录元数据。

        ``device_id`` 与 IP 当前不作为强制登录条件：设备标识用于限制重复注册，
        IP 用于审计最近登录来源。严格绑定会导致动态 IP、浏览器清理 localStorage
        或更换设备后无法登录，应由独立配置显式启用，而不是隐式改变现有账号行为。
        """
        username = (username or "").strip()
        with self._lock:
            account = self._accounts.get(username)
            if account is None or not verify_password(
                password or "", bytes.fromhex(account.salt), account.password_hash
            ):
                return False, "用户名或密码错误", None
            account.last_login_ip = ip or account.last_login_ip
            account.last_login_at = time.time()
            if device_id and not account.device_id:
                account.device_id = device_id
                self._device_index[device_id] = username
            # 即使账号注册时已有 device_id，也必须持久化 last_login_ip/at。
            self._save()
            return True, "", account

    def count(self) -> int:
        return len(self._accounts)

    # ---- 会话令牌 ----

    def create_token(self, username: str, ttl: int = SESSION_TTL) -> str:
        payload = json.dumps({"u": username, "exp": int(time.time()) + ttl}).encode(
            "utf-8"
        )
        data = _b64url_encode(payload)
        signature = hmac.new(
            self.secret, data.encode("ascii"), hashlib.sha256
        ).hexdigest()
        return f"{data}.{signature}"

    def verify_token(self, token: str | None) -> Optional[str]:
        if not token or "." not in token:
            return None
        data, signature = token.rsplit(".", 1)
        expected = hmac.new(
            self.secret, data.encode("ascii"), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expected, signature):
            return None
        try:
            payload = json.loads(_b64url_decode(data))
        except Exception:
            return None
        if float(payload.get("exp", 0)) < time.time():
            return None
        username = str(payload.get("u", ""))
        return username if username in self._accounts else None
