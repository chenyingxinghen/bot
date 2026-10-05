#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SillyTavern + NapCat 集成「全流程连通性」测试（standalone，不依赖 nonebot）。

运行：
    python bot/tests/test_full_flow.py

前置：本地真实服务——
    - Ollama  (http://127.0.0.1:11434) 且已拉取酒馆模型
    - ComfyUI (http://127.0.0.1:8188) 且具备 workflow 里引用的 checkpoint
    - 真实 SillyTavern 角色卡目录（默认 G:/git_proj/SillyTavern/data）
每个环节独立捕获异常并汇总 PASS / FAIL（部分环节为 WARN），最后给出汇总。
不发送任何真实 QQ 消息——只验证「读卡→构造 prompt→调 Ollama→解析生图标记→队列→
ComfyUI 生图→命令解析→模式/角色组合」这条链路在真实依赖下是否打通。
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

BOT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOT_ROOT))

import httpx  # noqa: E402

from core.character.client import (  # noqa: E402
    SillyTavernClient,
    CharacterSelection,
    CharacterCard,
    build_persona_prompt,
    build_lorebook_note,
)
from core.queue.queue import InferenceQueue  # noqa: E402
from core.image.comfyui import ComfyUIClient, ComfyUISettings  # noqa: E402
from core.modes.manager import ModeManager, build_modes  # noqa: E402
from core.commands.parser import parse_command  # noqa: E402
from core.qq.interactions import (  # noqa: E402
    parse_reply_actions,
    ReplyAction,
    StickerCatalog,
)

RESULTS: list[tuple[str, str, str]] = []  # (link, status, detail)


def record(link: str, ok: bool, detail: str = "", warn: bool = False) -> None:
    status = "WARN" if warn else ("PASS" if ok else "FAIL")
    RESULTS.append((link, status, detail))
    mark = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️"}[status]
    print(f"  {mark} {link}" + (f"：{detail}" if detail else ""))


# ─── 环境读取 ─────────────────────────────────────────────

def load_env_values(path: Path) -> dict:
    """极简 .env 解析：忽略注释与空行，KEY=VALUE。"""
    out: dict = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip().lower()] = v.strip()
    return out


def resolve_model(configured: str) -> str:
    """确认 configured 模型在 Ollama 中可用；否则按名称做最佳匹配回退。"""
    try:
        resp = httpx.get("http://127.0.0.1:11434/api/tags", timeout=10)
        tags = [m["name"] for m in resp.json().get("models", [])]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"无法连接 Ollama 获取模型列表：{exc}")
    if configured in tags:
        return configured
    base = configured.split(":")[0].lower()
    for t in tags:
        if t.lower() == base or base in t.lower():
            return t
    raise RuntimeError(
        f"Ollama 中既没有 '{configured}'，也没有匹配 '{base}' 的模型。"
        f"已安装：{tags}"
    )


# ─── 各链路测试 ────────────────────────────────────────────

async def link_card_discovery(st: SillyTavernClient) -> None:
    cards = st.discover_characters()
    names = st.list_character_names()
    record("A. 角色卡发现", bool(cards), f"发现 {len(cards)} 张：{', '.join(names)[:80]}")
    if not cards:
        return
    card = st.resolve_character("Bocchi") or st.resolve_character("auto")
    if card is None:
        record("A1. 解析 Bocchi", False, "resolve_character 返回 None")
        return
    record("A1. 解析 Bocchi", True,
           f"name={card.name!r} 有描述={'是' if card.description else '否'} "
           f"有世界书={'是' if card.has_lorebook else '否'}")


def link_persona_and_lore() -> None:
    card = CharacterCard(
        name="波奇酱",
        description="{{char}}是一名害羞的吉他手，{{user}}是她的室友",
        personality="内向、容易紧张",
        scenario="放学后的寝室",
        character_book={"entries": [
            {"keys": ["线索", "案件"], "content": "世界书：这是一桩密室谋杀案",
             "insertion_order": 0},
        ]},
    )
    persona = build_persona_prompt(card, "用户")
    no_ph = "{{char}}" not in persona and "{{user}}" not in persona
    has_name = "波奇酱" in persona
    record("B. 角色设定 prompt", no_ph and has_name,
           f"占位符已替换={no_ph} 角色名注入={has_name}")

    hit = build_lorebook_note(card, "今天的线索指向管家的房间")
    miss = build_lorebook_note(card, "今天天气真好")
    record("B1. 世界书命中", hit is not None and miss is None,
           f"命中={'是' if hit else '否'} 未命中={'是' if miss is None else '否'}")


async def link_ollama_direct(st: SillyTavernClient, model: str) -> None:
    messages = [
        {"role": "system", "content": "你是波奇酱，害羞的吉他手，用第一人称简短回应"},
        {"role": "user", "content": "嗨，今晚要不要一起练习吉他？"},
    ]
    try:
        reply = await st.generate_via_ollama(messages, model, 0.9, 120)
    except Exception as exc:  # noqa: BLE001
        record("D. Ollama 直连生成", False, f"{type(exc).__name__}: {exc}")
        return
    ok = bool(reply) and reply != ""
    record("D. Ollama 直连生成", ok, f"回复 {len(reply)} 字：{reply[:60]!r}")


async def link_proxy_fallback() -> None:
    """配置一个不可达的 SillyTavern 服务端，验证 chat() 自动回退 Ollama。"""
    env = load_env_values(BOT_ROOT / ".env")
    model = resolve_model(env.get("mode_tavern_model", ""))
    client = SillyTavernClient(
        data_root=env.get("sillytavern_data_root", "G:/git_proj/SillyTavern/data"),
        user=env.get("sillytavern_user", "default-user"),
        ollama_api_base=env.get("llm_api_base", "http://127.0.0.1:11434").replace("/v1", ""),
        ollama_keep_alive=env.get("llm_keep_alive", "30m"),
        ollama_think=False,
        st_url="http://127.0.0.1:9/dead-sillytavern",  # 故意不可达
        st_reverse_proxy="http://127.0.0.1:11434/v1",
        st_model=model,
    )
    messages = [{"role": "user", "content": "你好，用一句话回应我"}]
    try:
        reply = await client.chat(messages, model, 0.7, 80)
    except Exception as exc:  # noqa: BLE001
        record("D2. 代理失败回退 Ollama", False, f"{type(exc).__name__}: {exc}")
        return
    record("D2. 代理失败回退 Ollama", bool(reply) and reply != "",
           f"经回退拿到回复 {len(reply)} 字：{reply[:50]!r}")


def link_image_marker() -> None:
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="stk_"))
    catalog = StickerCatalog(tmp)
    raw = "[[image: 一只橘猫在月光下的窗台]] 顺便说，今晚月色真美"
    actions = parse_reply_actions(raw, {}, catalog, max_chars=80, max_actions=6)
    has_img = any(a.image_prompt for a in actions)
    record("E. 生图标记 [[image:]]", has_img,
           f"解析出 {len(actions)} 个动作，生图提示={actions and actions[-1].image_prompt!r}")

    raw2 = "[[生图: 赛博朋克城市夜景]]"
    actions2 = parse_reply_actions(raw2, {}, catalog, max_chars=80, max_actions=6)
    has_img2 = any(a.image_prompt for a in actions2)
    record("E1. 生图标记 [[生图:]]", has_img2,
           f"生图提示={actions2 and actions2[-1].image_prompt!r}")


async def link_queue() -> None:
    # (a) 正常：2 worker + 2 pending，4 个快任务应全部处理完
    q: InferenceQueue = InferenceQueue(max_workers=2, max_pending=2)
    q.start()
    done: list[int] = []

    async def make_task(i: int):
        async def _run():
            await asyncio.sleep(0.01)
            done.append(i)
        return _run

    for i in range(4):
        await q.submit(await make_task(i), kind="reply")
        await asyncio.sleep(0)  # 让出控制权，使 worker 能及时消费队列
    await asyncio.sleep(0.3)
    await q.shutdown(timeout=10)
    a_ok = sorted(done) == [0, 1, 2, 3] and q.stats.completed == 4 and q.stats.rejected == 0

    # (b) 溢出拒绝：不启动 worker，直接验证「待处理上限」被触发时提交被拒绝
    #     max_workers=1, max_pending=1 即队列 maxsize=1，最多容纳 1 个等待任务
    async def _noop():
        return None

    q2: InferenceQueue = InferenceQueue(max_workers=1, max_pending=1)
    ok_first = await q2.submit(lambda: _noop(), kind="reply")    # 进入队列
    ok_second = await q2.submit(lambda: _noop(), kind="reply")   # 队列已满 → 拒绝
    ok_third = await q2.submit(lambda: _noop(), kind="reply")    # 队列已满 → 拒绝
    b_ok = (ok_first is True) and (ok_second is False) and (ok_third is False) \
        and q2.stats.rejected >= 2

    record("F. 推理队列并发", a_ok and b_ok,
           f"处理完 {done}；溢出被拒(是否拒绝={ok_second is False and ok_third is False})")


async def link_comfyui(cfg: ComfyUISettings) -> None:
    client = ComfyUIClient(cfg)
    # 先确认服务可达（连通性最低要求）
    try:
        async with httpx.AsyncClient(timeout=10) as hc:
            r = await hc.get(f"{cfg.url}/system_stats")
            reachable = r.status_code == 200
    except Exception as exc:  # noqa: BLE001
        record("G. ComfyUI 生图", False, f"服务不可达：{exc}")
        return
    if not reachable:
        record("G. ComfyUI 生图", False, "服务返回非 200")
        return
    try:
        paths = await client.generate(
            "a cute orange cat on a windowsill, soft morning light, masterpiece")
    except Exception as exc:  # noqa: BLE001
        record("G. ComfyUI 生图", False,
               f"服务可达但生成失败（多为缺少 workflow 引用的 checkpoint）："
               f"{type(exc).__name__}: {exc}", warn=True)
        return
    ok = bool(paths) and all(p.exists() for p in paths)
    record("G. ComfyUI 生图", ok,
           f"生成 {len(paths)} 张：{', '.join(str(p) for p in paths)[:80]}")


def link_commands() -> None:
    cases = [
        ("/角色 Bocchi", ("character", "Bocchi")),
        ("/角色", ("character", "")),
        ("/状态", ("status",)),
        ("/mode 酒馆", ("mode", "tavern")),
        ("酒馆", ("mode", "tavern")),
        ("/生图 夕阳下的海边", ("image", "夕阳下的海边")),
        ("生图 一只猫", ("image", "一只猫")),
    ]
    fails = []
    for raw, expected in cases:
        got = parse_command(raw)
        if got != expected:
            fails.append(f"{raw!r}→{got!r}(期望{expected!r})")
    record("H. 命令解析", not fails, "; ".join(fails) if fails else "全部匹配")


def link_mode_character_combo(st: SillyTavernClient, model: str) -> None:
    """模拟「角色 Bocchi」：选角色 + 进入 tavern 模式 + 按该键解析卡片。"""
    env = load_env_values(BOT_ROOT / ".env")
    fake = types.SimpleNamespace(
        llm_model=model, llm_vision_model=model,
        mode_tavern_model=model, mode_clone_model=model, mode_writer_model=model,
        llm_temperature=0.6, llm_num_predict=192,
        mode_clone_prompt_path=BOT_ROOT / "data" / "modes" / "clone.txt",
    )
    modes = build_modes(fake, BOT_ROOT)
    mm = ModeManager(modes, BOT_ROOT / "data" / "_test_modes_state.json", default="clone")
    sel = CharacterSelection(BOT_ROOT / "data" / "_test_tavern_chars.json")

    key = ("self1", "peer1")
    card = st.resolve_character("Bocchi")
    if card is None:
        record("I. 角色+模式组合", False, "无法解析 Bocchi 角色")
        return
    sel.set(key, card.name)
    mm.set(key, "tavern")

    # 回读，验证组合状态一致
    cur_mode = mm.get(key)
    resolved = st.resolve_character(sel.get(key))
    ok = (cur_mode.key == "tavern" and resolved is not None
          and resolved.name == card.name)
    record("I. 角色+模式组合", ok,
           f"模式={cur_mode.key} 角色={resolved.name if resolved else None}")


async def main() -> int:
    print("=== SillyTavern + NapCat 全流程连通性测试 ===\n")
    env = load_env_values(BOT_ROOT / ".env")
    try:
        model = resolve_model(env.get("mode_tavern_model", ""))
    except Exception as exc:  # noqa: BLE001
        print(f"模型解析失败，终止：{exc}")
        return 1
    print(f"使用酒馆模型：{model}\n")

    st = SillyTavernClient(
        data_root=env.get("sillytavern_data_root", "G:/git_proj/SillyTavern/data"),
        user=env.get("sillytavern_user", "default-user"),
        ollama_api_base=env.get("llm_api_base", "http://127.0.0.1:11434").replace("/v1", ""),
        ollama_keep_alive=env.get("llm_keep_alive", "30m"),
        ollama_think=False,
        st_url=env.get("sillytavern_url", "") or "",
    )

    # 1) 角色发现
    await link_card_discovery(st)
    link_persona_and_lore()
    # 2) 真实生成
    await link_ollama_direct(st, model)
    await link_proxy_fallback()
    # 3) 生图标记
    link_image_marker()
    # 4) 队列
    await link_queue()
    # 5) ComfyUI
    comfy_cfg = ComfyUISettings.from_config(types.SimpleNamespace(**env), BOT_ROOT)
    await link_comfyui(comfy_cfg)
    # 6) 命令 & 组合
    link_commands()
    link_mode_character_combo(st, model)

    # 汇总
    fails = [r for r in RESULTS if r[1] == "FAIL"]
    warns = [r for r in RESULTS if r[1] == "WARN"]
    passes = [r for r in RESULTS if r[1] == "PASS"]
    print("\n=== 汇总 ===")
    print(f"PASS={len(passes)}  WARN={len(warns)}  FAIL={len(fails)}")
    if fails:
        print("失败项：")
        for link, _, detail in fails:
            print(f"  - {link}: {detail}")
    if warns:
        print("告警项（服务连通但数据/资产缺失，不影响代码链路）：")
        for link, _, detail in warns:
            print(f"  - {link}: {detail}")
    # 清理测试态文件
    for f in ("data/_test_modes_state.json", "data/_test_tavern_chars.json"):
        p = BOT_ROOT / f
        if p.exists():
            try:
                p.unlink()
            except Exception:  # noqa: BLE001
                pass
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
