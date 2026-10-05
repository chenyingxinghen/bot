# NapCat.Shell Bot

**跨平台多模式 AI 对话机器人**。QQ（OneBot V11）、Web/PWA、飞书三个入口共用同一套对话内核；领域层通过 `core/platform/` 的抽象与任何具体平台解耦，新增平台只需实现一个 Sender。

三种模式（酒馆角色扮演 / 人类模仿 / 小说作家）共享同一套消息收发、记忆与生图基础设施，区别只在「用哪个模型 + 哪套系统提示词 + 哪些控制标记」。

> **平台无关性**：领域代码不依赖 NapCat 或任何 QQ 实现。QQ 只是 `core/qq/adapter.py` 里的一个 OneBot V11 适配器；不开 QQ 时，Web 端可独立运行（`WEB_ENABLED=true`，无需任何 QQ 依赖）。
> `IncomingMessage` 用 `平台:原始ID` 作为身份键，避免多平台 ID 碰撞。

## 特性

- **三端接入**：QQ 私聊（OneBot V11）、Web/PWA（可装为桌面应用，支持流式回复）、飞书机器人，共享统一调度器。
- **三模式隔离**：`tavern` / `clone` / `writer` 各自独立的提示词、模型、长期记忆库，切换不串味。
- **分层记忆**：按 `模式 → 账号 → 会话 → 角色/作品` 四级隔离；长期记忆分 `person / self / relationship / episode` 类型，常驻层与召回层分离，在线抽取带语义去重。
- **SillyTavern 兼容**：酒馆模式支持角色卡 + 世界书（lorebook）命中注入，prompt 组装参照 SillyTavern 范式。
- **ComfyUI 生图**：LLM 可调用生图工具，支持尺寸/步数/CFG/采样器等参数与工作流节点寻址。
- **Web 鉴权与历史**：Web 端带账号注册/登录、会话历史持久化、WebSocket 流式输出。
- **平台无关内核**：`core/platform/` 定义统一入站消息模型与 `MessageDispatcher`，领域层不感知平台来源。

## 快速开始

```bash
# 1. 安装依赖（Python >= 3.10）
python -m venv .venv
.venv/Scripts/python -m pip install -e .        # Windows
# .venv/bin/pip install -e .                   # Linux / macOS

# 2. 配置
cp .env.example .env
# 编辑 .env：至少填写 LLM_API_BASE / LLM_MODEL；
# 启用哪个平台就开对应开关：WEB_ENABLED / FEISHU_ENABLED
#（QQ 端默认开启，如只用 Web 可不配 OneBot）

# 3. 运行
.venv/Scripts/python app.py
```

**按需启用平台**，三者互不依赖：

- **Web/PWA**：`.env` 设 `WEB_ENABLED=true` + `WEB_TOKEN`，浏览器访问 `http://<host>:<port><WEB_PATH>`（默认 `/bot`）。**无需任何 QQ 依赖**。
- **QQ**：需另行启动一个 OneBot V11 实现端（如 NapCat），在其中填入本服务地址与 access token。
- **飞书**：设 `FEISHU_ENABLED=true` + `FEISHU_APP_ID` / `FEISHU_APP_SECRET`，走长连接网关，无需公网地址。

## 模式与命令

| 模式 | 切换 | 模型 | 说明 |
|---|---|---|---|
| `tavern` 酒馆 | `/酒馆`、`/tavern` | 独立指定 | 沉浸式第一人称角色扮演，读角色卡 + 世界书，可生图 |
| `clone` 模仿 | `/模仿`、`/clone` | 视觉模型 | 模仿对方说话风格（默认模式） |
| `writer` 作家 | `/作家`、`/writer` | 写作模型 | 长篇小说创作，先出大纲再逐章写，可插图 |

常用命令（均支持中文别名）：

| 命令 | 作用 |
|---|---|
| `/状态` | 查看当前模式与推理队列状态 |
| `/记忆`、`/记忆 模式` | 只读回顾当前角色/作品、或整模式的记忆 |
| `/清记忆`、`/清记忆 模式` | 分层清除记忆（命名空间层 / 模式层） |
| `/帮助` | 分层帮助文本 |

三套系统提示词独立存放于 `data/modes/{tavern,clone,writer}.txt`，改提示词无需动代码。

## 架构

```text
QQ 私聊 ─┐
Web/PWA ─┼─> MessageDispatcher ─> ConversationService ─> LLM / Memory / Image
飞书 ────┘         （平台无关）        （业务编排）
```

- `app.py` 是薄入口：构建配置、装配服务、注册平台 handler，把事件翻译成领域调用。所有业务编排在 `core/` 内。
- `core/platform/` 定义平台无关的 `IncomingMessage` 与 `MessageDispatcher`；QQ / Web / 飞书只是三个 Sender 实现，新增平台无需改动领域层。
- `core/queue/` 的全局推理队列统一限流，避免回复、记忆抽取、生图并发互相抢占。
- 记忆抽取走同一队列，不绕过并发限制。

详见 `docs/MEMORY_SYSTEM.md`。

## 目录约定

本项目的目录结构遵循分层原则设计，确保易维护、易拓展：

```text
bot/
├── app.py                  # NoneBot 启动入口 + handler 注册（事件↔core 翻译，薄）。无业务编排。
├── pyproject.toml
├── start.bat
├── .env / .env.example / .env.prod     # 部署期值来源（密钥、模型名、路径开关）
│
├── config/                 # 【单一配置归宿】所有 Settings 定义 + .env 映射 + 组合 AppConfig
│   ├── __init__.py         # 导出 AppConfig；build_app_config(driver.config, bot_root)
│   ├── base.py             # BaseSettings 基类
│   ├── llm.py              # LLMSettings
│   ├── memory.py           # MemorySettings
│   ├── qq.py               # QQSettings
│   ├── platforms.py        # PlatformsSettings（web / feishu）
│   ├── image.py            # ImageSettings
│   ├── character.py        # CharacterSettings
│   ├── modes.py            # ModesSettings
│   └── conversation.py     # ConversationSettings
│
├── core/                   # 【领域能力层，全部与 NoneBot 解耦】
│   ├── llm/                # LLM 服务与解析
│   ├── conversation/       # 对话大脑与状态管理
│   ├── memory/             # 记忆存储与抽取
│   ├── platform/           # 平台无关消息模型 + MessageDispatcher
│   ├── qq/                 # QQ 消息交互（OneBot 适配）
│   ├── web/                # Web/PWA：鉴权、历史、流式 WebSocket
│   ├── feishu/             # 飞书网关与消息发送
│   ├── image/              # 图片处理与生图
│   ├── character/          # 角色卡与 SillyTavern 交互
│   ├── modes/              # 模式管理
│   ├── queue/              # 推理队列
│   ├── commands/           # 命令解析
│   └── state.py            # 原子写等共享状态工具
│
├── tools/                  # 【离线维护脚本，不被运行时 import】
│   └── memory/             # embedding 回填等维护工具
│
├── tests/                  # 与 core 结构镜像的测试用例
│
├── data/                   # 数据库、模式提示词、贴图、角色卡缓存、生成图
│   ├── modes/              # 三套系统提示词
│   ├── memory_{mode}.db    # 各模式独立记忆库
│   ├── web_history.db      # Web 端会话历史
│   └── generated/          # ComfyUI 产图
├── docs/                   # 设计与使用文档
└── runtime/                # 运行日志
```

## 配置要点

所有配置集中在 `.env`（生产用 `.env.prod`，由 `app.py` 以 `override=True` 加载）。常见分组：

| 分组 | 关键变量 |
|---|---|
| 服务 | `HOST`、`PORT`、`LOG_LEVEL` |
| LLM | `LLM_API_BASE`、`LLM_MODEL`、`LLM_VISION_MODEL`、`LLM_THINK`、`LLM_TEMPERATURE` |
| 记忆 | `MEMORY_DB`（支持 `{mode}` 占位）、embedding 服务地址 |
| 生图 | `COMFYUI_URL`、`COMFYUI_WORKFLOW`、各节点键名与采样参数 |
| Web | `WEB_ENABLED`、`WEB_PATH`、`WEB_TOKEN`、`WEB_STREAM` |
| 飞书 | `FEISHU_ENABLED`、`FEISHU_APP_ID`、`FEISHU_APP_SECRET` |

> `.env` 与 `.env.prod` 含密钥，已被 `.gitignore` 排除，**不要提交**。仅 `.env.example` 入库。

## 测试

```bash
.venv/Scripts/python -m pytest tests -q
```

`tests/` 下含离线单测与 `smoke_offline.py` 冒烟脚本；`test_real_connectivity.py`、`verify_*.py` 需要真实外部服务，未配置时会跳过。

## 本地自由修改的文件（云端保留模板）

`data/` 下的提示词模板既想在云端留一份作参考，又希望本地怎么改都不被推送。这些文件用 git 的 `skip-worktree` 标记：文件**继续被跟踪**（云端保留 HEAD 版本），但本地改动不进`git status`、不会被 `git add -A` 打包、不会被 push。

```bash
python -m tools.local_only_files status    # 查看当前标记状态
python -m tools.local_only_files apply     # 标记清单内全部文件
python -m tools.local_only_files release data/modes/writer.txt   # 取消单个文件
```

当前清单：`data/modes/{clone,tavern,writer}.txt`、`data/modes_state.json`、`data/t2i.json`、`data/tavern_characters.json`。

> ⚠ `skip-worktree` 是**本地索引状态，不随 push/clone 传播**。换机器或重新 clone 后需重跑 `apply`。
> 如需把某个文件的本地改动推上去，先 `release` 再正常提交。

