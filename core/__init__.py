"""Bot 公共基础设施。

本包按领域纵向切分（llm / conversation / memory / qq / image / character /
modes / queue / commands），全部与 NoneBot 解耦，便于独立单元测试。

``build_services`` 是依赖注入的组合根：拿到 ``AppConfig`` 后把所有领域服务装配好，
交给 ``app.py`` 的薄 handler 使用。依赖方向严格为 ``app -> core -> config``。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

from dataclasses import dataclass
from pathlib import Path

from core.character.client import CharacterSelection, SillyTavernClient
from core.conversation.service import ConversationService
from core.conversation.work_selection import WorkSelection
from core.image.comfyui import ComfyUIClient
from core.image.service import ImageService
from core.llm.client import OllamaClient
from core.llm.service import LLMService
from core.memory.embedding import OllamaEmbeddingClient
from core.memory.online import OnlineMemoryExtractor
from core.memory.store import MemoryStore
from core.modes.manager import ModeManager, build_modes
from core.qq.interactions import StickerCatalog
from core.qq.sender import QQSender
from core.queue.queue import InferenceQueue
from core.platform.dispatcher import MessageDispatcher


@dataclass
class Services:
    """装配好的全部运行时服务。"""

    memory_stores: dict[str, MemoryStore]
    embedding_client: OllamaEmbeddingClient
    llm_service: LLMService
    sillytavern_client: SillyTavernClient
    mode_manager: ModeManager
    character_selection: CharacterSelection
    work_selection: WorkSelection
    image_service: ImageService
    conversation_service: ConversationService
    qq_sender: QQSender
    inference_queue: InferenceQueue
    message_dispatcher: MessageDispatcher
    online_extractor: OnlineMemoryExtractor


def _build_memory_stores(
    app_config: AppConfig, bot_root: Path, modes: dict[str, object]
) -> dict[str, MemoryStore]:
    """为每个模式构造独立的记忆库；并把历史单一库迁移到「使用记忆的模式」。

    分库后 ``memories`` / ``messages`` / ``interaction_state`` 全部按模式隔离。
    旧版 ``data/memory.db``（所有模式共用）整体迁移为对应模式的库（默认 clone），
    保证历史记忆不丢失、且 tavern/writer 从此各自干净起步。
    """
    template = Path(str(app_config.memory.database))
    stores: dict[str, MemoryStore] = {}
    for key in modes:
        path = Path(str(template).replace("{mode}", key))
        stores[key] = MemoryStore(path)

    legacy = bot_root / "data" / "memory.db"
    owner = "clone" if "clone" in modes else next(iter(modes), None)
    if owner is not None:
        target = stores[owner].path
        if legacy.exists() and not target.exists():
            for ext in ("", "-wal", "-shm", "-journal"):
                src_file = legacy.parent / (legacy.name + ext)
                dst_file = target.parent / (target.name + ext)
                if src_file.exists():
                    src_file.replace(dst_file)
    return stores


def build_services(app_config: AppConfig, bot_root: Path) -> Services:
    """根据 AppConfig 装配所有服务（组合根）。"""
    modes = build_modes(app_config.modes, bot_root)
    mode_manager = ModeManager(
        modes,
        state_path=app_config.modes.modes_state_path,
    )
    memory_stores = _build_memory_stores(app_config, bot_root, modes)
    embedding_client = OllamaEmbeddingClient(
        api_base=app_config.memory.embedding.api_base,
        model=app_config.memory.embedding.model,
        timeout=app_config.memory.embedding.timeout,
        api_key=app_config.memory.embedding.api_key,
        num_gpu=app_config.memory.embedding.num_gpu,
        keep_alive=app_config.memory.embedding.keep_alive,
    )
    ollama_client = OllamaClient(app_config.llm)
    llm_service = LLMService(ollama_client, app_config.llm)

    sillytavern_client = SillyTavernClient(
        data_root=app_config.character.sillytavern_data_root,
        user=app_config.character.sillytavern_user,
        ollama_api_base=app_config.llm.api_base,
        ollama_api_key=app_config.llm.api_key,
        ollama_keep_alive=app_config.llm.keep_alive,
        ollama_think=app_config.llm.think,
        st_url=app_config.character.sillytavern_url,
        st_source=app_config.character.sillytavern_source,
        st_reverse_proxy=app_config.character.sillytavern_reverse_proxy,
        st_model=app_config.character.sillytavern_model,
        st_api_key=app_config.character.sillytavern_api_key,
        user_name=app_config.character.sillytavern_user_name,
        default_character=app_config.character.sillytavern_default_character,
        request_timeout=120.0,
    )

    character_selection = CharacterSelection(
        state_path=app_config.character.tavern_character_state_path
    )

    work_selection = WorkSelection(
        state_path=app_config.conversation.works_state_path
    )

    comfyui_client = ComfyUIClient(app_config.image.comfyui)
    image_service = ImageService(comfyui_client, app_config.image)

    extraction = app_config.memory.online_extraction
    online_extractor = OnlineMemoryExtractor(
        stores=memory_stores,
        self_id="",
        llm_api_base=app_config.llm.api_base,
        llm_model=app_config.llm.model,
        llm_api_key=app_config.llm.api_key,
        embedding_client=embedding_client,
        enabled=extraction.enabled,
        clone_enabled=extraction.clone_enabled,
        tavern_enabled=extraction.tavern_enabled,
        writer_enabled=extraction.writer_enabled,
        min_messages=extraction.min_messages,
        min_partner_messages=extraction.min_partner_messages,
        min_semantic_chars=extraction.min_semantic_chars,
        min_confidence=extraction.min_confidence,
        semantic_duplicate_threshold=extraction.semantic_duplicate_threshold,
        max_messages=extraction.max_messages,
        max_items=extraction.max_items,
        timeout=extraction.timeout,
        tavern_min_messages=extraction.tavern_min_messages,
        tavern_min_partner_messages=extraction.tavern_min_partner_messages,
        writer_min_messages=extraction.writer_min_messages,
        writer_min_partner_messages=extraction.writer_min_partner_messages,
        llm_keep_alive=app_config.llm.keep_alive,
        log_full_io=app_config.llm.log_full_io,
    )
    logger.info(
        "在线记忆抽取：总开关=%s，clone=%s(%d/%d)，tavern=%s(%d/%d)，"
        "writer=%s(%d/%d)",
        "开" if extraction.enabled else "关",
        "开" if extraction.enabled_for("clone") else "关",
        extraction.min_messages,
        extraction.min_partner_messages,
        "开" if extraction.enabled_for("tavern") else "关",
        extraction.tavern_min_messages,
        extraction.tavern_min_partner_messages,
        "开" if extraction.enabled_for("writer") else "关",
        extraction.writer_min_messages,
        extraction.writer_min_partner_messages,
    )

    conversation_service = ConversationService(
        memory_stores=memory_stores,
        llm_service=llm_service,
        mode_manager=mode_manager,
        conversation_settings=app_config.conversation,
        qq_settings=app_config.qq,
        sillytavern_client=sillytavern_client,
        character_selection=character_selection,
        work_selection=work_selection,
        user_name=app_config.character.sillytavern_user_name,
        embedding_client=embedding_client,
        online_extractor=online_extractor,
        image_service=image_service,
        inference_queue=None,  # 下方回填，保证构造顺序
    )

    qq_sender = QQSender(
        settings=app_config.qq,
        sticker_catalog=StickerCatalog(app_config.qq.sticker_dir),
        image_service=image_service,
        stream_pause=app_config.conversation.stream_pause,
    )

    inference_queue = InferenceQueue(
        max_workers=app_config.conversation.max_inference_workers,
        max_pending=app_config.conversation.max_pending_tasks,
    )
    # 回填队列引用（在线记忆抽取走队列，避免阻塞回复）
    conversation_service.inference_queue = inference_queue
    message_dispatcher = MessageDispatcher(conversation_service, inference_queue)

    return Services(
        memory_stores=memory_stores,
        embedding_client=embedding_client,
        llm_service=llm_service,
        sillytavern_client=sillytavern_client,
        mode_manager=mode_manager,
        character_selection=character_selection,
        work_selection=work_selection,
        image_service=image_service,
        conversation_service=conversation_service,
        qq_sender=qq_sender,
        inference_queue=inference_queue,
        message_dispatcher=message_dispatcher,
        online_extractor=online_extractor,
    )
