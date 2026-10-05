from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

from .embedding import OllamaEmbeddingClient, is_local_ollama, ollama_root
from .extraction import PROMPT, parse_json_array, salvage_json_array
from .settings import OnlineExtractionSettings
from .store import MemoryStore, memory_embedding_text, _scope_subject


_WRITER_USER_WORDS = ("用户", "作者", "写作者", "创作者", "对方")
_WRITER_PREFERENCE_WORDS = (
    "偏好", "喜欢", "希望", "要求", "倾向", "接受", "不喜欢", "避免", "不要",
    "写作", "创作", "作品", "小说", "故事", "题材", "风格", "叙事", "文笔",
    "笔触", "篇幅", "节奏", "视角", "情节", "描写", "表达",
)
_WRITER_FICTION_IDENTITY_WORDS = (
    "扮演", "饰演", "化身", "在故事中", "在小说中", "在作品中", "剧情中的",
    "主角", "女主", "男主", "角色", "人物", "npc", "载体",
)


def _writer_person_is_supported(item: dict, rows: list, peer_id: str) -> bool:
    """writer 的 person 只允许真实用户明确表达的跨作品写作偏好。

    小说人物、用户在作品中扮演的角色、以及仅从模型正文推断出的性格都必须拒绝，
    避免被 ``person -> 关于你`` 的展示语义绑定到真实用户账号。
    """
    summary = str(item.get("summary") or "").casefold()
    if not any(word in summary for word in _WRITER_USER_WORDS):
        return False
    if not any(word in summary for word in _WRITER_PREFERENCE_WORDS):
        return False
    if any(word in summary for word in _WRITER_FICTION_IDENTITY_WORDS):
        return False

    # 不能只凭模型生成的小说正文认定用户偏好；至少一条真实用户消息必须包含
    # 写作/偏好/约束语义。诸如“继续”“下一步”“生成图片”不构成长期写作偏好证据。
    partner_text = "\n".join(
        str(row["text"] or "").casefold()
        for row in rows
        if str(row["sender_id"]) == str(peer_id)
    )
    return any(word in partner_text for word in _WRITER_PREFERENCE_WORDS)


@dataclass
class OnlineMemoryExtractor:
    stores: dict[str, MemoryStore]
    self_id: str
    llm_api_base: str
    llm_model: str
    llm_api_key: str = ""
    embedding_client: OllamaEmbeddingClient | None = None
    enabled: bool = OnlineExtractionSettings.enabled
    clone_enabled: bool = OnlineExtractionSettings.clone_enabled
    tavern_enabled: bool = OnlineExtractionSettings.tavern_enabled
    writer_enabled: bool = OnlineExtractionSettings.writer_enabled
    min_messages: int = OnlineExtractionSettings.min_messages
    min_partner_messages: int = OnlineExtractionSettings.min_partner_messages
    min_semantic_chars: int = OnlineExtractionSettings.min_semantic_chars
    min_confidence: float = OnlineExtractionSettings.min_confidence
    semantic_duplicate_threshold: float = (
        OnlineExtractionSettings.semantic_duplicate_threshold
    )
    max_messages: int = OnlineExtractionSettings.max_messages
    max_items: int = OnlineExtractionSettings.max_items
    timeout: float = OnlineExtractionSettings.timeout
    tavern_min_messages: int = OnlineExtractionSettings.tavern_min_messages
    tavern_min_partner_messages: int = OnlineExtractionSettings.tavern_min_partner_messages
    writer_min_messages: int = OnlineExtractionSettings.writer_min_messages
    writer_min_partner_messages: int = OnlineExtractionSettings.writer_min_partner_messages
    llm_keep_alive: str = "30m"
    log_full_io: bool = False

    def enabled_for(self, mode_key: str) -> bool:
        if not self.enabled:
            return False
        if mode_key == "clone":
            return self.clone_enabled
        if mode_key == "tavern":
            return self.tavern_enabled
        if mode_key == "writer":
            return self.writer_enabled
        return False

    def _thresholds_for(self, mode_key: str) -> tuple[int, int]:
        if mode_key == "tavern":
            return self.tavern_min_messages, self.tavern_min_partner_messages
        if mode_key == "writer":
            return self.writer_min_messages, self.writer_min_partner_messages
        return self.min_messages, self.min_partner_messages

    async def maybe_extract(
        self,
        peer_id: str,
        mode_key: str = "clone",
        namespace: str | None = None,
        self_id: str | None = None,
    ) -> int:
        if not self.enabled_for(mode_key):
            return 0
        current_self_id = str(self_id if self_id is not None else self.self_id)
        store = self.stores[mode_key]
        min_messages, min_partner_messages = self._thresholds_for(mode_key)
        rows = store.get_online_extraction_batch(
            current_self_id,
            peer_id,
            self.max_messages,
            mode_key=mode_key,
            namespace=namespace,
        )
        if len(rows) < min_messages:
            logger.debug(
                "在线记忆等待更多对话：mode=%s namespace=%s messages=%d/%d",
                mode_key, namespace or "", len(rows), min_messages,
            )
            return 0
        semantic_chars = sum(len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", row["text"]))
                             for row in rows)
        speakers = {str(row["sender_id"]) for row in rows}
        partner_messages = sum(str(row["sender_id"]) == str(peer_id) for row in rows)
        last_rowid = int(rows[-1]["id"])
        if (semantic_chars < self.min_semantic_chars or len(speakers) < 2
                or partner_messages < min_partner_messages):
            logger.debug(
                "在线记忆批次信息不足，已推进游标：mode=%s namespace=%s "
                "semantic_chars=%d/%d speakers=%d partner_messages=%d/%d",
                mode_key, namespace or "", semantic_chars, self.min_semantic_chars,
                len(speakers), partner_messages, min_partner_messages,
            )
            store.finish_online_extraction(
                current_self_id,
                peer_id,
                last_rowid,
                mode_key=mode_key,
                namespace=namespace,
            )
            return 0

        if mode_key == "writer":
            dialogue = "\n".join(
                f"[{row['sent_at']}] "
                f"{'用户写作指令（真实用户，不等于小说角色）' if str(row['sender_id']) == str(peer_id) else '模型生成小说正文（虚构内容）'}: "
                f"{row['text']}"
                for row in rows
            )
        else:
            dialogue = "\n".join(
                f"[{row['sent_at']}] QQ{row['sender_id']}: {row['text']}" for row in rows
            )
        known_memories = "（无相关已有记忆）"
        known_query_embedding = None
        if self.embedding_client:
            try:
                known_vectors = await self.embedding_client.embed_async([
                    "为长期记忆去重检索相关事实：\n" + dialogue[-4000:]
                ])
                known_query_embedding = known_vectors[0]
            except Exception:
                known_query_embedding = None
        recalled = store.search_context(
            dialogue[-2500:], current_self_id, peer_id,
            limit=5, stable_limit=1, max_chars=2200,
            namespace=namespace,
            query_embedding=known_query_embedding,
            embedding_model=self.embedding_client.model if self.embedding_client else None,
            vector_min_similarity=.45,
        )
        if recalled:
            known_memories = recalled

        mode_context = {
            "clone": (
                "clone 人类模仿；只记录真实用户与双方真实关系。机器人生成内容不能单独作为事实证据。"
            ),
            "tavern": (
                "tavern 当前角色；self/relationship/episode 可记录角色设定和角色内剧情；"
                "person 只记录用户明确表达的真实偏好或边界，不能从虚构剧情推断。"
            ),
            "writer": (
                "writer 当前作品；relationship/episode 记录作品人物、关系、事件和伏笔；"
                "作品中的主角、女主、男主、NPC及用户扮演的角色都属于虚构内容，绝不输出为 person。"
                "person 只能记录真实用户明确说出的、可跨作品复用的写作偏好或创作边界，"
                "summary 必须明确写成‘用户偏好/用户要求……’，不得包含‘用户扮演某角色’。"
                "无法判断是现实用户还是小说人物时，禁止输出 person；优先使用 relationship 或 episode。"
            ),
        }.get(mode_key, mode_key)
        prompt = PROMPT.format(
            mode_context=mode_context,
            self_id=current_self_id,
            partner_id=peer_id,
            max_items=self.max_items,
            known_memories=known_memories,
            dialogue=dialogue,
        )
        local = is_local_ollama(self.llm_api_base)
        root = ollama_root(self.llm_api_base) if local else self.llm_api_base.rstrip("/")
        url = f"{root}/api/chat" if local else f"{root}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.llm_api_key:
            headers["Authorization"] = f"Bearer {self.llm_api_key}"
        payload = {
            "model": self.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        if local:
            payload.update({"think": False, "keep_alive": self.llm_keep_alive,
                            "options": {"temperature": .1, "num_predict": 1800}})
        else:
            payload.update({"temperature": .1, "max_tokens": 1800})

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
            if local:
                content = data.get("message", {}).get("content", "")
            else:
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            try:
                items = parse_json_array(content)
            except Exception:
                items = salvage_json_array(content)
            # /清记忆 可能在 LLM 推理期间推进抽取游标。此时当前结果来自清理前的
            # 旧批次，必须丢弃，防止刚清掉的记忆被异步任务重新写回。
            if not store.is_online_extraction_batch_current(
                current_self_id,
                peer_id,
                last_rowid,
                mode_key=mode_key,
                namespace=namespace,
            ):
                logger.info(
                    "在线记忆抽取结果已因状态重置失效：mode=%s namespace=%s last_rowid=%d",
                    mode_key, namespace or "", last_rowid,
                )
                return 0

            normalized = []
            for item in items[:self.max_items]:
                if not isinstance(item, dict) or item.get("kind") not in {
                    "self", "person", "relationship", "episode"
                }:
                    continue
                if (
                    mode_key == "writer"
                    and item.get("kind") == "person"
                    and not _writer_person_is_supported(item, rows, peer_id)
                ):
                    logger.info(
                        "writer 已拒绝疑似小说角色污染用户记忆：summary=%r",
                        str(item.get("summary") or "")[:300],
                    )
                    continue
                item["subject_id"] = _scope_subject(
                    namespace, item["kind"], current_self_id, peer_id)
                item["owner_id"] = current_self_id
                item["confidence"] = min(.95, max(.05, float(item.get("confidence", .7))))
                if item["confidence"] < self.min_confidence:
                    continue
                if item.get("valid_from") is None:
                    item["valid_from"] = int(rows[-1]["sent_at"])
                normalized.append(item)
            saved = merged = 0
            if normalized and self.embedding_client:
                try:
                    texts = [memory_embedding_text({
                        **item,
                        "keywords": " ".join(str(x) for x in (item.get("keywords") or [])),
                    }) for item in normalized]
                    embeddings = await self.embedding_client.embed_async(texts)
                    saved, merged, _ = store.save_memories_semantic(
                        normalized,
                        [str(row["msg_id"]) for row in rows],
                        self.embedding_client.model,
                        embeddings,
                        self.semantic_duplicate_threshold,
                    )
                except Exception:
                    # 向量服务异常时仍保存正文；向量可以由回填脚本稍后补齐
                    saved = store.save_memories(
                        normalized, [str(row["msg_id"]) for row in rows])
            elif normalized:
                saved = store.save_memories(
                    normalized, [str(row["msg_id"]) for row in rows])
            store.finish_online_extraction(
                current_self_id,
                peer_id,
                last_rowid,
                mode_key=mode_key,
                namespace=namespace,
            )
            logger.info(
                "在线记忆抽取完成：mode=%s namespace=%s messages=%d candidates=%d "
                "saved=%d merged=%d",
                mode_key, namespace or "", len(rows), len(normalized), saved, merged,
            )
            return saved + merged
        except Exception as exc:
            store.finish_online_extraction(
                current_self_id,
                peer_id,
                None,
                f"{type(exc).__name__}: {exc}",
                mode_key=mode_key,
                namespace=namespace,
            )
            raise
