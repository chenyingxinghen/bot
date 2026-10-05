from __future__ import annotations

import json
import hashlib
import math
import re
import sqlite3
import time
from array import array
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Mapping


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY, msg_id TEXT UNIQUE, self_id TEXT NOT NULL DEFAULT '',
  sender_id TEXT NOT NULL,
  peer_id TEXT NOT NULL, mode_key TEXT NOT NULL DEFAULT '',
  namespace TEXT NOT NULL DEFAULT '', sent_at INTEGER NOT NULL, text TEXT NOT NULL,
  send_type INTEGER, source_file TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_people_time
  ON messages(sender_id, peer_id, sent_at);
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
  text, content='messages', content_rowid='id', tokenize='trigram'
);
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
  INSERT INTO messages_fts(rowid,text) VALUES(new.id,new.text);
END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
  INSERT INTO messages_fts(messages_fts,rowid,text) VALUES('delete',old.id,old.text);
END;
CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
  INSERT INTO messages_fts(messages_fts,rowid,text) VALUES('delete',old.id,old.text);
  INSERT INTO messages_fts(rowid,text) VALUES(new.id,new.text);
END;
CREATE TABLE IF NOT EXISTS memories (
  id INTEGER PRIMARY KEY, owner_id TEXT NOT NULL DEFAULT '',
  kind TEXT NOT NULL, subject_id TEXT NOT NULL,
  object_id TEXT, summary TEXT NOT NULL, keywords TEXT NOT NULL DEFAULT '',
  confidence REAL NOT NULL DEFAULT .7, valid_from INTEGER, valid_to INTEGER,
  status TEXT NOT NULL DEFAULT 'active', evidence_json TEXT NOT NULL DEFAULT '[]',
  created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL,
  fingerprint TEXT UNIQUE
);
CREATE INDEX IF NOT EXISTS idx_memories_subject ON memories(subject_id,status,kind);
CREATE TABLE IF NOT EXISTS memory_embeddings (
  memory_id INTEGER NOT NULL, model TEXT NOT NULL, dimensions INTEGER NOT NULL,
  embedding BLOB NOT NULL, content_hash TEXT NOT NULL, embedded_at INTEGER NOT NULL,
  PRIMARY KEY(memory_id,model), FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_memory_embeddings_model ON memory_embeddings(model,memory_id);
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
  summary, keywords, content='memories', content_rowid='id', tokenize='trigram'
);
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
  INSERT INTO memories_fts(rowid,summary,keywords) VALUES(new.id,new.summary,new.keywords);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
  INSERT INTO memories_fts(memories_fts,rowid,summary,keywords)
    VALUES('delete',old.id,old.summary,old.keywords);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
  INSERT INTO memories_fts(memories_fts,rowid,summary,keywords)
    VALUES('delete',old.id,old.summary,old.keywords);
  INSERT INTO memories_fts(rowid,summary,keywords) VALUES(new.id,new.summary,new.keywords);
END;
CREATE TABLE IF NOT EXISTS extraction_jobs (
  chunk_key TEXT PRIMARY KEY, start_time INTEGER NOT NULL, end_time INTEGER NOT NULL,
  message_ids TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
  attempts INTEGER NOT NULL DEFAULT 0, error TEXT, updated_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS prepared_conversations (
  chunk_key TEXT PRIMARY KEY, self_id TEXT NOT NULL DEFAULT '', peer_id TEXT NOT NULL,
  start_time INTEGER NOT NULL, end_time INTEGER NOT NULL,
  message_ids TEXT NOT NULL, message_count INTEGER NOT NULL,
  informative_count INTEGER NOT NULL, speaker_count INTEGER NOT NULL,
  quality_score REAL NOT NULL, keep INTEGER NOT NULL,
  reason TEXT NOT NULL, prepared_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_prepared_peer_keep
  ON prepared_conversations(peer_id,keep,start_time);
CREATE TABLE IF NOT EXISTS preprocessing_runs (
  id INTEGER PRIMARY KEY, self_id TEXT NOT NULL DEFAULT '',
  peer_id TEXT NOT NULL, run_at INTEGER NOT NULL,
  source_messages INTEGER NOT NULL, removed_placeholders INTEGER NOT NULL,
  prepared_chunks INTEGER NOT NULL, kept_chunks INTEGER NOT NULL,
  rules_version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS interaction_state (
  self_id TEXT NOT NULL, peer_id TEXT NOT NULL,
  mode_key TEXT NOT NULL DEFAULT '', namespace TEXT NOT NULL DEFAULT '',
  last_user_at INTEGER, last_bot_at INTEGER, updated_at INTEGER NOT NULL,
  PRIMARY KEY(self_id,peer_id,mode_key,namespace)
);
CREATE TABLE IF NOT EXISTS online_extraction_state (
  self_id TEXT NOT NULL, peer_id TEXT NOT NULL,
  mode_key TEXT NOT NULL DEFAULT '', namespace TEXT NOT NULL DEFAULT '',
  last_message_rowid INTEGER NOT NULL DEFAULT 0,
  last_extracted_at INTEGER, status TEXT NOT NULL DEFAULT 'idle',
  error TEXT, updated_at INTEGER NOT NULL,
  PRIMARY KEY(self_id,peer_id,mode_key,namespace)
);
CREATE TABLE IF NOT EXISTS context_reset_state (
  self_id TEXT NOT NULL, peer_id TEXT NOT NULL,
  mode_key TEXT NOT NULL DEFAULT '', namespace TEXT NOT NULL DEFAULT '',
  last_message_rowid INTEGER NOT NULL DEFAULT 0, updated_at INTEGER NOT NULL,
  PRIMARY KEY(self_id,peer_id,mode_key,namespace)
);
CREATE TABLE IF NOT EXISTS conflict_jobs (
  pair_key TEXT PRIMARY KEY, left_memory_id INTEGER NOT NULL,
  right_memory_id INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
  attempts INTEGER NOT NULL DEFAULT 0, relation TEXT, confidence REAL,
  result_json TEXT, error TEXT, updated_at INTEGER NOT NULL,
  UNIQUE(left_memory_id,right_memory_id)
);
CREATE TABLE IF NOT EXISTS memory_conflicts (
  id INTEGER PRIMARY KEY, left_memory_id INTEGER NOT NULL,
  right_memory_id INTEGER NOT NULL, relation TEXT NOT NULL,
  confidence REAL NOT NULL, reason TEXT NOT NULL DEFAULT '',
  winner_id INTEGER, resolved_at INTEGER NOT NULL,
  UNIQUE(left_memory_id,right_memory_id)
);
"""


MEMORY_EXTRA_COLUMNS = {
    "owner_id": "TEXT NOT NULL DEFAULT ''",
    "supersedes_id": "INTEGER",
    "conflict_group": "TEXT",
}


def _migrate_account_scoped_tables(db: sqlite3.Connection) -> None:
    """为旧数据库补上机器人账号维度，避免多账号共用同一联系人状态。"""
    message_columns = {row["name"] for row in db.execute("PRAGMA table_info(messages)")}
    if "self_id" not in message_columns:
        db.execute("ALTER TABLE messages ADD COLUMN self_id TEXT NOT NULL DEFAULT ''")
    if "mode_key" not in message_columns:
        db.execute("ALTER TABLE messages ADD COLUMN mode_key TEXT NOT NULL DEFAULT ''")
    if "namespace" not in message_columns:
        db.execute("ALTER TABLE messages ADD COLUMN namespace TEXT NOT NULL DEFAULT ''")
    db.execute("""CREATE INDEX IF NOT EXISTS idx_messages_account_peer_time
      ON messages(self_id,peer_id,sent_at)""")
    db.execute("""CREATE INDEX IF NOT EXISTS idx_messages_scope_time
      ON messages(self_id,peer_id,mode_key,namespace,sent_at)""")

    for table in ("prepared_conversations", "preprocessing_runs"):
        columns = {row["name"] for row in db.execute(f"PRAGMA table_info({table})")}
        if "self_id" not in columns:
            db.execute(f"ALTER TABLE {table} ADD COLUMN self_id TEXT NOT NULL DEFAULT ''")
    db.execute("""CREATE INDEX IF NOT EXISTS idx_prepared_account_peer_keep
      ON prepared_conversations(self_id,peer_id,keep,start_time)""")

    interaction_info = list(db.execute("PRAGMA table_info(interaction_state)"))
    interaction_columns = {row["name"] for row in interaction_info}
    interaction_pk = [
        row["name"] for row in sorted(interaction_info, key=lambda item: item["pk"])
        if row["pk"]
    ]
    expected_interaction_pk = ["self_id", "peer_id", "mode_key", "namespace"]
    if interaction_pk != expected_interaction_pk:
        legacy = "interaction_state_legacy_scope"
        db.execute(f"ALTER TABLE interaction_state RENAME TO {legacy}")
        db.execute("""CREATE TABLE interaction_state (
          self_id TEXT NOT NULL, peer_id TEXT NOT NULL,
          mode_key TEXT NOT NULL DEFAULT '', namespace TEXT NOT NULL DEFAULT '',
          last_user_at INTEGER, last_bot_at INTEGER, updated_at INTEGER NOT NULL,
          PRIMARY KEY(self_id,peer_id,mode_key,namespace))""")
        self_expr = "self_id" if "self_id" in interaction_columns else "''"
        mode_expr = "mode_key" if "mode_key" in interaction_columns else "''"
        namespace_expr = "namespace" if "namespace" in interaction_columns else "''"
        db.execute(f"""INSERT OR REPLACE INTO interaction_state
          (self_id,peer_id,mode_key,namespace,last_user_at,last_bot_at,updated_at)
          SELECT {self_expr},peer_id,{mode_expr},{namespace_expr},last_user_at,last_bot_at,
                 updated_at FROM {legacy}""")
        db.execute(f"DROP TABLE {legacy}")

    extraction_info = list(db.execute("PRAGMA table_info(online_extraction_state)"))
    extraction_columns = {row["name"] for row in extraction_info}
    extraction_pk = [
        row["name"] for row in sorted(extraction_info, key=lambda item: item["pk"])
        if row["pk"]
    ]
    expected_pk = ["self_id", "peer_id", "mode_key", "namespace"]
    if extraction_pk != expected_pk:
        legacy = "online_extraction_state_legacy_scope"
        db.execute(f"ALTER TABLE online_extraction_state RENAME TO {legacy}")
        db.execute("""CREATE TABLE online_extraction_state (
          self_id TEXT NOT NULL, peer_id TEXT NOT NULL,
          mode_key TEXT NOT NULL DEFAULT '', namespace TEXT NOT NULL DEFAULT '',
          last_message_rowid INTEGER NOT NULL DEFAULT 0,
          last_extracted_at INTEGER, status TEXT NOT NULL DEFAULT 'idle',
          error TEXT, updated_at INTEGER NOT NULL,
          PRIMARY KEY(self_id,peer_id,mode_key,namespace))""")
        self_expr = "self_id" if "self_id" in extraction_columns else "''"
        mode_expr = "mode_key" if "mode_key" in extraction_columns else "''"
        namespace_expr = "namespace" if "namespace" in extraction_columns else "''"
        db.execute(f"""INSERT OR REPLACE INTO online_extraction_state
          (self_id,peer_id,mode_key,namespace,last_message_rowid,last_extracted_at,
           status,error,updated_at)
          SELECT {self_expr},peer_id,{mode_expr},{namespace_expr},last_message_rowid,
                 last_extracted_at,status,error,updated_at FROM {legacy}""")
        db.execute(f"DROP TABLE {legacy}")

    memory_columns = {row["name"] for row in db.execute("PRAGMA table_info(memories)")}
    owner_was_missing = "owner_id" not in memory_columns
    if owner_was_missing:
        db.execute("ALTER TABLE memories ADD COLUMN owner_id TEXT NOT NULL DEFAULT ''")
        # self / relationship / episode 旧记录可以可靠推导所属账号
        db.execute("""UPDATE memories SET owner_id=subject_id
          WHERE owner_id='' AND kind='self'""")
        relationship_rows = db.execute("""SELECT id,subject_id FROM memories
          WHERE owner_id='' AND kind IN ('relationship','episode')
          AND subject_id LIKE 'relationship:%'""").fetchall()
        for row in relationship_rows:
            parts = str(row["subject_id"]).split(":", 2)
            if len(parts) == 3 and parts[1]:
                db.execute("UPDATE memories SET owner_id=? WHERE id=?",
                           (parts[1], int(row["id"])))
        owners = {str(row["owner_id"]) for row in db.execute(
            "SELECT DISTINCT owner_id FROM memories WHERE owner_id<>''")}
        if len(owners) == 1:
            # 旧版 person 记忆没有记录 owner；单账号数据库可无歧义迁移
            db.execute("UPDATE memories SET owner_id=? WHERE owner_id=''", (owners.pop(),))
        for row in db.execute(
            "SELECT id,owner_id,kind,subject_id,summary FROM memories"
        ).fetchall():
            fingerprint = (f"{row['owner_id']}|{row['kind']}|"
                           f"{row['subject_id']}|{row['summary']}")
            db.execute("UPDATE memories SET fingerprint=? WHERE id=?",
                       (fingerprint, int(row["id"])))
    db.execute("""CREATE INDEX IF NOT EXISTS idx_memories_owner_subject
      ON memories(owner_id,subject_id,status,kind)""")


def memory_embedding_text(row: sqlite3.Row | dict) -> str:
    kind = str(row["kind"])
    summary = str(row["summary"])
    keywords = str(row["keywords"] or "")
    return f"记忆类型：{kind}\n内容：{summary}\n关键词：{keywords}".strip()


def embedding_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pack_embedding(values: list[float]) -> tuple[bytes, int]:
    vector = array("f", (float(x) for x in values))
    norm = math.sqrt(sum(float(x) * float(x) for x in vector))
    if not vector or norm <= 0:
        raise ValueError("embedding 不能为空或零向量")
    normalized = array("f", (float(x) / norm for x in vector))
    return normalized.tobytes(), len(normalized)


def unpack_embedding(blob: bytes, dimensions: int) -> array:
    vector = array("f")
    vector.frombytes(blob)
    if len(vector) != dimensions:
        raise ValueError(f"embedding 维度不匹配：期望 {dimensions}，实际 {len(vector)}")
    return vector


def _relative_time(timestamp: int, now: int) -> str:
    delta = now - timestamp
    future = delta < 0
    seconds = abs(delta)
    if seconds < 3600:
        amount, unit = max(1, int(seconds // 60)), "分钟"
    elif seconds < 86400:
        amount, unit = int(seconds // 3600), "小时"
    elif seconds < 86400 * 30:
        amount, unit = int(seconds // 86400), "天"
    elif seconds < 86400 * 365:
        amount, unit = int(seconds // (86400 * 30)), "个月"
    else:
        amount, unit = int(seconds // (86400 * 365)), "年"
    return f"约{amount}{unit}{'后' if future else '前'}"


def format_memory_time(kind: str, valid_from: int | None, valid_to: int | None,
                       now: int | None = None) -> str:
    now = int(time.time()) if now is None else now
    if valid_from is None and valid_to is None:
        return "时间未知"
    start = datetime.fromtimestamp(valid_from).strftime("%Y-%m-%d") if valid_from else None
    end = datetime.fromtimestamp(valid_to).strftime("%Y-%m-%d") if valid_to else None
    if start and end:
        return f"有效起 {start} 至 {end}"
    if end:
        return f"截至 {end}"
    relative = _relative_time(int(valid_from), now)
    if kind == "episode":
        return f"发生于 {start}，{relative}"
    return f"自 {start} 起，{relative}记录"


def _scope_subject(namespace: str | None, kind: str, self_id: str, partner_id: str) -> str:
    """记忆 subject 套上命名空间前缀，形成「模式内第三维隔离」（角色 / 作品）。

    - person：关于用户本人，跨命名空间全局共享，不加前缀（subject = partner_id）。
    - self：角色 / 作者自身设定 → 加前缀，尾部带 partner_id 便于安全清理。
    - relationship / episode：互动 / 剧情 → 加前缀。
    namespace 为 None（clone 模式或尚未选定实体）时退化为旧格式，向后兼容。
    """
    if kind == "person":
        return partner_id
    if kind == "self":
        return f"{namespace}:self:{partner_id}" if namespace else self_id
    # relationship / episode
    return (f"{namespace}:rel:{self_id}:{partner_id}"
            if namespace else f"relationship:{self_id}:{partner_id}")


def _terms(text: str) -> list[str]:
    """提取查询词；保留中文短词，并补充少量确定性的意图同义词。"""
    normalized = text.lower()
    chunks = re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9_]{2,}", normalized)
    terms: list[str] = []
    for chunk in chunks:
        terms.append(chunk)
        # 自然聊天通常没有空格。补齐 2~4 字窗口后，“你还喜欢猫吗”也能命中“喜欢猫咪”
        if re.fullmatch(r"[\u4e00-\u9fff]+", chunk) and len(chunk) > 4:
            for size in (2, 3, 4):
                terms.extend(chunk[i:i + size] for i in range(len(chunk) - size + 1))
    if re.search(r"(?:叫|称呼).{0,4}(?:什么|啥)|(?:怎么|如何).{0,3}称呼", normalized):
        terms.extend(("称呼", "叫法", "名字"))
    return list(dict.fromkeys(terms))[:36]


def _ngrams(text: str, size: int = 2) -> set[str]:
    compact = "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9_]", text.lower()))
    if len(compact) < size:
        return {compact} if compact else set()
    return {compact[i:i + size] for i in range(len(compact) - size + 1)}


def _relevance(row: sqlite3.Row, terms: list[str], query_grams: set[str]) -> float:
    summary = row["summary"].lower()
    keywords = row["keywords"].lower()
    keyword_tokens = set(keywords.split())
    score = 0.0
    matched_terms = 0
    for term in terms:
        matched = False
        if term in keyword_tokens:
            score += 12.0
            matched = True
        elif term in keywords:
            score += 7.0
            matched = True
        if term in summary:
            score += 4.0
            matched = True
        matched_terms += int(matched)
    if terms:
        score += 6.0 * matched_terms / len(terms)
    overlap = query_grams & _ngrams(f"{keywords} {summary}")
    score += min(len(overlap), 12) * 0.8
    # 相关度主导排序，置信度只负责打破接近的候选，避免高置信无关记忆挤占位置
    score += float(row["confidence"]) * 1.5
    return score


def _runtime_relevance(text: str, terms: list[str], query_grams: set[str]) -> float:
    """衡量原始对话与当前问题的词面相关度，不依赖 embedding 服务。"""
    normalized = (text or "").lower()
    matched_terms = sum(term in normalized for term in terms)
    overlap = query_grams & _ngrams(normalized)
    if matched_terms == 0 and not overlap:
        return 0.0
    coverage = matched_terms / max(1, len(terms))
    return matched_terms * 4.0 + min(len(overlap), 16) * 0.8 + coverage * 3.0


def _has_temporal_intent(query: str) -> bool:
    return bool(re.search(
        r"今天|昨天|前天|明天|最近|现在|目前|当时|以前|后来|上次|多久|哪天|"
        r"今年|去年|本周|上周|这几天|过去|曾经|\d{4}年|\d{1,2}月",
        query,
    ))


def _dot(left: array, right: array) -> float:
    if len(left) != len(right):
        return -1.0
    return sum(a * b for a, b in zip(left, right))


@dataclass
class MemoryStore:
    path: Path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.executescript(SCHEMA)
            _migrate_account_scoped_tables(db)
            columns = {row["name"] for row in db.execute("PRAGMA table_info(memories)")}
            for name, declaration in MEMORY_EXTRA_COLUMNS.items():
                if name not in columns:
                    db.execute(f"ALTER TABLE memories ADD COLUMN {name} {declaration}")
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def import_history(self, source: Path, peer_id: str, self_id: str = "") -> int:
        records = json.loads(source.read_text(encoding="utf-8-sig"))
        rows = []
        for item in records:
            text = "".join(str(x) for x in (item.get("elements") or [])).strip()
            sender = str(item.get("senderQQNum") or "0")
            if not text or sender == "0":
                continue
            rows.append((str(item.get("msgId")), self_id, sender, peer_id,
                         int(item.get("msgTime") or 0), text,
                         int(item.get("sendType") or 0), str(source)))
        with self.connect() as db:
            before = db.execute("SELECT count(*) FROM messages").fetchone()[0]
            db.executemany("""INSERT OR IGNORE INTO messages
              (msg_id,self_id,sender_id,peer_id,sent_at,text,send_type,source_file)
              VALUES(?,?,?,?,?,?,?,?)""", rows)
            after = db.execute("SELECT count(*) FROM messages").fetchone()[0]
            return after - before

    def search_context(self, query: str, self_id: str, partner_id: str,
                       limit: int = 4, stable_limit: int = 0,
                       max_chars: int = 900,
                       query_embedding: list[float] | None = None,
                       embedding_model: str | None = None,
                       vector_min_similarity: float = 0.55,
                       namespace: str | None = None,
                       allowed_kinds: set[str] | None = None) -> str:
        """召回与当前话题有关的有效记忆。

        ``allowed_kinds`` 供不同业务模式限制可见记忆类型。subject 与 kind 成对过滤，
        避免历史脏数据仅因 subject 碰巧相同而跨语义范围进入 prompt。
        """
        query = (query or "").strip()
        terms = _terms(query)
        query_grams = _ngrams(query)
        scoped_subjects = {
            "person": partner_id,
            "self": _scope_subject(namespace, "self", self_id, partner_id),
            "relationship": _scope_subject(namespace, "relationship", self_id, partner_id),
            "episode": _scope_subject(namespace, "episode", self_id, partner_id),
        }
        kinds = set(allowed_kinds or scoped_subjects)
        kinds &= set(scoped_subjects)
        if not kinds:
            return ""
        scope_pairs = [(kind, scoped_subjects[kind]) for kind in sorted(kinds)]
        scope_sql = " OR ".join("(kind=? AND subject_id=?)" for _ in scope_pairs)
        scope_values = [value for pair in scope_pairs for value in pair]
        with self.connect() as db:
            rows = db.execute(f"""SELECT id,owner_id,kind,subject_id,summary,keywords,confidence,
              valid_from,valid_to,status,updated_at,conflict_group FROM memories
              WHERE status IN ('active','disputed') AND owner_id IN (?,'')
              AND ({scope_sql})
              AND (valid_to IS NULL OR valid_to>=strftime('%s','now'))""",
              (self_id, *scope_values)).fetchall()
            embedding_rows = []
            if query_embedding and embedding_model:
                embedding_rows = db.execute(f"""SELECT e.memory_id,e.dimensions,e.embedding
                  FROM memory_embeddings e JOIN memories m ON m.id=e.memory_id
                  WHERE e.model=? AND m.status IN ('active','disputed')
                  AND m.owner_id IN (?,'') AND ({scope_sql})
                  AND (m.valid_to IS NULL OR m.valid_to>=strftime('%s','now'))""",
                  (embedding_model, self_id, *scope_values)).fetchall()
        if not rows:
            return ""

        by_id = {int(row["id"]): row for row in rows}
        lexical = [(_relevance(row, terms, query_grams), row) for row in rows]
        lexical = [item for item in lexical
                   if item[0] > float(item[1]["confidence"]) * 1.5]
        lexical.sort(key=lambda item: item[0], reverse=True)
        lexical_rank = {int(row["id"]): rank
                        for rank, (_, row) in enumerate(lexical[:60], 1)}

        vector_scores: list[tuple[float, int]] = []
        if embedding_rows and query_embedding:
            packed_query, dimensions = pack_embedding(query_embedding)
            query_vector = unpack_embedding(packed_query, dimensions)
            for embedded in embedding_rows:
                try:
                    vector = unpack_embedding(embedded["embedding"], embedded["dimensions"])
                except ValueError:
                    continue
                similarity = _dot(query_vector, vector)
                if similarity >= vector_min_similarity:
                    vector_scores.append((similarity, int(embedded["memory_id"])))
            vector_scores.sort(reverse=True)
        vector_rank = {memory_id: rank
                       for rank, (_, memory_id) in enumerate(vector_scores[:60], 1)}

        # Reciprocal Rank Fusion：词面和向量分数量纲不同，用排名融合更稳定。
        candidate_ids = set(lexical_rank) | set(vector_rank)
        temporal_intent = _has_temporal_intent(query)
        if temporal_intent:
            # “最近发生了什么”通常没有事件正文关键词，主动加入最近 episode 再做排序。
            recent_episode_ids = [
                int(row["id"]) for row in sorted(
                    (row for row in rows if row["kind"] == "episode" and row["valid_from"]),
                    key=lambda row: row["valid_from"], reverse=True,
                )[:max(limit * 2, 4)]
            ]
            candidate_ids.update(recent_episode_ids)
        recent = sorted(
            (by_id[mid] for mid in candidate_ids if by_id[mid]["valid_from"]),
            key=lambda row: row["valid_from"], reverse=True,
        )
        temporal_rank = {int(row["id"]): rank for rank, row in enumerate(recent, 1)}

        def fused(memory_id: int) -> float:
            score = 0.0
            if memory_id in vector_rank:
                score += 1.0 / (60 + vector_rank[memory_id])
            if memory_id in lexical_rank:
                score += .85 / (60 + lexical_rank[memory_id])
            row = by_id[memory_id]
            if memory_id in temporal_rank and (temporal_intent or row["kind"] == "episode"):
                weight = .70 if temporal_intent else .10
                score += weight / (60 + temporal_rank[memory_id])
            score += .05 * float(row["confidence"]) / 61
            if row["status"] == "disputed":
                score *= .82
            return score

        ranked = sorted(
            (by_id[mid] for mid in candidate_ids),
            key=lambda row: (fused(int(row["id"])), row["confidence"], row["updated_at"]),
            reverse=True,
        )
        selected_related = []
        normalized_used: set[str] = set()
        for row in ranked:
            normalized = re.sub(r"\s+", "", str(row["summary"])).casefold()
            if normalized in normalized_used:
                continue
            normalized_used.add(normalized)
            selected_related.append(row)
            if len(selected_related) >= limit:
                break
        used = {row["summary"] for row in selected_related}
        stable = sorted(
            (row for row in rows
             if row["summary"] not in used and row["kind"] != "episode"
             and float(row["confidence"]) >= .85 and row["status"] == "active"),
            key=lambda row: (row["confidence"], row["updated_at"]), reverse=True,
        )[:stable_limit]

        sections = []
        if selected_related:
            sections.append(("【当前话题相关】", selected_related))
        if stable:
            sections.append(("【稳定背景】", stable))
        lines: list[str] = []
        length = 0
        for heading, section_rows in sections:
            if lines:
                lines.append("")
                length += 1
            lines.append(heading)
            length += len(heading)
            for row in section_rows:
                time_label = format_memory_time(row["kind"], row["valid_from"], row["valid_to"])
                conflict_label = "，信息存在未决冲突" if row["status"] == "disputed" else ""
                line = (f"- [{row['kind']}，{time_label}{conflict_label}] {row['summary']}"
                        f"（可信度 {row['confidence']:.2f}）")
                if length + len(line) + 1 > max_chars:
                    return "\n".join(lines)
                lines.append(line)
                length += len(line) + 1
        return "\n".join(lines)

    def baseline_memories(
        self,
        self_id: str,
        partner_id: str,
        *,
        namespace: str | None = None,
        min_confidence: float = .85,
        kind_limits: Mapping[str, int] | None = None,
    ) -> list[dict]:
        """选择每轮可常驻的稳定基础记忆。

        常驻层只允许 active 的 person/self/relationship；episode、未决冲突和过期事实
        继续交给按话题召回，避免旧事件或冲突状态持续污染模型。subject 与 kind 仍成对
        校验，并按类型设置配额，防止某一类记忆挤占整个胶囊。
        """
        limits = dict(kind_limits or {"person": 4, "self": 3, "relationship": 3})
        limits = {
            kind: max(0, int(limits.get(kind, 0)))
            for kind in ("person", "self", "relationship")
        }
        kinds = {kind for kind, limit in limits.items() if limit > 0}
        if not kinds:
            return []
        scoped_subjects = {
            "person": partner_id,
            "self": _scope_subject(namespace, "self", self_id, partner_id),
            "relationship": _scope_subject(
                namespace, "relationship", self_id, partner_id
            ),
        }
        scope_pairs = [
            (kind, scoped_subjects[kind]) for kind in sorted(kinds)
        ]
        scope_sql = " OR ".join("(kind=? AND subject_id=?)" for _ in scope_pairs)
        scope_values = [value for pair in scope_pairs for value in pair]
        with self.connect() as db:
            rows = db.execute(
                f"""SELECT id,kind,subject_id,summary,keywords,confidence,
                    valid_from,valid_to,updated_at
                    FROM memories
                    WHERE status='active' AND owner_id IN (?,'')
                    AND confidence>=? AND ({scope_sql})
                    AND (valid_to IS NULL OR valid_to>=strftime('%s','now'))
                    ORDER BY confidence DESC, updated_at DESC, id DESC""",
                (self_id, float(min_confidence), *scope_values),
            ).fetchall()

        selected: list[dict] = []
        counts = {kind: 0 for kind in limits}
        normalized_used: set[str] = set()
        for row in rows:
            kind = str(row["kind"])
            if counts.get(kind, 0) >= limits.get(kind, 0):
                continue
            normalized = re.sub(r"\s+", "", str(row["summary"])).casefold()
            if not normalized or normalized in normalized_used:
                continue
            normalized_used.add(normalized)
            counts[kind] += 1
            selected.append(dict(row))
        return selected

    def memories_needing_embeddings(self, model: str, limit: int = 0) -> list[sqlite3.Row]:
        # The hash depends on each row, so stale rows are filtered in Python.
        with self.connect() as db:
            rows = db.execute("""SELECT m.id,m.kind,m.summary,m.keywords,e.content_hash
              FROM memories m LEFT JOIN memory_embeddings e
              ON e.memory_id=m.id AND e.model=? ORDER BY m.id""", (model,)).fetchall()
        pending = [row for row in rows
                   if row["content_hash"] != embedding_content_hash(memory_embedding_text(row))]
        return pending[:limit] if limit else pending

    def save_embeddings(self, model: str,
                        rows: list[tuple[int, str, list[float]]]) -> int:
        now = int(time.time())
        packed = []
        for memory_id, text, values in rows:
            blob, dimensions = pack_embedding(values)
            packed.append((memory_id, model, dimensions, blob,
                           embedding_content_hash(text), now))
        with self.connect() as db:
            db.executemany("""INSERT INTO memory_embeddings
              (memory_id,model,dimensions,embedding,content_hash,embedded_at)
              VALUES(?,?,?,?,?,?) ON CONFLICT(memory_id,model) DO UPDATE SET
              dimensions=excluded.dimensions,embedding=excluded.embedding,
              content_hash=excluded.content_hash,embedded_at=excluded.embedded_at""", packed)
        return len(packed)

    @staticmethod
    def _insert_reset_marker(
        db: sqlite3.Connection,
        self_id: str,
        peer_id: str,
        mode_key: str,
        namespace: str,
        minimum_rowid: int = 0,
    ) -> int:
        """写入不含正文的严格递增水位标记，防止删除后 SQLite 复用旧 rowid。"""
        marker_id = f"reset-{time.time_ns()}-{hashlib.sha1(f'{self_id}|{peer_id}|{mode_key}|{namespace}'.encode()).hexdigest()[:12]}"
        next_rowid = max(
            int(minimum_rowid) + 1,
            int(db.execute("SELECT coalesce(max(id),0)+1 FROM messages").fetchone()[0]),
        )
        cursor = db.execute(
            """INSERT INTO messages
               (id,msg_id,self_id,sender_id,peer_id,mode_key,namespace,sent_at,text,
                send_type,source_file)
               VALUES(?,?,?,?,?,?,?,strftime('%s','now'),'','-1','context_reset')""",
            (next_rowid, marker_id, self_id, self_id, peer_id, mode_key, namespace),
        )
        return int(cursor.lastrowid)

    @staticmethod
    def _set_context_reset_rowid(
        db: sqlite3.Connection,
        self_id: str,
        peer_id: str,
        mode_key: str,
        namespace: str,
        last_message_rowid: int,
    ) -> None:
        """推进短期上下文恢复水位；原始消息保留，但水位之前的消息不再恢复。"""
        db.execute(
            """INSERT INTO context_reset_state
               (self_id,peer_id,mode_key,namespace,last_message_rowid,updated_at)
               VALUES(?,?,?,?,?,strftime('%s','now'))
               ON CONFLICT(self_id,peer_id,mode_key,namespace) DO UPDATE SET
               last_message_rowid=max(last_message_rowid,excluded.last_message_rowid),
               updated_at=excluded.updated_at""",
            (self_id, peer_id, mode_key, namespace, int(last_message_rowid)),
        )

    def search_pending_runtime_context(
        self,
        query: str,
        self_id: str,
        peer_id: str,
        *,
        mode_key: str = "",
        namespace: str | None = None,
        limit: int = 4,
        scan_limit: int = 48,
        max_chars: int = 1600,
    ) -> str:
        """召回尚未进入长期记忆抽取游标的持久化对话尾巴。

        在线抽取按批次触发，最后不足阈值的几轮会保留在 runtime 表中等待后续消息。
        当进程重启且短期上下文已超时，这些消息既不在内存上下文，也尚未进入长期
        记忆。这里仅在当前 bot+用户+模式+命名空间内做相关性召回，填补这个断层。
        """
        query = (query or "").strip()
        if not query or limit <= 0 or scan_limit <= 0 or max_chars <= 0:
            return ""
        scope = namespace or ""
        with self.connect() as db:
            state = db.execute(
                """SELECT last_message_rowid FROM online_extraction_state
                   WHERE self_id=? AND peer_id=? AND mode_key=? AND namespace=?""",
                (self_id, peer_id, mode_key, scope),
            ).fetchone()
            after_id = int(state["last_message_rowid"]) if state else 0
            reset = db.execute(
                """SELECT last_message_rowid FROM context_reset_state
                   WHERE self_id=? AND peer_id=? AND mode_key=? AND namespace=?""",
                (self_id, peer_id, mode_key, scope),
            ).fetchone()
            if reset:
                after_id = max(after_id, int(reset["last_message_rowid"]))
            rows = db.execute(
                """SELECT id,msg_id,sender_id,sent_at,text,send_type FROM messages
                   WHERE self_id=? AND peer_id=? AND mode_key=? AND namespace=?
                   AND source_file='runtime' AND id>? AND text<>''
                   ORDER BY id DESC LIMIT ?""",
                (self_id, peer_id, mode_key, scope, after_id, int(scan_limit)),
            ).fetchall()
        if not rows:
            return ""

        terms = _terms(query)
        query_grams = _ngrams(query)
        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            score = _runtime_relevance(str(row["text"] or ""), terms, query_grams)
            if score > 0:
                scored.append((score, row))
        if not scored:
            return ""
        scored.sort(key=lambda item: (item[0], int(item[1]["id"])), reverse=True)
        selected_ids = {int(row["id"]) for _, row in scored[:limit]}
        # 给命中的消息补同一轮相邻消息，保留问答语义；最终仍按原时间顺序注入。
        ordered = sorted(rows, key=lambda row: int(row["id"]))
        by_id = {int(row["id"]): index for index, row in enumerate(ordered)}
        expanded_ids = set(selected_ids)
        for message_id in tuple(selected_ids):
            index = by_id[message_id]
            for nearby in ordered[max(0, index - 1):index + 2]:
                expanded_ids.add(int(nearby["id"]))

        lines: list[str] = []
        length = 0
        for row in ordered:
            if int(row["id"]) not in expanded_ids:
                continue
            speaker = "角色" if int(row["send_type"] or 0) == 1 else "用户"
            text = " ".join(str(row["text"] or "").split())[:600]
            line = f"- {speaker}：{text}"
            if length + len(line) + 1 > max_chars:
                break
            lines.append(line)
            length += len(line) + 1
        if not lines:
            return ""
        return "【尚未完成长期抽取的相关对话】\n" + "\n".join(lines)

    def load_recent_context(
        self,
        self_id: str,
        peer_id: str,
        *,
        mode_key: str = "",
        namespace: str | None = None,
        limit: int = 20,
        since: int | None = None,
    ) -> list[sqlite3.Row]:
        """读取可用于重建短期上下文的最近 runtime 消息。

        独立的 ``context_reset_state`` 水位用于隔离 `/清记忆` 前的历史；不能复用
        在线抽取游标，因为正常长期记忆抽取也会推进该游标。
        """
        if limit <= 0:
            return []
        scope = namespace or ""
        with self.connect() as db:
            state = db.execute(
                """SELECT last_message_rowid FROM context_reset_state
                   WHERE self_id=? AND peer_id=? AND mode_key=? AND namespace=?""",
                (self_id, peer_id, mode_key, scope),
            ).fetchone()
            after_id = int(state["last_message_rowid"]) if state else 0
            params: list[object] = [self_id, peer_id, mode_key, scope, after_id]
            time_filter = ""
            if since is not None:
                time_filter = " AND sent_at>=?"
                params.append(int(since))
            params.append(int(limit))
            return db.execute(
                """SELECT * FROM (
                     SELECT id,msg_id,sender_id,sent_at,text,send_type
                     FROM messages
                     WHERE self_id=? AND peer_id=? AND mode_key=? AND namespace=?
                     AND source_file='runtime' AND id>?"""
                + time_filter
                + " ORDER BY id DESC LIMIT ?) ORDER BY id",
                params,
            ).fetchall()

    def load_recent_rounds(
        self,
        self_id: str,
        peer_id: str,
        *,
        mode_key: str = "",
        namespace: str | None = None,
        rounds: int = 2,
    ) -> list[sqlite3.Row]:
        """读取当前作用域最近若干轮 runtime 原始消息。

        一轮以一条用户消息（``send_type=0``）开始，并包含其后的 Bot 回复。
        查询遵守 ``context_reset_state`` 水位，因此 `/清记忆` 前的消息不会重新出现在
        Web 重连窗口。返回结果按数据库 id 正序排列。
        """
        rounds = max(0, min(int(rounds), 20))
        if rounds <= 0:
            return []
        scope = namespace or ""
        with self.connect() as db:
            state = db.execute(
                """SELECT last_message_rowid FROM context_reset_state
                   WHERE self_id=? AND peer_id=? AND mode_key=? AND namespace=?""",
                (self_id, peer_id, mode_key, scope),
            ).fetchone()
            after_id = int(state["last_message_rowid"]) if state else 0
            boundary = db.execute(
                """SELECT id FROM messages
                   WHERE self_id=? AND peer_id=? AND mode_key=? AND namespace=?
                     AND source_file='runtime' AND id>? AND send_type=0
                   ORDER BY id DESC LIMIT 1 OFFSET ?""",
                (self_id, peer_id, mode_key, scope, after_id, rounds - 1),
            ).fetchone()
            if boundary is None:
                # 用户消息不足 N 轮时返回清理水位后的全部现有消息；通常仅几行。
                start_id = after_id
                inclusive = ">"
            else:
                start_id = int(boundary["id"])
                inclusive = ">="
            return db.execute(
                f"""SELECT id,msg_id,sender_id,sent_at,text,send_type
                    FROM messages
                    WHERE self_id=? AND peer_id=? AND mode_key=? AND namespace=?
                      AND source_file='runtime' AND id{inclusive}?
                    ORDER BY id""",
                (self_id, peer_id, mode_key, scope, start_id),
            ).fetchall()

    def clear_session(
        self,
        self_id: str,
        peer_id: str,
        namespace: str | None = None,
        mode_key: str = "",
    ) -> int:
        """清除某会话作用域的长期记忆、状态与 runtime 原始消息。

        按 owner_id（含空串全局）与 subject_id 定位，级联删除其向量。
        - 给定 ``namespace``（tavern 的当前角色 / writer 的当前作品）时，只清该命名空间的
          self / relationship / episode，**保留 person（关于用户本人，跨命名空间共享）**与其他命名空间。
        - ``namespace`` 为 None（clone 或尚未选定实体）时退化为旧行为：清 peer_id + relationship。
        """
        if namespace:
            subjects = (f"{namespace}:self:{peer_id}", f"{namespace}:rel:{self_id}:{peer_id}")
        else:
            subjects = (peer_id, f"relationship:{self_id}:{peer_id}")
        with self.connect() as db:
            cur = db.execute(
                "DELETE FROM memories WHERE owner_id IN (?, '') AND subject_id IN (?,?)",
                (self_id, *subjects),
            )
            deleted = cur.rowcount
            scope = namespace or ""
            db.execute(
                """DELETE FROM interaction_state
                   WHERE self_id=? AND peer_id=? AND mode_key=? AND namespace=?""",
                (self_id, peer_id, mode_key, scope),
            )
            previous_max_rowid = int(
                db.execute("SELECT coalesce(max(id),0) FROM messages").fetchone()[0]
            )
            db.execute(
                """DELETE FROM messages
                   WHERE self_id=? AND peer_id=? AND mode_key=? AND namespace=?""",
                (self_id, peer_id, mode_key, scope),
            )
            last_rowid = self._insert_reset_marker(
                db, self_id, peer_id, mode_key, scope, previous_max_rowid
            )
            db.execute(
                """INSERT INTO online_extraction_state
                   (self_id,peer_id,mode_key,namespace,last_message_rowid,last_extracted_at,
                    status,error,updated_at) VALUES(?,?,?,?,?,NULL,'idle',NULL,strftime('%s','now'))
                   ON CONFLICT(self_id,peer_id,mode_key,namespace) DO UPDATE SET
                   last_message_rowid=excluded.last_message_rowid,last_extracted_at=NULL,
                   status='idle',error=NULL,updated_at=excluded.updated_at""",
                (self_id, peer_id, mode_key, scope, int(last_rowid)),
            )
            self._set_context_reset_rowid(
                db, self_id, peer_id, mode_key, scope, int(last_rowid)
            )
        return deleted

    def clear_user(self, self_id: str, peer_id: str, mode_key: str = "") -> int:
        """清除当前 bot 与当前用户在本模式的全部记忆、状态和 runtime 消息。"""
        relation_like = f"%:rel:{self_id}:{peer_id}"
        self_like = f"%:self:{peer_id}"
        legacy_relation = f"relationship:{self_id}:{peer_id}"
        with self.connect() as db:
            cur = db.execute(
                """DELETE FROM memories WHERE owner_id IN (?,'') AND
                   (subject_id=? OR subject_id=? OR subject_id=?
                    OR subject_id LIKE ? OR subject_id LIKE ?)""",
                (self_id, peer_id, self_id, legacy_relation, relation_like, self_like),
            )
            deleted = cur.rowcount
            db.execute(
                "DELETE FROM interaction_state WHERE self_id=? AND peer_id=? AND mode_key=?",
                (self_id, peer_id, mode_key),
            )
            scopes = {
                str(row["namespace"])
                for row in db.execute(
                    """SELECT DISTINCT namespace FROM messages
                       WHERE self_id=? AND peer_id=? AND mode_key=?
                       AND source_file='runtime'""",
                    (self_id, peer_id, mode_key),
                ).fetchall()
            }
            # interaction_state 可能代表仅有开场状态、尚无 runtime 消息的作用域。
            scopes.update(
                str(row["namespace"])
                for row in db.execute(
                    """SELECT DISTINCT namespace FROM online_extraction_state
                       WHERE self_id=? AND peer_id=? AND mode_key=?""",
                    (self_id, peer_id, mode_key),
                ).fetchall()
            )
            if not scopes:
                scopes.add("")
            previous_max_rowid = int(
                db.execute("SELECT coalesce(max(id),0) FROM messages").fetchone()[0]
            )
            db.execute(
                """DELETE FROM messages
                   WHERE self_id=? AND peer_id=? AND mode_key=?""",
                (self_id, peer_id, mode_key),
            )
            marker_floor = previous_max_rowid
            for scope in scopes:
                last_id = self._insert_reset_marker(
                    db, self_id, peer_id, mode_key, scope, marker_floor
                )
                marker_floor = last_id
                db.execute(
                    """INSERT INTO online_extraction_state
                       (self_id,peer_id,mode_key,namespace,last_message_rowid,last_extracted_at,
                        status,error,updated_at) VALUES(?,?,?,?,?,NULL,'idle',NULL,strftime('%s','now'))
                       ON CONFLICT(self_id,peer_id,mode_key,namespace) DO UPDATE SET
                       last_message_rowid=excluded.last_message_rowid,last_extracted_at=NULL,
                       status='idle',error=NULL,updated_at=excluded.updated_at""",
                    (self_id, peer_id, mode_key, scope, int(last_id)),
                )
                self._set_context_reset_rowid(
                    db, self_id, peer_id, mode_key, scope, int(last_id)
                )
        return deleted

    def clear_all(self) -> int:
        """运维接口：清除本库所有用户的长期记忆；聊天命令不得调用。"""
        with self.connect() as db:
            cur = db.execute("DELETE FROM memories")
            deleted = cur.rowcount
            db.execute("DELETE FROM interaction_state")
            scopes = db.execute(
                """SELECT self_id,peer_id,mode_key,namespace,coalesce(max(id),0) last_id
                   FROM messages WHERE source_file='runtime'
                   GROUP BY self_id,peer_id,mode_key,namespace"""
            ).fetchall()
            for row in scopes:
                self._set_context_reset_rowid(
                    db,
                    row["self_id"],
                    row["peer_id"],
                    row["mode_key"],
                    row["namespace"],
                    int(row["last_id"]),
                )
            db.execute("""UPDATE online_extraction_state SET
              last_message_rowid=coalesce((SELECT max(m.id) FROM messages m
                WHERE m.self_id=online_extraction_state.self_id
                AND m.peer_id=online_extraction_state.peer_id
                AND m.mode_key=online_extraction_state.mode_key
                AND m.namespace=online_extraction_state.namespace),0),
              last_extracted_at=NULL,status='idle',error=NULL,
              updated_at=strftime('%s','now')""")
        return deleted

    def has_namespace_memories(
        self, self_id: str, peer_id: str, namespace: str
    ) -> bool:
        """判断角色/作品命名空间是否已有专属长期记忆。

        ``person`` 是跨角色共享的用户事实，不计入某个角色是否已与用户见过。
        """
        subjects = (
            f"{namespace}:self:{peer_id}",
            f"{namespace}:rel:{self_id}:{peer_id}",
        )
        with self.connect() as db:
            row = db.execute(
                """SELECT 1 FROM memories
                   WHERE owner_id IN (?, '') AND subject_id IN (?, ?)
                   AND status IN ('active','disputed')
                   AND (valid_to IS NULL OR valid_to>=strftime('%s','now'))
                   LIMIT 1""",
                (self_id, *subjects),
            ).fetchone()
        return row is not None

    def list_memories(self, self_id: str, peer_id: str,
                      namespace: str | None = None, limit: int = 50) -> list[dict]:
        """列出某作用域下的长期记忆（只读回顾用）。

        - 给定 ``namespace``：列该命名空间（角色/作品）的 self/relationship/episode，
          并附带 person（关于用户本人，跨命名空间共享，在该上下文同样相关）。
        - ``namespace`` 为 None：列 peer_id + relationship（旧格式，本用户当前模式）。
        """
        if namespace:
            subjects = (f"{namespace}:self:{peer_id}",
                        f"{namespace}:rel:{self_id}:{peer_id}", peer_id)
        else:
            subjects = (peer_id, f"relationship:{self_id}:{peer_id}")
        with self.connect() as db:
            sub_ph = ",".join("?" for _ in subjects)
            rows = db.execute(
                f"SELECT kind, summary, keywords, confidence, valid_from, valid_to "
                f"FROM memories WHERE owner_id IN (?, '') AND subject_id IN ({sub_ph}) "
                f"AND status IN ('active','disputed') "
                f"AND (valid_to IS NULL OR valid_to>=strftime('%s','now')) "
                f"ORDER BY kind, valid_from DESC LIMIT ?",
                (self_id, *subjects, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_user(
        self, self_id: str, peer_id: str, limit: int = 200
    ) -> list[dict]:
        """列出当前 bot 与当前用户在本模式的全部有效记忆。"""
        relation_like = f"%:rel:{self_id}:{peer_id}"
        self_like = f"%:self:{peer_id}"
        legacy_relation = f"relationship:{self_id}:{peer_id}"
        with self.connect() as db:
            rows = db.execute(
                """SELECT kind,summary,keywords,confidence,valid_from,valid_to
                   FROM memories WHERE owner_id IN (?,'')
                   AND status IN ('active','disputed')
                   AND (valid_to IS NULL OR valid_to>=strftime('%s','now'))
                   AND (subject_id=? OR subject_id=? OR subject_id=?
                        OR subject_id LIKE ? OR subject_id LIKE ?)
                   ORDER BY kind,valid_from DESC LIMIT ?""",
                (self_id, peer_id, self_id, legacy_relation,
                 relation_like, self_like, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_all(self, limit: int = 200) -> list[dict]:
        """运维接口：列出本库所有用户的有效记忆；聊天命令不得调用。"""
        with self.connect() as db:
            rows = db.execute(
                """SELECT kind,summary,keywords,confidence,valid_from,valid_to
                   FROM memories WHERE status IN ('active','disputed')
                   AND (valid_to IS NULL OR valid_to>=strftime('%s','now'))
                   ORDER BY kind,valid_from DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def has_runtime_messages(
        self,
        self_id: str,
        peer_id: str,
        mode_key: str = "",
        namespace: str | None = None,
    ) -> bool:
        """判断某会话作用域是否已有持久化原始消息。"""
        with self.connect() as db:
            row = db.execute(
                """SELECT 1 FROM messages
                   WHERE self_id=? AND peer_id=? AND mode_key=? AND namespace=?
                   AND source_file='runtime' LIMIT 1""",
                (self_id, peer_id, mode_key, namespace or ""),
            ).fetchone()
        return row is not None

    def save_runtime_message(
        self,
        msg_id: str,
        self_id: str,
        sender_id: str,
        peer_id: str,
        sent_at: int,
        text: str,
        send_type: int,
        mode_key: str = "",
        namespace: str | None = None,
    ) -> int:
        with self.connect() as db:
            cursor = db.execute("""INSERT OR IGNORE INTO messages
              (msg_id,self_id,sender_id,peer_id,mode_key,namespace,sent_at,text,send_type,source_file)
              VALUES(?,?,?,?,?,?,?,?,?,'runtime')""",
              (msg_id, self_id, sender_id, peer_id, mode_key, namespace or "",
               sent_at, text, send_type))
            return cursor.rowcount

    def get_interaction_state(
        self, self_id: str, peer_id: str, mode_key: str = "", namespace: str | None = None
    ) -> sqlite3.Row | None:
        with self.connect() as db:
            return db.execute("""SELECT * FROM interaction_state
              WHERE self_id=? AND peer_id=? AND mode_key=? AND namespace=?""",
              (self_id, peer_id, mode_key, namespace or "")).fetchone()

    def touch_interaction(
        self,
        self_id: str,
        peer_id: str,
        *,
        user_at: int | None = None,
        bot_at: int | None = None,
        mode_key: str = "",
        namespace: str | None = None,
    ) -> None:
        now = int(time.time())
        with self.connect() as db:
            db.execute("""INSERT INTO interaction_state
              (self_id,peer_id,mode_key,namespace,last_user_at,last_bot_at,updated_at)
              VALUES(?,?,?,?,?,?,?)
              ON CONFLICT(self_id,peer_id,mode_key,namespace) DO UPDATE SET
              last_user_at=coalesce(excluded.last_user_at,last_user_at),
              last_bot_at=coalesce(excluded.last_bot_at,last_bot_at),
              updated_at=excluded.updated_at""",
              (self_id, peer_id, mode_key, namespace or "", user_at, bot_at, now))

    def get_online_extraction_batch(
        self,
        self_id: str,
        peer_id: str,
        limit: int = 40,
        mode_key: str = "",
        namespace: str | None = None,
    ) -> list[sqlite3.Row]:
        scope = namespace or ""
        with self.connect() as db:
            state = db.execute("SELECT last_message_rowid FROM online_extraction_state "
                               "WHERE self_id=? AND peer_id=? AND mode_key=? AND namespace=?",
                               (self_id, peer_id, mode_key, scope)).fetchone()
            after_id = int(state["last_message_rowid"]) if state else 0
            return db.execute("""SELECT id,msg_id,sender_id,sent_at,text FROM messages
              WHERE self_id=? AND peer_id=? AND mode_key=? AND namespace=?
              AND source_file='runtime' AND id>? ORDER BY id LIMIT ?""",
              (self_id, peer_id, mode_key, scope, after_id, limit)).fetchall()

    def is_online_extraction_batch_current(
        self,
        self_id: str,
        peer_id: str,
        last_message_rowid: int,
        mode_key: str = "",
        namespace: str | None = None,
    ) -> bool:
        """判断抽取批次是否仍有效；清理推进游标后，旧批次必须作废。"""
        with self.connect() as db:
            state = db.execute(
                """SELECT last_message_rowid FROM online_extraction_state
                   WHERE self_id=? AND peer_id=? AND mode_key=? AND namespace=?""",
                (self_id, peer_id, mode_key, namespace or ""),
            ).fetchone()
        return state is None or int(state["last_message_rowid"]) < int(last_message_rowid)

    def finish_online_extraction(
        self,
        self_id: str,
        peer_id: str,
        last_message_rowid: int | None,
        error: str | None = None,
        mode_key: str = "",
        namespace: str | None = None,
    ) -> None:
        now = int(time.time())
        status = "failed" if error else "idle"
        scope = namespace or ""
        with self.connect() as db:
            db.execute("""INSERT INTO online_extraction_state
              (self_id,peer_id,mode_key,namespace,last_message_rowid,last_extracted_at,
               status,error,updated_at)
              VALUES(?,?,?,?,?,?,?,?,?)
              ON CONFLICT(self_id,peer_id,mode_key,namespace) DO UPDATE SET
              last_message_rowid=CASE WHEN excluded.last_message_rowid>0
                THEN excluded.last_message_rowid ELSE last_message_rowid END,
              last_extracted_at=CASE WHEN excluded.error IS NULL
                THEN excluded.last_extracted_at ELSE last_extracted_at END,
              status=excluded.status,error=excluded.error,updated_at=excluded.updated_at""",
              (self_id, peer_id, mode_key, scope, last_message_rowid or 0,
               None if error else now, status, error[:1000] if error else None, now))

    def memory_ids_for_items(self, items: list[dict]) -> list[int]:
        fingerprints = []
        for item in items:
            kind = str(item.get("kind", "")).strip()
            owner = str(item.get("owner_id", "")).strip()
            subject = str(item.get("subject_id", "")).strip()
            summary = str(item.get("summary", "")).strip()
            if kind and subject and summary:
                fingerprints.append(f"{owner}|{kind}|{subject}|{summary}")
        if not fingerprints:
            return []
        marks = ",".join("?" for _ in fingerprints)
        with self.connect() as db:
            return [int(row["id"]) for row in db.execute(
                f"SELECT id FROM memories WHERE fingerprint IN ({marks})", fingerprints)]

    def memories_by_ids(self, memory_ids: list[int]) -> list[sqlite3.Row]:
        if not memory_ids:
            return []
        marks = ",".join("?" for _ in memory_ids)
        with self.connect() as db:
            return db.execute(f"SELECT id,kind,summary,keywords FROM memories "
                              f"WHERE id IN ({marks})", memory_ids).fetchall()

    def save_memories(self, items: list[dict], evidence_ids: list[str]) -> int:
        now = int(time.time())
        rows = []
        for x in items:
            summary = str(x.get("summary", "")).strip()
            owner = str(x.get("owner_id", "")).strip()
            subject = str(x.get("subject_id", "")).strip()
            kind = str(x.get("kind", "")).strip()
            if not summary or not subject or not kind:
                continue
            fingerprint = f"{owner}|{kind}|{subject}|{summary}"
            rows.append((owner, kind, subject, x.get("object_id"), summary,
                         " ".join(x.get("keywords") or []),
                         float(x.get("confidence", .7)), x.get("valid_from"),
                         x.get("valid_to"), json.dumps(evidence_ids, ensure_ascii=False),
                         now, now, fingerprint))
        with self.connect() as db:
            before = db.execute("SELECT count(*) FROM memories").fetchone()[0]
            db.executemany("""INSERT INTO memories
              (owner_id,kind,subject_id,object_id,summary,keywords,confidence,valid_from,valid_to,
               evidence_json,created_at,updated_at,fingerprint)
              VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(fingerprint) DO UPDATE SET
              confidence=max(confidence,excluded.confidence),updated_at=excluded.updated_at,
              evidence_json=excluded.evidence_json""", rows)
            after = db.execute("SELECT count(*) FROM memories").fetchone()[0]
            return after - before

    def save_memories_semantic(
        self,
        items: list[dict],
        evidence_ids: list[str],
        embedding_model: str,
        embeddings: list[list[float]],
        similarity_threshold: float = .84,
    ) -> tuple[int, int, list[int]]:
        """写入新记忆，并把高相似的同主体、同类型表述合并到已有记忆。

        返回 ``(新增数, 语义合并数, 最终记忆ID)``。阈值故意保持较高，避免
        “相关但不同”的事实自动合并；较复杂的旧数据清理由离线模型判定完成。
        """
        if len(items) != len(embeddings):
            raise ValueError("items 与 embeddings 数量不一致")

        now = int(time.time())
        evidence = json.dumps(list(dict.fromkeys(str(x) for x in evidence_ids)),
                              ensure_ascii=False)
        created = merged = 0
        memory_ids: list[int] = []

        with self.connect() as db:
            for item, values in zip(items, embeddings):
                summary = str(item.get("summary", "")).strip()
                owner = str(item.get("owner_id", "")).strip()
                subject = str(item.get("subject_id", "")).strip()
                kind = str(item.get("kind", "")).strip()
                if not summary or not subject or not kind:
                    continue
                keywords = " ".join(str(x).strip() for x in (item.get("keywords") or [])
                                    if str(x).strip())
                confidence = min(.95, max(.05, float(item.get("confidence", .7))))
                fingerprint = f"{owner}|{kind}|{subject}|{summary}"
                packed, dimensions = pack_embedding(values)
                vector = unpack_embedding(packed, dimensions)

                exact = db.execute(
                    "SELECT * FROM memories WHERE fingerprint=?", (fingerprint,)
                ).fetchone()
                target = None
                if exact:
                    target = exact
                    if exact["status"] not in {"active", "disputed"} and exact["supersedes_id"]:
                        replacement = db.execute(
                            "SELECT * FROM memories WHERE id=?", (exact["supersedes_id"],)
                        ).fetchone()
                        if replacement and replacement["status"] in {"active", "disputed"}:
                            target = replacement

                if target is None:
                    candidates = db.execute("""SELECT m.*,e.dimensions,e.embedding
                      FROM memories m JOIN memory_embeddings e ON e.memory_id=m.id
                      WHERE e.model=? AND m.owner_id=? AND m.subject_id=? AND m.kind=?
                      AND m.status IN ('active','disputed')""",
                      (embedding_model, owner, subject, kind)).fetchall()
                    best_similarity = -1.0
                    for candidate in candidates:
                        try:
                            candidate_vector = unpack_embedding(
                                candidate["embedding"], int(candidate["dimensions"]))
                        except ValueError:
                            continue
                        similarity = _dot(vector, candidate_vector)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            target = candidate
                    if best_similarity < similarity_threshold:
                        target = None

                if target is not None:
                    old_evidence = json.loads(target["evidence_json"] or "[]")
                    new_evidence = json.loads(evidence)
                    merged_evidence = json.dumps(
                        list(dict.fromkeys(str(x) for x in old_evidence + new_evidence)),
                        ensure_ascii=False,
                    )
                    old_keywords = str(target["keywords"] or "").split()
                    merged_keywords = " ".join(dict.fromkeys(old_keywords + keywords.split()))
                    db.execute("""UPDATE memories SET confidence=max(confidence,?),
                      keywords=?,evidence_json=?,updated_at=? WHERE id=?""",
                      (confidence, merged_keywords, merged_evidence, now, int(target["id"])))
                    memory_ids.append(int(target["id"]))
                    merged += 1
                    continue

                cursor = db.execute("""INSERT INTO memories
                  (owner_id,kind,subject_id,object_id,summary,keywords,confidence,valid_from,valid_to,
                   status,evidence_json,created_at,updated_at,fingerprint)
                  VALUES(?,?,?,?,?,?,?,?,?, 'active',?,?,?,?)""",
                  (owner, kind, subject, item.get("object_id"), summary, keywords, confidence,
                   item.get("valid_from"), item.get("valid_to"), evidence, now, now,
                   fingerprint))
                memory_id = int(cursor.lastrowid)
                db.execute("""INSERT INTO memory_embeddings
                  (memory_id,model,dimensions,embedding,content_hash,embedded_at)
                  VALUES(?,?,?,?,?,?)""",
                  (memory_id, embedding_model, dimensions, packed,
                   embedding_content_hash(memory_embedding_text({
                       "kind": kind, "summary": summary, "keywords": keywords,
                   })), now))
                memory_ids.append(memory_id)
                created += 1

        return created, merged, memory_ids
