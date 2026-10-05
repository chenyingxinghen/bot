from __future__ import annotations

import tempfile
import time
import unittest
import sys
import sqlite3
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.memory.online import OnlineMemoryExtractor, _writer_person_is_supported
from core.memory.store import MemoryStore, format_memory_time, memory_embedding_text, _scope_subject
from tools.memory.resolve_conflicts import (
    apply_result,
    collapse_duplicate_links,
    repair_duplicate_cycles,
)


class MemorySystemTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.tmp.name) / "memory.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_schema_migrates_and_persists_interaction(self):
        self.store.touch_interaction("self-a", "peer", user_at=100, bot_at=120)
        self.store.touch_interaction("self-b", "peer", user_at=200, bot_at=220)
        state = self.store.get_interaction_state("self-a", "peer")
        self.assertEqual(state["last_user_at"], 100)
        self.assertEqual(state["last_bot_at"], 120)
        other = self.store.get_interaction_state("self-b", "peer")
        self.assertEqual(other["last_user_at"], 200)
        with self.store.connect() as db:
            columns = {row["name"] for row in db.execute("pragma table_info(memories)")}
            message_columns = {
                row["name"] for row in db.execute("pragma table_info(messages)")
            }
        self.assertIn("supersedes_id", columns)
        self.assertIn("conflict_group", columns)
        self.assertIn("self_id", message_columns)
        self.assertIn("mode_key", message_columns)
        self.assertIn("namespace", message_columns)

    def test_legacy_account_tables_are_migrated(self):
        legacy_path = Path(self.tmp.name) / "legacy.db"
        db = sqlite3.connect(legacy_path)
        db.executescript("""
          CREATE TABLE messages (
            id INTEGER PRIMARY KEY, msg_id TEXT UNIQUE, sender_id TEXT NOT NULL,
            peer_id TEXT NOT NULL, sent_at INTEGER NOT NULL, text TEXT NOT NULL,
            send_type INTEGER, source_file TEXT NOT NULL);
          CREATE TABLE interaction_state (
            peer_id TEXT PRIMARY KEY, last_user_at INTEGER, last_bot_at INTEGER,
            updated_at INTEGER NOT NULL);
          INSERT INTO interaction_state VALUES('peer',100,120,130);
          CREATE TABLE online_extraction_state (
            peer_id TEXT PRIMARY KEY, last_message_rowid INTEGER NOT NULL DEFAULT 0,
            last_extracted_at INTEGER, status TEXT NOT NULL DEFAULT 'idle',
            error TEXT, updated_at INTEGER NOT NULL);
        """)
        db.commit()
        db.close()

        store = MemoryStore(legacy_path)
        state = store.get_interaction_state("", "peer")
        self.assertEqual(state["last_bot_at"], 120)
        with store.connect() as migrated:
            columns = {
                row["name"] for row in migrated.execute("pragma table_info(messages)")
            }
        self.assertIn("self_id", columns)

    def test_character_first_visit_checks_are_scope_specific(self):
        now = int(time.time())
        namespace = "char:bocchi"
        # 跨角色共享的 person 事实不代表 Bocchi 已经与该用户见过。
        self.store.save_memories([
            {"owner_id": "self", "kind": "person", "subject_id": "peer",
             "summary": "用户喜欢吉他", "keywords": ["吉他"], "confidence": .9,
             "valid_from": now, "valid_to": None},
        ], ["person-seed"])
        self.assertFalse(self.store.has_namespace_memories("self", "peer", namespace))
        self.assertFalse(self.store.has_runtime_messages(
            "self", "peer", mode_key="tavern", namespace=namespace
        ))

        self.store.save_memories([
            {"owner_id": "self", "kind": "relationship",
             "subject_id": f"{namespace}:rel:self:peer",
             "summary": "Bocchi 已经认识用户", "keywords": ["认识"],
             "confidence": .9, "valid_from": now, "valid_to": None},
        ], ["relationship-seed"])
        self.assertTrue(self.store.has_namespace_memories("self", "peer", namespace))
        self.assertFalse(self.store.has_namespace_memories(
            "self", "peer", "char:another"
        ))

        self.store.save_runtime_message(
            "greeting-1", "self", "self", "peer", now, "你好呀", 1,
            mode_key="tavern", namespace=namespace,
        )
        self.assertTrue(self.store.has_runtime_messages(
            "self", "peer", mode_key="tavern", namespace=namespace
        ))
        self.assertFalse(self.store.has_runtime_messages(
            "self", "peer", mode_key="tavern", namespace="char:another"
        ))

    def test_embedding_roundtrip_and_hybrid_retrieval(self):
        now = int(time.time())
        self.store.save_memories([
            {"owner_id": "self", "kind": "person", "subject_id": "peer",
             "summary": "对方喜欢阅读科幻小说",
             "keywords": ["阅读", "科幻", "小说"], "confidence": .9,
             "valid_from": now - 86400 * 10, "valid_to": None},
            {"owner_id": "self", "kind": "person", "subject_id": "peer",
             "summary": "对方喜欢打篮球",
             "keywords": ["运动", "篮球"], "confidence": .8,
             "valid_from": now - 86400 * 20, "valid_to": None},
        ], ["1"])
        rows = self.store.memories_needing_embeddings("test")
        vectors = {"对方喜欢阅读科幻小说": [1.0, 0.0], "对方喜欢打篮球": [0.0, 1.0]}
        self.store.save_embeddings("test", [
            (row["id"], memory_embedding_text(row), vectors[row["summary"]])
            for row in rows
        ])
        context = self.store.search_context(
            "最近读什么", "self", "peer", limit=1, stable_limit=0,
            query_embedding=[1.0, 0.0], embedding_model="test",
        )
        self.assertIn("科幻小说", context)
        # person 类记忆的时间标签形如「自 yyyy-mm-dd 起，N天前记录」（episode 才是「发生于」），
        # 这里验证时间标签确实被格式化输出（测试仅插入 person 类记忆，故用「记录」而非「发生」）。
        self.assertIn("记录", context)
        other_context = self.store.search_context(
            "最近读什么", "other-self", "peer", limit=1, stable_limit=0,
            query_embedding=[1.0, 0.0], embedding_model="test",
        )
        self.assertEqual(other_context, "")

    def test_retrieval_pairs_kind_with_subject_and_rejects_dirty_rows(self):
        now = int(time.time())
        self.store.save_memories([
            {"owner_id": "self", "kind": "person", "subject_id": "peer",
             "summary": "用户喜欢苹果", "keywords": ["苹果"], "confidence": .9,
             "valid_from": now, "valid_to": None},
            # 历史脏数据：self 类型却挂在 person subject，不能进入召回。
            {"owner_id": "self", "kind": "self", "subject_id": "peer",
             "summary": "错误的自我设定苹果", "keywords": ["苹果"], "confidence": .95,
             "valid_from": now, "valid_to": None},
        ], ["1"])
        context = self.store.search_context("苹果", "self", "peer")
        self.assertIn("用户喜欢苹果", context)
        self.assertNotIn("错误的自我设定", context)

    def test_temporal_intent_recalls_recent_episode_without_topic_words(self):
        now = int(time.time())
        subject = _scope_subject("work:book", "episode", "self", "peer")
        self.store.save_memories([
            {"owner_id": "self", "kind": "episode", "subject_id": subject,
             "summary": "主角在钟楼发现了一封密信", "keywords": ["钟楼", "密信"],
             "confidence": .88, "valid_from": now - 60, "valid_to": None},
        ], ["1"])
        context = self.store.search_context(
            "最近发生了什么", "self", "peer", namespace="work:book",
            allowed_kinds={"episode"}, stable_limit=0,
        )
        self.assertIn("钟楼", context)

    def test_stable_background_is_not_injected_by_default(self):
        self.store.save_memories([
            {"owner_id": "self", "kind": "person", "subject_id": "peer",
             "summary": "用户喜欢篮球", "keywords": ["篮球"], "confidence": .95,
             "valid_from": None, "valid_to": None},
        ], ["1"])
        self.assertEqual(self.store.search_context("天气怎么样", "self", "peer"), "")

    def test_baseline_memories_are_stable_scoped_and_exclude_episodes(self):
        now = int(time.time())
        ns_a, ns_b = "char:bocchi", "char:melissa"
        self.store.save_memories([
            {"owner_id": "self", "kind": "person", "subject_id": "peer",
             "summary": "用户喜欢科幻", "keywords": ["科幻"], "confidence": .92,
             "valid_from": now, "valid_to": None},
            {"owner_id": "self", "kind": "self", "subject_id": f"{ns_a}:self:peer",
             "summary": "Bocchi 是吉他手", "keywords": ["吉他"], "confidence": .9,
             "valid_from": now, "valid_to": None},
            {"owner_id": "self", "kind": "relationship",
             "subject_id": f"{ns_a}:rel:self:peer", "summary": "双方约定叫对方主人",
             "keywords": ["主人"], "confidence": .88,
             "valid_from": now, "valid_to": None},
            {"owner_id": "self", "kind": "episode",
             "subject_id": f"{ns_a}:rel:self:peer", "summary": "昨天一起去了 live",
             "keywords": ["live"], "confidence": .95,
             "valid_from": now, "valid_to": None},
            {"owner_id": "self", "kind": "self", "subject_id": f"{ns_b}:self:peer",
             "summary": "Melissa 是社长", "keywords": ["社长"], "confidence": .95,
             "valid_from": now, "valid_to": None},
            {"owner_id": "self", "kind": "person", "subject_id": "peer",
             "summary": "用户可能喜欢跑步", "keywords": ["跑步"], "confidence": .7,
             "valid_from": now, "valid_to": None},
        ], ["baseline"])
        with self.store.connect() as db:
            db.execute(
                "UPDATE memories SET status='disputed' WHERE summary=?",
                ("用户喜欢科幻",),
            )
        rows = self.store.baseline_memories("self", "peer", namespace=ns_a)
        summaries = {row["summary"] for row in rows}
        self.assertEqual(summaries, {
            "Bocchi 是吉他手", "双方约定叫对方主人",
        })
        self.assertNotIn("用户喜欢科幻", summaries)
        self.assertNotIn("昨天一起去了 live", summaries)
        self.assertNotIn("Melissa 是社长", summaries)
        self.assertNotIn("用户可能喜欢跑步", summaries)

    def test_baseline_memories_respect_kind_quotas(self):
        now = int(time.time())
        self.store.save_memories([
            {"owner_id": "self", "kind": "person", "subject_id": "peer",
             "summary": f"稳定事实{i}", "keywords": [f"事实{i}"],
             "confidence": .9 + i * .01, "valid_from": now, "valid_to": None}
            for i in range(4)
        ], ["quota"])
        rows = self.store.baseline_memories(
            "self", "peer", kind_limits={"person": 2, "self": 0, "relationship": 0}
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            [row["summary"] for row in rows], ["稳定事实3", "稳定事实2"]
        )

    def test_runtime_messages_and_extraction_cursor(self):
        self.assertEqual(self.store.save_runtime_message(
            "m1", "self-a", "peer", "peer", 100, "你好", 1), 1)
        self.assertEqual(self.store.save_runtime_message(
            "m1", "self-a", "peer", "peer", 100, "你好", 1), 0)
        self.assertEqual(self.store.save_runtime_message(
            "m2", "self-b", "peer", "peer", 101, "另一个账号", 1), 1)
        batch = self.store.get_online_extraction_batch("self-a", "peer")
        self.assertEqual(len(batch), 1)
        self.store.finish_online_extraction("self-a", "peer", batch[-1]["id"])
        self.assertEqual(self.store.get_online_extraction_batch("self-a", "peer"), [])
        self.assertEqual(
            len(self.store.get_online_extraction_batch("self-b", "peer")), 1)

    def test_runtime_messages_are_isolated_by_mode_and_namespace(self):
        self.store.save_runtime_message(
            "a1", "self", "peer", "peer", 100, "作品A用户", 0,
            mode_key="writer", namespace="work:a")
        self.store.save_runtime_message(
            "a2", "self", "self", "peer", 101, "作品A回复", 1,
            mode_key="writer", namespace="work:a")
        self.store.save_runtime_message(
            "b1", "self", "peer", "peer", 102, "作品B用户", 0,
            mode_key="writer", namespace="work:b")
        batch_a = self.store.get_online_extraction_batch(
            "self", "peer", mode_key="writer", namespace="work:a")
        batch_b = self.store.get_online_extraction_batch(
            "self", "peer", mode_key="writer", namespace="work:b")
        self.assertEqual([row["text"] for row in batch_a], ["作品A用户", "作品A回复"])
        self.assertEqual([row["text"] for row in batch_b], ["作品B用户"])

    def test_load_recent_context_respects_scope_timeout_and_limit(self):
        now = int(time.time())
        self.store.save_runtime_message(
            "a-old", "self", "peer", "peer", now - 9999, "过期消息", 0,
            mode_key="writer", namespace="work:a")
        self.store.save_runtime_message(
            "a-user", "self", "peer", "peer", now - 3, "作品A用户", 0,
            mode_key="writer", namespace="work:a")
        self.store.save_runtime_message(
            "a-bot", "self", "self", "peer", now - 2, "作品A回复", 1,
            mode_key="writer", namespace="work:a")
        self.store.save_runtime_message(
            "b-user", "self", "peer", "peer", now - 1, "作品B用户", 0,
            mode_key="writer", namespace="work:b")

        rows = self.store.load_recent_context(
            "self", "peer", mode_key="writer", namespace="work:a",
            limit=2, since=now - 100,
        )
        self.assertEqual([row["text"] for row in rows], ["作品A用户", "作品A回复"])

    def test_load_recent_rounds_returns_exactly_last_two_complete_rounds(self):
        for index in range(1, 4):
            self.store.save_runtime_message(
                f"u{index}", "self", "peer", "peer", 100 + index * 2,
                f"用户{index}", 0, mode_key="clone")
            self.store.save_runtime_message(
                f"b{index}", "self", "self", "peer", 101 + index * 2,
                f"回复{index}", 1, mode_key="clone")
        rows = self.store.load_recent_rounds(
            "self", "peer", mode_key="clone", rounds=2)
        self.assertEqual(
            [row["text"] for row in rows],
            ["用户2", "回复2", "用户3", "回复3"],
        )

    def test_load_recent_rounds_respects_scope_and_reset_watermark(self):
        self.store.save_runtime_message(
            "old-u", "self", "peer", "peer", 100, "旧用户", 0,
            mode_key="writer", namespace="work:a")
        self.store.save_runtime_message(
            "other", "self", "peer", "peer", 101, "其他作品", 0,
            mode_key="writer", namespace="work:b")
        self.store.clear_session("self", "peer", "work:a", mode_key="writer")
        self.store.save_runtime_message(
            "new-u", "self", "peer", "peer", 102, "新用户", 0,
            mode_key="writer", namespace="work:a")
        self.store.save_runtime_message(
            "new-b", "self", "self", "peer", 103, "新回复", 1,
            mode_key="writer", namespace="work:a")
        rows = self.store.load_recent_rounds(
            "self", "peer", mode_key="writer", namespace="work:a", rounds=2)
        self.assertEqual([row["text"] for row in rows], ["新用户", "新回复"])

    def test_title_question_recalls_relationship_memory_without_embedding(self):
        self.store.save_memories([{
            "kind": "relationship",
            "subject_id": "char:bocchi:rel:self:peer",
            "object_id": "peer",
            "summary": "Bocchi答应以后称呼对方为“主人”。",
            "keywords": ["主人", "称呼", "约定"],
            "confidence": .9,
            "valid_from": int(time.time()),
            "valid_to": None,
        }], ["e-title"])
        recalled = self.store.search_context(
            "你应该叫我什么？", "self", "peer",
            namespace="char:bocchi", allowed_kinds={"relationship"})
        self.assertIn("称呼对方为“主人”", recalled)

    def test_pending_runtime_context_bridges_unextracted_tail_after_restart(self):
        now = int(time.time())
        self.store.save_runtime_message(
            "done-user", "self", "peer", "peer", now - 10000, "已抽取旧问题", 0,
            mode_key="tavern", namespace="char:bocchi")
        self.store.save_runtime_message(
            "done-bot", "self", "self", "peer", now - 9999, "已抽取旧回复", 1,
            mode_key="tavern", namespace="char:bocchi")
        done = self.store.get_online_extraction_batch(
            "self", "peer", mode_key="tavern", namespace="char:bocchi")
        self.store.finish_online_extraction(
            "self", "peer", int(done[-1]["id"]),
            mode_key="tavern", namespace="char:bocchi")

        self.store.save_runtime_message(
            "pending-user", "self", "peer", "peer", now - 9000, "以后叫我主人", 0,
            mode_key="tavern", namespace="char:bocchi")
        self.store.save_runtime_message(
            "pending-bot", "self", "self", "peer", now - 8999, "好的，主人。", 1,
            mode_key="tavern", namespace="char:bocchi")
        self.store.save_runtime_message(
            "other-character", "self", "peer", "peer", now - 8998, "叫我船长", 0,
            mode_key="tavern", namespace="char:melissa")

        recalled = self.store.search_pending_runtime_context(
            "你应该叫我什么？", "self", "peer",
            mode_key="tavern", namespace="char:bocchi")
        self.assertIn("以后叫我主人", recalled)
        self.assertIn("好的，主人", recalled)
        self.assertNotIn("已抽取旧问题", recalled)
        self.assertNotIn("船长", recalled)

    def test_pending_runtime_context_ignores_unrelated_tail(self):
        now = int(time.time())
        self.store.save_runtime_message(
            "pending-weather", "self", "peer", "peer", now, "今天天气不错", 0,
            mode_key="tavern", namespace="char:bocchi")
        recalled = self.store.search_pending_runtime_context(
            "你应该叫我什么？", "self", "peer",
            mode_key="tavern", namespace="char:bocchi")
        self.assertEqual(recalled, "")

    def test_clear_session_blocks_old_context_restore_but_allows_new_messages(self):
        now = int(time.time())
        self.store.save_runtime_message(
            "old-1", "self", "peer", "peer", now, "清理前一", 0,
            mode_key="writer", namespace="work:a")
        self.store.save_runtime_message(
            "old-2", "self", "self", "peer", now, "清理前二", 1,
            mode_key="writer", namespace="work:a")
        old_batch = self.store.get_online_extraction_batch(
            "self", "peer", mode_key="writer", namespace="work:a"
        )
        old_last_rowid = int(old_batch[-1]["id"])
        self.store.clear_session("self", "peer", "work:a", mode_key="writer")
        self.assertFalse(self.store.is_online_extraction_batch_current(
            "self", "peer", old_last_rowid,
            mode_key="writer", namespace="work:a",
        ))
        self.assertEqual(self.store.load_recent_context(
            "self", "peer", mode_key="writer", namespace="work:a",
            limit=20, since=now - 100,
        ), [])

        self.store.save_runtime_message(
            "new", "self", "peer", "peer", now + 1, "清理后", 0,
            mode_key="writer", namespace="work:a")
        rows = self.store.load_recent_context(
            "self", "peer", mode_key="writer", namespace="work:a",
            limit=20, since=now - 100,
        )
        self.assertEqual([row["text"] for row in rows], ["清理后"])

    def test_clear_session_advances_cursor_instead_of_reextracting_history(self):
        self.store.save_runtime_message(
            "m1", "self", "peer", "peer", 100, "旧消息", 0,
            mode_key="writer", namespace="work:a")
        self.store.clear_session("self", "peer", "work:a", mode_key="writer")
        self.assertEqual(self.store.get_online_extraction_batch(
            "self", "peer", mode_key="writer", namespace="work:a"), [])
        self.store.save_runtime_message(
            "m2", "self", "peer", "peer", 101, "新消息", 0,
            mode_key="writer", namespace="work:a")
        batch = self.store.get_online_extraction_batch(
            "self", "peer", mode_key="writer", namespace="work:a")
        self.assertEqual([row["text"] for row in batch], ["新消息"])

    def test_clear_session_invalidates_inflight_extraction_batch(self):
        namespace = "work:a"
        self.store.save_runtime_message(
            "m1", "self", "peer", "peer", 100, "旧消息", 0,
            mode_key="writer", namespace=namespace,
        )
        rows = self.store.get_online_extraction_batch(
            "self", "peer", mode_key="writer", namespace=namespace
        )
        self.assertEqual(len(rows), 1)
        last_rowid = int(rows[-1]["id"])
        self.assertTrue(self.store.is_online_extraction_batch_current(
            "self", "peer", last_rowid, mode_key="writer", namespace=namespace
        ))

        self.store.clear_session(
            "self", "peer", namespace, mode_key="writer"
        )
        self.assertFalse(self.store.is_online_extraction_batch_current(
            "self", "peer", last_rowid, mode_key="writer", namespace=namespace
        ))
        # 清理后的新消息形成新批次，不被旧游标误伤。
        self.store.save_runtime_message(
            "m2", "self", "peer", "peer", 101, "新消息", 0,
            mode_key="writer", namespace=namespace,
        )
        new_rows = self.store.get_online_extraction_batch(
            "self", "peer", mode_key="writer", namespace=namespace
        )
        self.assertEqual([row["text"] for row in new_rows], ["新消息"])
        self.assertTrue(self.store.is_online_extraction_batch_current(
            "self", "peer", int(new_rows[-1]["id"]),
            mode_key="writer", namespace=namespace,
        ))

    def test_mode_level_operations_are_scoped_to_current_user(self):
        now = int(time.time())
        self.store.save_memories([
            {"owner_id": "self", "kind": "person", "subject_id": "peer-a",
             "summary": "用户A", "keywords": ["A"], "confidence": .9,
             "valid_from": now, "valid_to": None},
            {"owner_id": "self", "kind": "person", "subject_id": "peer-b",
             "summary": "用户B", "keywords": ["B"], "confidence": .9,
             "valid_from": now, "valid_to": None},
        ], ["1"])
        rows = self.store.list_user("self", "peer-a")
        self.assertEqual([row["summary"] for row in rows], ["用户A"])
        self.assertEqual(self.store.clear_user("self", "peer-a", "clone"), 1)
        self.assertEqual([row["summary"] for row in self.store.list_user("self", "peer-b")],
                         ["用户B"])

    def test_readable_time(self):
        label = format_memory_time("episode", 1_700_000_000, None, 1_700_086_400)
        self.assertIn("发生", label)
        self.assertIn("1天前", label)

    def test_conflict_supersedes_writes_history(self):
        self.store.save_memories([
            {"kind": "person", "subject_id": "peer", "summary": "对方住在北京",
             "keywords": ["居住", "北京"], "confidence": .85,
             "valid_from": 100, "valid_to": None},
            {"kind": "person", "subject_id": "peer", "summary": "对方已经搬到上海",
             "keywords": ["居住", "上海"], "confidence": .9,
             "valid_from": 200, "valid_to": None},
        ], ["m1"])
        with self.store.connect() as db:
            rows = db.execute("SELECT * FROM memories ORDER BY valid_from").fetchall()
            db.execute("""INSERT INTO conflict_jobs
              (pair_key,left_memory_id,right_memory_id,status,updated_at)
              VALUES('pair',?,?, 'running',1)""", (rows[0]["id"], rows[1]["id"]))
            job = db.execute("SELECT * FROM conflict_jobs WHERE pair_key='pair'").fetchone()
            apply_result(db, job, rows[0], rows[1], {
                "relation": "supersedes", "winner": "right",
                "confidence": .95, "reason": "搬家形成新状态",
            }, .82)
        with self.store.connect() as db:
            old, new = db.execute("SELECT * FROM memories ORDER BY valid_from").fetchall()
            self.assertEqual(old["status"], "historical")
            self.assertEqual(old["valid_to"], 199)
            self.assertEqual(old["supersedes_id"], new["id"])
            self.assertEqual(new["status"], "active")

    def test_semantic_save_merges_paraphrase_and_keeps_evidence(self):
        original = {
            "kind": "person", "subject_id": "peer",
            "summary": "对方喜欢科幻小说", "keywords": ["科幻", "小说"],
            "confidence": .82, "valid_from": 100, "valid_to": None,
        }
        self.store.save_memories([original], ["m1"])
        row = self.store.memories_needing_embeddings("test")[0]
        self.store.save_embeddings("test", [
            (int(row["id"]), memory_embedding_text(row), [1.0, 0.0]),
        ])

        created, merged, memory_ids = self.store.save_memories_semantic([{
            **original,
            "summary": "对方平时爱看科幻类小说",
            "keywords": ["阅读", "科幻小说"],
            "confidence": .9,
        }], ["m2"], "test", [[.999, .001]], similarity_threshold=.9)
        self.assertEqual((created, merged), (0, 1))
        self.assertEqual(memory_ids, [int(row["id"])])
        with self.store.connect() as db:
            memories = db.execute("SELECT * FROM memories").fetchall()
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]["confidence"], .9)
        self.assertEqual(set(__import__("json").loads(memories[0]["evidence_json"])),
                         {"m1", "m2"})

    def test_online_extraction_defaults_are_stricter(self):
        extractor = OnlineMemoryExtractor(
            stores={"clone": self.store, "tavern": self.store, "writer": self.store},
            self_id="self", llm_api_base="http://127.0.0.1:11434",
            llm_model="test",
        )
        self.assertTrue(extractor.enabled)
        self.assertTrue(extractor.enabled_for("clone"))
        self.assertTrue(extractor.enabled_for("tavern"))
        self.assertTrue(extractor.enabled_for("writer"))
        self.assertEqual(extractor._thresholds_for("clone"), (16, 6))
        self.assertEqual(extractor._thresholds_for("tavern"), (8, 4))
        self.assertEqual(extractor._thresholds_for("writer"), (8, 4))
        self.assertEqual(extractor.max_items, 3)
        self.assertGreaterEqual(extractor.min_confidence, .75)

    def test_writer_person_filter_rejects_fictional_character(self):
        rows = [
            {"sender_id": "peer", "text": "继续，并生成一张图片"},
            {"sender_id": "self", "text": "莉娅拥抱了主角。"},
        ]
        item = {
            "kind": "person",
            "summary": "莉娅是一个天真纯粹的女性角色，拥有很强的学习能力。",
        }
        self.assertFalse(_writer_person_is_supported(item, rows, "peer"))

    def test_writer_person_filter_rejects_user_role_identity(self):
        rows = [{"sender_id": "peer", "text": "我希望这部小说采用第一人称"}]
        item = {
            "kind": "person",
            "summary": "用户在作品中扮演理性的男主角。",
        }
        self.assertFalse(_writer_person_is_supported(item, rows, "peer"))

    def test_writer_person_filter_accepts_explicit_writing_preference(self):
        rows = [{"sender_id": "peer", "text": "我喜欢第一人称叙事，希望保持细腻笔触"}]
        item = {
            "kind": "person",
            "summary": "用户偏好第一人称叙事，并要求小说保持细腻笔触。",
        }
        self.assertTrue(_writer_person_is_supported(item, rows, "peer"))

    def test_clone_can_be_disabled_without_disabling_other_modes(self):
        extractor = OnlineMemoryExtractor(
            stores={"clone": self.store, "tavern": self.store, "writer": self.store},
            self_id="self", llm_api_base="http://127.0.0.1:11434",
            llm_model="test", clone_enabled=False,
        )
        self.assertFalse(extractor.enabled_for("clone"))
        self.assertTrue(extractor.enabled_for("tavern"))
        self.assertTrue(extractor.enabled_for("writer"))

    def test_disabled_online_extraction_does_not_consume_messages(self):
        self.store.save_runtime_message(
            "m1", "self", "peer", "peer", 100, "值得记住的消息", 0,
            mode_key="tavern", namespace="char:test",
        )
        extractor = OnlineMemoryExtractor(
            stores={"tavern": self.store}, self_id="self",
            llm_api_base="http://127.0.0.1:11434", llm_model="test",
            enabled=False, tavern_min_messages=1, tavern_min_partner_messages=1,
        )
        result = __import__("asyncio").run(extractor.maybe_extract(
            "peer", "tavern", "char:test", self_id="self"
        ))
        self.assertEqual(result, 0)
        self.assertEqual(len(self.store.get_online_extraction_batch(
            "self", "peer", mode_key="tavern", namespace="char:test"
        )), 1)

    def test_duplicate_links_are_collapsed(self):
        self.store.save_memories([
            {"kind": "person", "subject_id": "peer", "summary": f"事实{i}",
             "keywords": ["事实"], "confidence": .8,
             "valid_from": 100 + i, "valid_to": None}
            for i in range(3)
        ], ["m1"])
        with self.store.connect() as db:
            rows = db.execute("SELECT id FROM memories ORDER BY id").fetchall()
            db.execute("UPDATE memories SET status='duplicate',supersedes_id=? WHERE id=?",
                       (rows[1]["id"], rows[0]["id"]))
            db.execute("UPDATE memories SET status='duplicate',supersedes_id=? WHERE id=?",
                       (rows[2]["id"], rows[1]["id"]))
            self.assertEqual(collapse_duplicate_links(db), 1)
        with self.store.connect() as db:
            first = db.execute("SELECT * FROM memories WHERE id=?", (rows[0]["id"],)).fetchone()
        self.assertEqual(first["supersedes_id"], rows[2]["id"])

    def test_duplicate_cycle_is_repaired_without_losing_evidence(self):
        self.store.save_memories([
            {"kind": "person", "subject_id": "peer", "summary": f"同义事实{i}",
             "keywords": [f"词{i}"], "confidence": .8 + i * .01,
             "valid_from": 100, "valid_to": None}
            for i in range(3)
        ], ["base"])
        with self.store.connect() as db:
            rows = db.execute("SELECT * FROM memories ORDER BY id").fetchall()
            db.execute("UPDATE memories SET status='duplicate',supersedes_id=id")
            db.execute("UPDATE memories SET supersedes_id=? WHERE id IN (?,?)",
                       (rows[0]["id"], rows[1]["id"], rows[2]["id"]))
            db.execute("""INSERT INTO memory_conflicts
              (left_memory_id,right_memory_id,relation,confidence,reason,winner_id,resolved_at)
              VALUES(?,?, 'duplicate',.9,'test',?,1)""",
              (rows[0]["id"], rows[1]["id"], rows[1]["id"]))
            self.assertEqual(repair_duplicate_cycles(db), 1)
        with self.store.connect() as db:
            repaired = db.execute("SELECT * FROM memories ORDER BY id").fetchall()
        active = [row for row in repaired if row["status"] == "active"]
        self.assertEqual([row["id"] for row in active], [rows[1]["id"]])
        self.assertTrue(all(
            row["supersedes_id"] == rows[1]["id"]
            for row in repaired if row["status"] == "duplicate"
        ))


    def test_clear_session_removes_only_peer_memories(self):
        now = int(time.time())
        self.store.save_memories([
            {"owner_id": "self", "kind": "person", "subject_id": "peer",
             "summary": "关于对方", "keywords": ["x"], "confidence": .9,
             "valid_from": now - 100, "valid_to": None},
            {"owner_id": "self", "kind": "person", "subject_id": "other",
             "summary": "关于他人", "keywords": ["y"], "confidence": .9,
             "valid_from": now - 100, "valid_to": None},
        ], ["1", "2"])
        self.store.save_runtime_message("m1", "self", "self", "peer", now, "hi", 0)
        self.store.save_runtime_message(
            "other-user", "self", "other", "other", now, "保留他人", 0)
        self.store.touch_interaction("self", "peer", user_at=now)
        deleted = self.store.clear_session("self", "peer")
        self.assertEqual(deleted, 1)
        with self.store.connect() as db:
            subjects = [r["subject_id"] for r in db.execute("SELECT subject_id FROM memories")]
            runtime_rows = db.execute(
                "SELECT peer_id,text FROM messages WHERE source_file='runtime'"
            ).fetchall()
            marker_count = db.execute(
                "SELECT count(*) c FROM messages WHERE source_file='context_reset'"
            ).fetchone()["c"]
        self.assertEqual(subjects, ["other"])
        self.assertEqual(
            [(row["peer_id"], row["text"]) for row in runtime_rows],
            [("other", "保留他人")],
        )
        self.assertEqual(marker_count, 1)

    def test_clear_all_removes_everything_but_messages(self):
        now = int(time.time())
        self.store.save_memories([
            {"owner_id": "self", "kind": "person", "subject_id": "peer",
             "summary": "a", "keywords": ["x"], "confidence": .9,
             "valid_from": now - 100, "valid_to": None},
        ], ["1"])
        self.store.touch_interaction("self", "peer", user_at=now)
        deleted = self.store.clear_all()
        self.assertEqual(deleted, 1)
        with self.store.connect() as db:
            mem_count = db.execute("SELECT count(*) c FROM memories").fetchone()["c"]
            state_count = db.execute("SELECT count(*) c FROM interaction_state").fetchone()["c"]
        self.assertEqual(mem_count, 0)
        self.assertEqual(state_count, 0)

    def test_namespace_isolation_keeps_entities_separate(self):
        """同一模式内，不同角色/作品的记忆应互不串味；person（关于用户）应共享。"""
        now = int(time.time())
        ns_a, ns_b = "char:bocchi", "char:melissa"
        self.store.save_memories([
            {"owner_id": "self", "kind": "person", "subject_id": "peer",
             "summary": "用户喜欢科幻", "keywords": ["科幻"], "confidence": .9,
             "valid_from": now, "valid_to": None},
            {"owner_id": "self", "kind": "self", "subject_id": f"{ns_a}:self:peer",
             "summary": "Bocchi 是害羞的吉他手", "keywords": ["bocchi"], "confidence": .9,
             "valid_from": now, "valid_to": None},
            {"owner_id": "self", "kind": "episode", "subject_id": f"{ns_a}:rel:self:peer",
             "summary": "Bocchi 和用户去了 live", "keywords": ["live"], "confidence": .9,
             "valid_from": now, "valid_to": None},
            {"owner_id": "self", "kind": "self", "subject_id": f"{ns_b}:self:peer",
             "summary": "Melissa 是傲娇社长", "keywords": ["melissa"], "confidence": .9,
             "valid_from": now, "valid_to": None},
            {"owner_id": "self", "kind": "episode", "subject_id": f"{ns_b}:rel:self:peer",
             "summary": "Melissa 和用户开了会", "keywords": ["开会"], "confidence": .9,
             "valid_from": now, "valid_to": None},
        ], ["1"])

        # 角色 A 的检索只命中 A 的设定/互动，不命中 B（用 2+ 字符词，确保词面命中）
        ctx_a = self.store.search_context("bocchi live", "self", "peer", namespace=ns_a)
        self.assertIn("Bocchi", ctx_a)
        self.assertIn("live", ctx_a)
        self.assertNotIn("Melissa", ctx_a)
        self.assertNotIn("开会", ctx_a)
        # 角色 B 同理
        ctx_b = self.store.search_context("melissa 开会", "self", "peer", namespace=ns_b)
        self.assertIn("Melissa", ctx_b)
        self.assertIn("开了会", ctx_b)
        self.assertNotIn("Bocchi", ctx_b)
        # person（关于用户本人）跨命名空间共享
        for ns in (ns_a, ns_b):
            self.assertIn("科幻", self.store.search_context("科幻", "self", "peer", namespace=ns))

    def test_clear_session_scoped_removes_only_namespace(self):
        """/清记忆 应只清当前角色/作品的记忆，保留其他命名空间与 person。"""
        now = int(time.time())
        ns_a, ns_b = "work:novel1", "work:novel2"
        self.store.save_memories([
            {"owner_id": "self", "kind": "episode", "subject_id": f"{ns_a}:rel:self:peer",
             "summary": "小说1 的剧情 X", "keywords": ["x1"], "confidence": .9,
             "valid_from": now, "valid_to": None},
            {"owner_id": "self", "kind": "episode", "subject_id": f"{ns_b}:rel:self:peer",
             "summary": "小说2 的剧情 Y", "keywords": ["y2"], "confidence": .9,
             "valid_from": now, "valid_to": None},
            {"owner_id": "self", "kind": "person", "subject_id": "peer",
             "summary": "用户的写作偏好", "keywords": ["写作"], "confidence": .9,
             "valid_from": now, "valid_to": None},
        ], ["1"])

        deleted = self.store.clear_session("self", "peer", ns_a)
        self.assertEqual(deleted, 1)  # 只删了小说1 的 episode
        self.assertNotIn("小说1", self.store.search_context("x1", "self", "peer", namespace=ns_a))
        self.assertIn("小说2", self.store.search_context("y2", "self", "peer", namespace=ns_b))
        # person 全局，不被 scoped clear 删除
        self.assertIn("用户的写作偏好",
                      self.store.search_context("写作", "self", "peer", namespace=None))

    def test_store_and_retrieve_share_same_subject_key(self):
        """「作品键一致」的唯一保证：存储（online.maybe_extract 第118行）与检索
        （store.search_context）都必须用同一个 ``_scope_subject`` 构造 subject_id。

        本测试直接走真实键构造函数，证明：
        1) 用 ``_scope_subject(ns, kind, self, peer)`` 存的记忆，能被同一 ``ns`` 的
           ``search_context`` 命中（存储键 == 检索键）；
        2) 不同 ``ns`` 用同一个函数还原出不同 subject，互不命中（跨作品隔离）。
        """
        now = int(time.time())
        ns = "work:my_novel"
        # 存储侧：与 online.py:118 完全相同的构造方式
        subject = _scope_subject(ns, "episode", "self", "peer")
        self.store.save_memories([
            {"owner_id": "self", "kind": "episode", "subject_id": subject,
             "summary": "主角在雨夜遇见了神秘人", "keywords": ["雨夜", "神秘人"],
             "confidence": .9, "valid_from": now, "valid_to": None},
        ], ["1"])
        # 检索侧：search_context 内部用 _scope_subject(ns, ...) 还原同样的 subject
        ctx = self.store.search_context("雨夜 神秘人", "self", "peer", namespace=ns)
        self.assertIn("雨夜", ctx)
        # 不同作品：相同文本，但 namespace 不同 → 同一函数还原出不同 subject → 命中不到
        ctx_other = self.store.search_context(
            "雨夜 神秘人", "self", "peer", namespace="work:other_book")
        self.assertNotIn("雨夜", ctx_other)

    def test_list_memories_scoped_by_namespace(self):
        """list_memories 只读回顾：命名空间层只列该角色/作品的记忆（+ person），不串。"""
        now = int(time.time())
        ns_a, ns_b = "char:bocchi", "char:melissa"
        self.store.save_memories([
            {"owner_id": "self", "kind": "self", "subject_id": f"{ns_a}:self:peer",
             "summary": "Bocchi 害羞", "keywords": ["bocchi"], "confidence": .9,
             "valid_from": now, "valid_to": None},
            {"owner_id": "self", "kind": "episode", "subject_id": f"{ns_a}:rel:self:peer",
             "summary": "Bocchi 去了 live", "keywords": ["live"], "confidence": .9,
             "valid_from": now, "valid_to": None},
            {"owner_id": "self", "kind": "self", "subject_id": f"{ns_b}:self:peer",
             "summary": "Melissa 傲娇", "keywords": ["melissa"], "confidence": .9,
             "valid_from": now, "valid_to": None},
            {"owner_id": "self", "kind": "person", "subject_id": "peer",
             "summary": "用户喜欢科幻", "keywords": ["科幻"], "confidence": .9,
             "valid_from": now, "valid_to": None},
        ], ["1"])
        rows_a = self.store.list_memories("self", "peer", ns_a)
        summaries = {r["summary"] for r in rows_a}
        self.assertEqual(summaries, {"Bocchi 害羞", "Bocchi 去了 live", "用户喜欢科幻"})
        self.assertNotIn("Melissa 傲娇", summaries)  # 另一角色不串
        # 无命名空间时按本用户旧格式（peer_id + relationship），命中 person
        rows_none = self.store.list_memories("self", "peer", None)
        self.assertIn("用户喜欢科幻", {r["summary"] for r in rows_none})

    def test_list_all_returns_everything_in_mode(self):
        """list_all（模式层回顾）返回本模式库全部记忆。"""
        now = int(time.time())
        self.store.save_memories([
            {"owner_id": "self", "kind": "self", "subject_id": "char:bocchi:self:peer",
             "summary": "Bocchi 害羞", "keywords": ["bocchi"], "confidence": .9,
             "valid_from": now, "valid_to": None},
            {"owner_id": "self", "kind": "self", "subject_id": "char:melissa:self:peer",
             "summary": "Melissa 傲娇", "keywords": ["melissa"], "confidence": .9,
             "valid_from": now, "valid_to": None},
        ], ["1"])
        all_rows = self.store.list_all()
        self.assertEqual({r["summary"] for r in all_rows},
                         {"Bocchi 害羞", "Melissa 傲娇"})


if __name__ == "__main__":
    unittest.main()
