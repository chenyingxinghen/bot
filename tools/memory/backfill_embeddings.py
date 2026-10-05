"""为已有记忆批量生成向量；可重复运行，只处理缺失或内容变化的条目。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

BOT_DIR = Path(__file__).resolve().parents[2]

from core.memory.embedding import OllamaEmbeddingClient
from core.memory.settings import EmbeddingSettings
from core.memory.store import MemoryStore, memory_embedding_text


def main() -> None:
    load_dotenv(BOT_DIR / ".env")
    defaults = EmbeddingSettings.from_environment()
    parser = argparse.ArgumentParser(description="使用 Ollama 为记忆批量回填 embedding")
    parser.add_argument("--db", type=Path, default=BOT_DIR / "data" / "memory.db")
    parser.add_argument("--api-base", default=defaults.api_base)
    parser.add_argument("--api-key", default=defaults.api_key)
    parser.add_argument("--model", default=defaults.model)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=defaults.timeout)
    parser.add_argument("--num-gpu", type=int, default=defaults.num_gpu,
                        help="embedding 使用的 GPU 层数；默认 0，即完全在 CPU 上运行")
    parser.add_argument("--keep-alive", default=defaults.keep_alive)
    args = parser.parse_args()

    store = MemoryStore(args.db)
    pending = store.memories_needing_embeddings(args.model, args.limit)
    if not pending:
        print(f"无需回填：模型 {args.model} 的向量均为最新")
        return

    client = OllamaEmbeddingClient(args.api_base, args.model, args.timeout,
                                   args.api_key, args.num_gpu, args.keep_alive)
    saved = 0
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        texts = [memory_embedding_text(row) for row in batch]
        embeddings = client.embed(texts)
        saved += store.save_embeddings(args.model, [
            (int(row["id"]), text, embedding)
            for row, text, embedding in zip(batch, texts, embeddings)
        ])
        print(f"已写入 {saved}/{len(pending)}")
    print(f"回填完成：模型 {args.model}，共写入 {saved} 条")


if __name__ == "__main__":
    main()
