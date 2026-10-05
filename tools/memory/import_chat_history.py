import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

BOT_DIR = Path(__file__).resolve().parents[2]

from core.memory.store import MemoryStore


def main():
    parser = argparse.ArgumentParser(description="导入 QQ 私聊历史到长期记忆库")
    parser.add_argument("history", type=Path, help="导出的历史 JSON 文件路径")
    parser.add_argument("--self-id", required=True, help="该聊天历史所属的本人 QQ")
    parser.add_argument("--peer", required=True, help="交谈对象 QQ")
    parser.add_argument("--db", type=Path, default=BOT_DIR / "data" / "memory.db")
    args = parser.parse_args()
    count = MemoryStore(args.db).import_history(
        args.history, args.peer, str(args.self_id))
    print(f"导入完成：新增 {count} 条消息，数据库 {args.db.resolve()}")


if __name__ == "__main__":
    main()
