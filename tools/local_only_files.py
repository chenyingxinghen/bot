"""管理「云端保留模板记录、本地自由修改」的 skip-worktree 清单。

背景：`data/` 下的提示词模板既想留一份在云端作参考，又希望本地怎么改都不被推送。
`skip-worktree` 正好满足：文件继续被跟踪（云端保留 HEAD 版本），但本地改动不进
`git status`、不会被 `git add -A` 打包、不会被 push。

注意：`skip-worktree` 是**本地索引状态，不随 push/clone 传播**。换机器或重新
clone 后需要重跑本脚本：

    python -m tools.local_only_files apply     # 标记为本地自由修改
    python -m tools.local_only_files status    # 查看当前标记状态
    python -m tools.local_only_files release data/modes/writer.txt
                                           # 取消单个文件的标记，恢复跟踪
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BOT_DIR = Path(__file__).resolve().parents[1]

# 云端保留一份作模板参考，但本地改动一律不推送。
LOCAL_ONLY = (
    "data/modes/clone.txt",
    "data/modes/tavern.txt",
    "data/modes/writer.txt",
    "data/modes_state.json",
    "data/t2i.json",
    "data/tavern_characters.json",
)


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=BOT_DIR, capture_output=True, text=True, check=True
    )


def _tracked(paths: tuple[str, ...]) -> list[str]:
    """过滤掉尚未被跟踪的路径（未跟踪文件本来就不会被推送，无需标记）。"""
    out = _git("ls-files", "--", *paths).stdout.split()
    return [p for p in paths if p in out]


def apply() -> None:
    targets = _tracked(LOCAL_ONLY)
    missing = sorted(set(LOCAL_ONLY) - set(targets))
    if targets:
        _git("update-index", "--skip-worktree", *targets)
        print(f"已标记为本地自由修改（{len(targets)} 个）：")
        for path in targets:
            print(f"  {path}")
    if missing:
        print("\n未被跟踪、无需标记：")
        for path in missing:
            print(f"  {path}")
    print(
        "\n提示：本地改动不会再进入 git status / push。"
        "如需把某个文件的改动推上去，先 release 再提交。"
    )


def status() -> None:
    lines = _git("ls-files", "-v").stdout.splitlines()
    # 输出形如 "S data/modes/writer.txt"：首字符是标记位，空格后是路径。
    # 中文路径可能被双引号包裹（core.quotepath），去掉引号才能匹配上。
    flags = {}
    for line in lines:
        if not line:
            continue
        path = line[1:].strip().strip('"')
        flags[path] = line[0]
    print(f"{'标记':<6}{'路径'}")
    for path in LOCAL_ONLY:
        flag = flags.get(path, " ")
        mark = "skip" if flag == "S" else ("assume" if flag == "h" else "-")
        print(f"{mark:<8}{path}")


def release(paths: list[str]) -> None:
    targets = _tracked(tuple(paths))
    if not targets:
        print("指定路径均未被跟踪，无需取消标记。")
        return
    _git("update-index", "--no-skip-worktree", *targets)
    for path in targets:
        print(f"已恢复跟踪：{path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("apply", help="按清单设置 skip-worktree")
    sub.add_parser("status", help="查看当前标记状态")
    rel = sub.add_parser("release", help="取消指定文件的标记")
    rel.add_argument("paths", nargs="+")

    args = parser.parse_args()
    if args.cmd == "apply":
        apply()
    elif args.cmd == "status":
        status()
    else:
        release(args.paths)


if __name__ == "__main__":
    main()
