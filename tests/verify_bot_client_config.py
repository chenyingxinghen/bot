"""用 bot 自己的 ComfyUIClient + 修复后的 .env.prod 配置，验证生图代码路径。"""
import asyncio, os, sys
from pathlib import Path
from types import SimpleNamespace

BOT_ROOT = Path("F:/NapCat.Shell/bot")
sys.path.insert(0, str(BOT_ROOT))

# 解析 .env.prod 成 config 命名空间（属性名 = 小写 env key）
cfg = {}
for line in (BOT_ROOT / ".env.prod").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, _, v = line.partition("=")
    cfg[k.strip().lower()] = v.strip()
config = SimpleNamespace(**cfg)

from core.image.comfyui import ComfyUISettings, ComfyUIClient

settings = ComfyUISettings.from_config(config, BOT_ROOT)
print("workflow_path:", settings.workflow_path, "exists=", settings.workflow_path.exists())
print("positive_node/key:", settings.positive_node, settings.positive_key)
print("negative_node/key:", settings.negative_node, settings.negative_key)
print("seed_node/key:", settings.seed_node, settings.seed_key)
print("size_node/w/h keys:", settings.size_node, settings.width_key, settings.height_key)
print("output_node:", settings.output_node)
print("ckpt_node/name:", settings.ckpt_node, settings.ckpt_name)

client = ComfyUIClient(settings)

async def main():
    paths = await client.generate("a fox girl with red hair, masterpiece, anime style",
                                  seed=20260801, width=512, height=768)
    print("RETURNED PATHS:", paths)
    ok = False
    for p in paths:
        data = open(p, "rb").read()
        ok = data[:8] == b"\x89PNG\r\n\x1a\n"
        print("  file:", p, "bytes=", len(data), "PNG=", ok)
    print("RESULT:", "PASS" if (paths and ok) else "FAIL")

asyncio.run(main())
