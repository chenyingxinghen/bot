"""临时验证：把当前 data/t2i.json 直接发到 ComfyUI，
确认工作流能正常排队、执行、产出图片，并把图片拉回本地校验。
用 `python -u` 运行以实时看到输出。"""
import json
import sys
import time
import urllib.request
import urllib.error
import os

COMFY = "http://127.0.0.1:8188"
WF_PATH = "data/t2i.json"
CLIENT_ID = "verify_wf_%d" % int(time.time())
OUT_LOCAL = "data/generated/verify_comfyui_result.png"


def post_json(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        COMFY + path, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def get_bytes(path):
    req = urllib.request.Request(COMFY + path)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main():
    wf = json.load(open(WF_PATH, encoding="utf-8"))
    wf["67"]["inputs"]["text"] = (
        "a cute white cat sitting on a windowsill, anime style, masterpiece"
    )
    wf["71"]["inputs"]["text"] = "low quality, worst quality"
    wf["69"]["inputs"]["seed"] = 777

    status, body = post_json("/prompt", {"prompt": wf, "client_id": CLIENT_ID})
    if status != 200:
        print(f"FAIL: /prompt 返回 {status}: {body}", flush=True)
        return 1
    prompt_id = body.get("prompt_id")
    print(f"OK: 已提交 prompt_id={prompt_id}", flush=True)

    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        s, hist = post_json(f"/history/{prompt_id}", {})
        if s == 200 and hist.get(prompt_id):
            entry = hist[prompt_id]
            outputs = entry.get("outputs", {})
            imgs = []
            for node, out in outputs.items():
                for im in out.get("images", []):
                    imgs.append(im)
            st = entry.get("status", {}).get("status_str")
            print(f"COMPLETE: status={st}, images={imgs}", flush=True)
            if not imgs:
                print("RESULT: FAIL —— 完成但无图片", flush=True)
                return 1
            # 拉回图片并校验 PNG
            im = imgs[0]
            q = f"/view?filename={urllib.parse.quote(im['filename'])}&subfolder={urllib.parse.quote(im.get('subfolder',''))}&type={urllib.parse.quote(im.get('type','output'))}"
            raw = get_bytes(q)
            os.makedirs(os.path.dirname(OUT_LOCAL), exist_ok=True)
            with open(OUT_LOCAL, "wb") as f:
                f.write(raw)
            is_png = raw[:8] == b"\x89PNG\r\n\x1a\n"
            print(f"DOWNLOAD: {len(raw)} bytes, PNG头有效={is_png}, 存至 {OUT_LOCAL}", flush=True)
            print("RESULT: PASS —— 工作流成功出图" if is_png else "RESULT: FAIL —— 非有效PNG", flush=True)
            return 0 if is_png else 1
        time.sleep(3)
    print("RESULT: FAIL —— 超时未完成", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
