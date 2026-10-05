"""验证 bot 当前配置的 T2I 工作流能否出图。"""
import json, urllib.request, urllib.parse, os, sys, time

COMFY = "http://127.0.0.1:8188"
WF = "data/t2i.json"
# 正确映射（依据 t2i.json 实际结构）
POS, NEG, SEED, SIZE, OUT = "67", "71", "69", "68", "77"

wf = json.load(open(WF, encoding="utf-8"))
prompt = "a cute cat sitting by a window, anime style"
wf[POS]["inputs"]["text"] = prompt
wf[NEG]["inputs"]["text"] = "low quality, worst quality"
wf[SEED]["inputs"]["seed"] = 123456789
wf[SIZE]["inputs"]["width"] = 512
wf[SIZE]["inputs"]["height"] = 768

payload = {"prompt": wf, "client_id": "verify_cfg"}
req = urllib.request.Request(COMFY + "/prompt",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"})
try:
    r = urllib.request.urlopen(req, timeout=60)
    data = json.loads(r.read())
    pid = data.get("prompt_id")
    print("SUBMIT OK prompt_id=", pid)
except urllib.error.HTTPError as e:
    print("SUBMIT FAILED", e.code, e.read().decode())
    sys.exit(1)

# 轮询
deadline = time.time() + 400
while time.time() < deadline:
    h = json.loads(urllib.request.urlopen(COMFY + "/history/" + pid, timeout=30).read())
    if pid in h:
        e = h[pid]
        st = e.get("status", {}).get("status_str")
        if st == "error":
            print("ERROR", e.get("status", {}).get("messages"))
            sys.exit(1)
        outs = e.get("outputs", {})
        imgs = outs.get(OUT, {}).get("images", [])
        if imgs:
            im = imgs[0]
            q = "/view?filename=%s&subfolder=%s&type=%s" % (
                urllib.parse.quote(im["filename"]),
                urllib.parse.quote(im.get("subfolder", "")),
                urllib.parse.quote(im.get("type", "output")))
            raw = urllib.request.urlopen(COMFY + q, timeout=60).read()
            os.makedirs("data/generated", exist_ok=True)
            p = "data/generated/verify_configured_workflow.png"
            open(p, "wb").write(raw)
            ok = raw[:8] == b"\x89PNG\r\n\x1a\n"
            print("SUCCESS bytes=%d PNG=%s path=%s" % (len(raw), ok, p))
            print("RESULT: PASS" if ok else "RESULT: FAIL")
            sys.exit(0)
    time.sleep(3)
print("TIMEOUT")
sys.exit(1)
