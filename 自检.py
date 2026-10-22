# -*- coding: utf-8 -*-
"""逐环节自检：定位到底哪一步出问题"""

import os
import sys
import json
import time
import math
import base64
import urllib.request
import urllib.error

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ASR_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
MT_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
SR = 16000


def conf_key():
    import re
    KEY_RE = re.compile(r"sk-[A-Za-z0-9._\-]{12,}")

    def valid(k):
        return bool(k) and all(ord(c) < 128 for c in k) and bool(KEY_RE.fullmatch(k))

    p = os.path.join(APP_DIR, "config.json")
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f:
                k = json.load(f).get("key", "")
            if valid(k):
                return k
        except Exception:
            pass

    # 回退到 .env
    e = os.path.join(APP_DIR, ".env")
    if os.path.exists(e):
        for line in open(e, encoding="utf-8"):
            line = line.strip()
            if line.startswith("DASHSCOPE_API_KEY"):
                k = line.partition("=")[2].strip().strip('"').strip("'")
                if valid(k):
                    return k
    return ""


def step(n, title):
    print("\n" + "─" * 58)
    print("[%d] %s" % (n, title))
    print("─" * 58)


def post(url, payload, key, timeout=45):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "ignore")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "ignore")
    except Exception as e:
        return -1, str(e)


def wav_data_url(mono):
    pcm = np.clip(mono, -1.0, 1.0)
    raw = (pcm * 32767).astype("<i2").tobytes()
    b = bytearray()
    b += b"RIFF" + (36 + len(raw)).to_bytes(4, "little") + b"WAVEfmt "
    b += (16).to_bytes(4, "little") + (1).to_bytes(2, "little") + (1).to_bytes(2, "little")
    b += SR.to_bytes(4, "little") + (SR * 2).to_bytes(4, "little")
    b += (2).to_bytes(2, "little") + (16).to_bytes(2, "little")
    b += b"data" + len(raw).to_bytes(4, "little") + raw
    return "data:audio/wav;base64," + base64.b64encode(bytes(b)).decode()


ok_all = True

# ---------- 1. Key ----------
step(1, "API Key")
key = conf_key()
if not key:
    print("❌ config.json 里没有 key")
    ok_all = False
else:
    print("✅ 读到 key：长度 %d，前缀 %s…" % (len(key), key[:8]))

if not ok_all:
    sys.exit(1)

# ---------- 2. 音频设备 ----------
step(2, "系统音频捕获")
try:
    import soundcard as sc
    spk = sc.default_speaker()
    print("默认输出设备：%s" % spk.name)
    loop = sc.get_microphone(spk.name, include_loopback=True)
    print("回环设备：%s" % loop.name)
    with loop.recorder(samplerate=SR, channels=1) as rec:
        chunks = [rec.record(numframes=1600) for _ in range(6)]
    data = np.concatenate(chunks)[:, 0]
    rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
    db = 20 * math.log10(rms + 1e-9)
    print("✅ 录到 %.1f 秒，音量 %.1f dB %s" %
          (len(data) / SR, db, "（当前没有声音在放，正常）" if db < -60 else ""))
    if len(data) == 0:
        print("❌ 录不到数据")
        ok_all = False
except Exception as e:
    print("❌ 音频失败：%s" % e)
    ok_all = False

# ---------- 3. ASR 接口 ----------
step(3, "语音识别接口")
# 用 1 秒静音测接口连通性（识别结果为空是正常的，只要 HTTP 是 200）
silence = np.zeros(SR, dtype=np.float32)
code, body = post(ASR_URL, {
    "model": "qwen3-asr-flash",
    "input": {"messages": [{"role": "user", "content": [{"audio": wav_data_url(silence)}]}]},
}, key)
if code == 200:
    print("✅ HTTP 200，接口正常")
    print("   返回：%s" % body[:200].replace("\n", " "))
else:
    print("❌ HTTP %s" % code)
    print("   %s" % body[:400].replace("\n", " "))
    ok_all = False

# ---------- 4. 翻译接口 ----------
step(4, "翻译接口")
code, body = post(MT_URL, {
    "model": "qwen-mt-flash",
    "messages": [{"role": "user", "content": "Hello, this is a test."}],
    "temperature": 0.2, "max_tokens": 50,
    "translation_options": {"source_lang": "auto", "target_lang": "Chinese"},
}, key)
if code == 200:
    try:
        txt = json.loads(body)["choices"][0]["message"]["content"]
        print("✅ 译文：%s" % txt.strip())
    except Exception:
        print("✅ HTTP 200，但解析异常：%s" % body[:200])
else:
    print("❌ HTTP %s" % code)
    print("   %s" % body[:400].replace("\n", " "))
    ok_all = False

# ---------- 5. 端到端 ----------
step(5, "端到端（合成一段语音再识别）")
try:
    import soundcard  # noqa
    code, body = post(
        "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
        {"model": "qwen3-tts-flash", "input": {"text": "这是一段自检语音。", "voice": "Cherry"}},
        key, timeout=60)
    if code != 200:
        print("⚠️ TTS 不可用（不影响主流程）：HTTP %s" % code)
    else:
        url = json.loads(body)["output"]["audio"]["url"]
        t0 = time.time()
        code2, body2 = post(ASR_URL, {
            "model": "qwen3-asr-flash",
            "input": {"messages": [{"role": "user", "content": [{"audio": url}]}]},
        }, key)
        cost = time.time() - t0
        if code2 == 200:
            try:
                c = json.loads(body2)["output"]["choices"][0]["message"]["content"]
                txt = c[0].get("text") if isinstance(c, list) else c
                print("✅ 识别：%s  (%.2fs)" % (txt, cost))
            except Exception:
                print("⚠️ 解析异常：%s" % body2[:200])
        else:
            print("❌ 识别失败 HTTP %s %s" % (code2, body2[:200]))
            ok_all = False
except Exception as e:
    print("⚠️ 跳过：%s" % e)

print("\n" + "=" * 58)
print("自检结果：%s" % ("全部通过" if ok_all else "有环节失败，见上面 ❌"))
print("=" * 58)
