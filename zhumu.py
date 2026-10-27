# -*- coding: utf-8 -*-
"""
字幕助手 · 桌面版 v2
抓系统声音 → 语音识别 → 悬浮字幕（透明背景）

依赖：numpy、soundcard（装在 C:\\Python314）
"""

import os
import re
import sys
import json
import time
import math
import queue
import threading
import base64
import uuid
import asyncio
import ctypes
import urllib.request
import urllib.error
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox

import numpy as np
import aiohttp
import warnings
import soundcard as sc

# soundcard 的 MediaFoundation 后端在 Windows 上偶发「data discontinuity」，
# 是缓冲区瞬时不同步的无害警告，音频流本身是通的，抑制掉免得刷屏
warnings.filterwarnings("ignore", message=".*data discontinuity.*")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CONF_PATH = os.path.join(APP_DIR, "config.json")
LOG_PATH = os.path.join(APP_DIR, "session.log")

SR = 16000
ASR_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
ASR_WS = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
ASR_MODEL = "fun-asr-realtime"  # 30 语种+16 方言，比 paraformer 新、延迟更低（实测 0.55s）
MT_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

FONT = "Microsoft YaHei"

# 语言码：界面中文名 -> ASR 提示码 / 翻译目标名
# fun-asr-realtime 支持 30 语种 + 16 方言；es/pt/it 已实测可用（paraformer 不行）
# 延迟徽标配色：(底色, 字色)
LAT_THEMES = {
    "深灰": ("#23272E", "#E6E8EB"),
    "深蓝": ("#16202E", "#8FC6FF"),
    "纯黑": ("#0A0A0A", "#F0F0F0"),
    "暖灰": ("#2B2723", "#EDE4D8"),
    "米白": ("#F2EFE8", "#242424"),
}
LAT_THEME_DEFAULT = "深灰"

SRC_LANGS = [("auto", "自动"), ("zh", "中文"), ("en", "英语"), ("ja", "日语"),
             ("ko", "韩语"), ("ru", "俄语"), ("fr", "法语"), ("de", "德语"),
             ("es", "西班牙语"), ("pt", "葡萄牙语"), ("it", "意大利语")]
# 目标语言可以比源语言多：翻译模型支持 92 种，识别不支持也能翻过去
DST_LANGS = [("zh", "中文"), ("en", "英语"), ("ja", "日语"), ("ko", "韩语"),
             ("ru", "俄语"), ("fr", "法语"), ("de", "德语"),
             ("es", "西班牙语"), ("pt", "葡萄牙语"), ("it", "意大利语")]

# 目标语言传给翻译模型的英文名
DST_EN = {"zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean",
          "ru": "Russian", "fr": "French", "de": "German",
          "es": "Spanish", "pt": "Portuguese", "it": "Italian"}
TRANSPARENT_KEY = "#0A0A0B"   # 用作透明键，正文里不会用到这个颜色

PRESETS = {
    "outline": {"text": "#FFFFFF", "outline": "#000000"},
    "invert":  {"text": "#0A0A0A", "outline": "#FFFFFF"},
    "yellow":  {"text": "#FFD84D", "outline": "#000000"},
}


def log(msg):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write("%s  %s\n" % (time.strftime("%H:%M:%S"), msg))
    except Exception:
        pass


log("=" * 46)
log("boot: imports ok")


# ============================ 配置 ============================
KEY_RE = re.compile(r"sk-[A-Za-z0-9._\-]{12,}")


def is_valid_key(k):
    if not k:
        return False
    k = k.strip()
    if any(ord(c) > 127 for c in k):
        return False
    return bool(KEY_RE.fullmatch(k))


def key_from_env():
    p = os.path.join(APP_DIR, ".env")
    if not os.path.exists(p):
        return ""
    try:
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line.startswith("DASHSCOPE_API_KEY"):
                return line.partition("=")[2].strip().strip('"').strip("'")
    except Exception:
        pass
    return ""


def mask_key(k):
    if not k:
        return ""
    return (k[:7] + "…" + k[-4:]) if len(k) > 14 else k[:7] + "…"


DEFAULT_CONF = {
    "keys": [],            # [{"name": ..., "key": ...}]
    "active": 0,
    "rate": 1,
    "preset": "outline",
    "bg_mode": "transparent",
    "alpha": 0.62,
    "x": 200, "y": 760, "w": 1000, "h": 200,
    "ratio": 0.16,
    "auto_width": False,
    "idle_hide": 5.0,
    "min_show": 2.5,
    "punct_batch": True,
    "show_latency": False,
    "latency_theme": "深灰",
    "audio_device": "",
    "show_source": False,
    "src_lang": "auto",
    "dst_lang": "zh",
    "asr_engine": "cloud",
    "local_model_dir": "D:/sherpa-models/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05",
    "local_punct_dir": "D:/sherpa-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8",
    "show_sub": True,
}


def load_conf():
    d = dict(DEFAULT_CONF)
    try:
        if os.path.exists(CONF_PATH):
            with open(CONF_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            d.update(raw)
            # 旧版单 key 迁移
            old = raw.get("key")
            if old and not d.get("keys"):
                if is_valid_key(old):
                    d["keys"] = [{"name": "默认", "key": old}]
                else:
                    log("丢弃无效旧 key（前 20 字：%s）" % str(old)[:20])
    except Exception:
        pass

    # 清理无效条目
    d["keys"] = [k for k in d.get("keys", []) if is_valid_key(k.get("key", ""))]

    # 一个都没有就从 .env 补
    if not d["keys"]:
        env = key_from_env()
        if is_valid_key(env):
            d["keys"] = [{"name": "默认", "key": env}]
            log("key: 从 .env 载入 %s" % mask_key(env))

    if d.get("active", 0) >= len(d["keys"]):
        d["active"] = 0

    d.pop("key", None)
    return d


def save_conf(c):
    try:
        with open(CONF_PATH, "w", encoding="utf-8") as f:
            json.dump(c, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def active_key(conf):
    ks = conf.get("keys") or []
    i = conf.get("active", 0)
    if 0 <= i < len(ks):
        return ks[i].get("key", "")
    return ""


# ============================ API ============================
def http_post(url, payload, key, timeout=60):
    req = urllib.request.Request(
        url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


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


def do_asr(data_url, key):
    obj = http_post(ASR_URL, {
        "model": "qwen3-asr-flash",
        "input": {"messages": [{"role": "user", "content": [{"audio": data_url}]}]},
    }, key)
    try:
        c = obj["output"]["choices"][0]["message"]["content"]
        return (c[0].get("text") if isinstance(c, list) else str(c)).strip()
    except Exception:
        return ""


def mostly_chinese(s):
    # 有假名就肯定不是中文（日语里汉字多，光看汉字比例会误判）
    if any("\u3040" <= ch <= "\u30ff" for ch in s):
        return False
    han = sum(1 for ch in s if "\u4e00" <= ch <= "\u9fa5")
    n = len(s.replace(" ", ""))
    return n > 0 and han / n > 0.5


def do_translate(text, key, rate, dst="zh"):
    if dst == "zh" and mostly_chinese(text):
        return text
    if rate >= 1.5:
        level = ("极致压缩：用最少的字表达，去掉一切修辞和连接词，但必须保留全部信息点"
                 if rate >= 2 else
                 "精简：删掉口头语、语气词、重复表述，长句拆短，但不省略任何信息点")
        prompt = "把下面这段字幕翻译成简体中文并" + level + "。只输出译文。\n\n" + text
        obj = http_post(MT_URL, {
            "model": "qwen-plus",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2, "max_tokens": 300}, key)
        return obj["choices"][0]["message"]["content"].strip()

    obj = http_post(MT_URL, {
        "model": "qwen-mt-flash",
        "messages": [{"role": "user", "content": text}],
        "temperature": 0.2, "max_tokens": 300,
        "translation_options": {"source_lang": "auto",
                                "target_lang": DST_EN.get(dst, "Chinese")}}, key)
    return obj["choices"][0]["message"]["content"].strip()


# ============================ 断句 ============================
SPLIT_RE = re.compile(r"(?<=[。！？!?；;])")


def split_sentences(text):
    """按句末标点切成独立的句子。ASR 自带标点，用它做视觉上的逐句推进"""
    if not text:
        return []
    parts = [p.strip() for p in SPLIT_RE.split(text)]
    parts = [p for p in parts if p]
    # 太碎的合并（比如只有一个标点的片段）
    out = []
    for p in parts:
        if out and len(re.sub(r"[。！？!?；;，,]", "", p)) <= 1:
            out[-1] += p
        else:
            out.append(p)
    return out or [text]


class AsrStream(threading.Thread):
    """流式识别：边说话边出字，首字延迟约 0.8 秒（实测）"""

    def __init__(self, on_partial, on_final, on_error, lang_hint=None):
        super().__init__(daemon=True)
        self.lang_hint = lang_hint
        self.on_partial = on_partial
        self.on_final = on_final
        self.on_error = on_error
        self.q = queue.Queue(maxsize=200)
        self.running = True
        self.first_token_at = None

    def feed(self, pcm_bytes):
        try:
            self.q.put_nowait(pcm_bytes)
        except queue.Full:
            pass

    def run(self):
        try:
            asyncio.run(self._main())
        except Exception as e:
            self.on_error("流式连接异常：%s" % str(e)[:120])

    async def _main(self):
        task_id = uuid.uuid4().hex
        headers = {"Authorization": "Bearer " + active_key_holder.get("k", "")}
        async with aiohttp.ClientSession() as session:
            ws = await session.ws_connect(ASR_WS, headers=headers, heartbeat=20)
            await ws.send_str(json.dumps({
                "header": {"action": "run-task", "task_id": task_id, "streaming": "duplex"},
                "payload": {
                    "task_group": "audio", "task": "asr", "function": "recognition",
                    "model": ASR_MODEL,
                    "parameters": dict(
                        {"format": "pcm", "sample_rate": SR},
                        **({"language_hints": [self.lang_hint]} if self.lang_hint else {})
                    ),
                    "input": {},
                }}))

            started = False
            while self.running and not started:
                msg = await asyncio.wait_for(ws.receive(), timeout=10)
                if msg.type == aiohttp.WSMsgType.TEXT:
                    obj = json.loads(msg.data)
                    ev = obj.get("header", {}).get("event", "")
                    if ev == "task-started":
                        started = True
                    elif ev == "task-failed":
                        raise RuntimeError(json.dumps(obj.get("header", {}), ensure_ascii=False)[:160])
            if not started:
                raise RuntimeError("没收到 task-started")

            async def sender():
                while self.running:
                    try:
                        data = self.q.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(0.02)
                        continue
                    await ws.send_bytes(data)

            async def receiver():
                cur_begin = None
                async for msg in ws:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    obj = json.loads(msg.data)
                    ev = obj.get("header", {}).get("event", "")
                    if ev == "task-finished":
                        break
                    if ev != "result-generated":
                        continue
                    out = obj.get("payload", {}).get("output", {}) or {}
                    sent = out.get("sentence") or {}
                    text = (sent.get("text") or "").strip()
                    if not text:
                        continue
                    # 用词级时间戳找语音停顿（相邻词间隔 > 400ms），
                    # 在停顿处插切分标记 \u0001 —— 说话人「顿一下」就是天然句边界
                    words = sent.get("words") or []
                    if words:
                        marked = []
                        for wi, w in enumerate(words):
                            if wi > 0:
                                pe = words[wi - 1].get("end_time")
                                cb = w.get("begin_time")
                                if (pe is not None and cb is not None
                                        and cb - pe > 400):
                                    marked.append("\u0001")
                            marked.append((w.get("text") or "") +
                                          (w.get("punctuation") or ""))
                        text = "".join(marked).strip("\u0001")
                        if not text:
                            continue
                    begin = sent.get("begin_time")
                    if cur_begin is None:
                        cur_begin = begin
                    elif begin != cur_begin:
                        cur_begin = begin
                    self.on_partial(text, sent.get("sentence_end", False),
                                    sent.get("begin_time"))

            await asyncio.gather(sender(), receiver())
            try:
                await ws.send_str(json.dumps({
                    "header": {"action": "finish-task", "task_id": task_id},
                    "payload": {"input": {}}}))
                await ws.close()
            except Exception:
                pass


active_key_holder = {"k": ""}


def split_units(text):
    """把长句切成「翻译单元」。

    切分优先级（高→低）：
      1. 语音停顿（\\u0001 标记，相邻词间隔 > 400ms）
      2. 标点（句末 。！？!? 与逗号类 ，、,;；）—— 标点处必切，不再累积
      3. 12 字兜底（只用于完全没有标点的长段）
    标点切出来的短句不合并（它们是完整语义单元）。
    """
    if not text:
        return []
    out = []
    # 1) 先按语音停顿切
    for seg in [x.strip() for x in text.split("\u0001") if x.strip()]:
        # 2) 再按标点切 —— 标点处必切（优先级高于字数兜底）
        for p in [x.strip() for x in re.split(r"(?<=[。！？!?，、,;；])", seg) if x.strip()]:
            # 3) 只有「没有标点结尾的长段」才按 12 字兜底
            while len(p) > 12 and not p.rstrip().endswith(tuple("。！？!?，、,;；")):
                out.append(p[:12])
                p = p[12:]
            if p:
                out.append(p)
    # 过短块合并：<4 字且不以标点结尾的碎片才合并（标点结尾的是完整短句）
    merged = []
    for u in out:
        prev_ok = (merged and len(merged[-1]) < 4
                   and not merged[-1].rstrip().endswith(tuple("。！？!?，、,;；")))
        if prev_ok:
            merged[-1] += u
        else:
            merged.append(u)
    return merged


def list_loopback_devices():
    try:
        return [m.name for m in sc.all_microphones(include_loopback=True)
                if getattr(m, "isloopback", False)]
    except Exception:
        return []


# ============================ 音频引擎 ============================
class LocalAsr(threading.Thread):
    """本地流式识别（sherpa-onnx + Zipformer），接口与 AsrStream 对齐。

    模型：sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct
    特点：中英双语 + 原生标点；纯 CPU；首字几百毫秒。
    """

    def __init__(self, on_partial, on_final, on_error, model_dir, punct_dir=None):
        super().__init__(daemon=True)
        self.on_partial = on_partial
        self.on_final = on_final
        self.on_error = on_error
        self.model_dir = model_dir
        self.punct_dir = punct_dir
        self._punct = None
        self.running = False
        self.q = queue.Queue(maxsize=400)
        self._last = ""

    def stop(self):
        self.running = False
        try:
            self.q.put_nowait(None)
        except Exception:
            pass

    def feed(self, pcm):
        try:
            self.q.put_nowait(pcm)
        except queue.Full:
            pass

    @staticmethod
    def _find_files(root):
        """递归找模型四件套，优先 int8"""
        enc = dec = joi = tok = None
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                fl = f.lower()
                full = os.path.join(dirpath, f)
                if fl == "tokens.txt":
                    tok = tok or full
                elif fl.endswith(".onnx"):
                    for key, cur in (("encoder", enc), ("decoder", dec), ("joiner", joi)):
                        if key in fl:
                            if cur is None or "int8" in fl:
                                if key == "encoder":
                                    enc = full
                                elif key == "decoder":
                                    dec = full
                                else:
                                    joi = full
        return enc, dec, joi, tok

    def run(self):
        try:
            import sherpa_onnx
        except Exception as e:
            self.on_error("未安装 sherpa-onnx（pip install sherpa-onnx）：%s" % e)
            return

        if not self.model_dir or not os.path.isdir(self.model_dir):
            self.on_error("本地模型目录不存在：%s" % self.model_dir)
            return

        enc, dec, joi, tok = self._find_files(self.model_dir)
        if not (enc and dec and tok):
            self.on_error("本地模型文件不全（需 encoder/decoder/tokens.txt）：%s"
                          % self.model_dir)
            return

        try:
            common = dict(
                tokens=tok, encoder=enc, decoder=dec,
                num_threads=1,                    # 小模型单线程最快（多线程反而慢）
                sample_rate=16000, feature_dim=80,
                decoding_method="greedy_search",
                enable_endpoint_detection=True,   # 端点检测：静音自动断句
                rule1_min_trailing_silence=2.4,
                rule2_min_trailing_silence=1.2,
                rule3_min_utterance_length=300,
            )
            if joi:
                # Zipformer Transducer（有 joiner）
                rec = sherpa_onnx.OnlineRecognizer.from_transducer(joiner=joi, **common)
            else:
                # Paraformer（CTC，无 joiner）—— 中英流式首选，无重复/垃圾 token
                rec = sherpa_onnx.OnlineRecognizer.from_paraformer(**common)
            stream = rec.create_stream()
        except Exception as e:
            self.on_error("本地识别初始化失败：%s" % e)
            return

        # 标点模型（可选，缺失则不加标点）
        if self.punct_dir and os.path.isdir(self.punct_dir):
            try:
                pm = None
                for f in os.listdir(self.punct_dir):
                    if f.endswith(".onnx"):
                        if pm is None or "int8" in f.lower():
                            pm = os.path.join(self.punct_dir, f)
                if pm:
                    self._punct = sherpa_onnx.OfflinePunctuation(
                        sherpa_onnx.OfflinePunctuationConfig(
                            model=sherpa_onnx.OfflinePunctuationModelConfig(
                                ct_transformer=pm, num_threads=1)))
                    log("标点模型已加载")
            except Exception as e:
                log("标点模型加载失败（将不加标点）：%s" % e)

        def with_punct(t):
            if not t:
                return t
            # 模型自带标点（如 X-ASR）时不再重复加
            if re.search(r"[。！？!?，、,;；]", t):
                return t
            if not self._punct:
                return t
            try:
                return self._punct.add_punctuation(t)
            except Exception:
                return t

        log("本地识别就绪：%s" % os.path.basename(self.model_dir.rstrip("\\/")))
        self.running = True
        while self.running:
            try:
                item = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            try:
                arr = np.frombuffer(item, dtype="<i2").astype(np.float32) / 32768.0
                stream.accept_waveform(16000, arr)
                while rec.is_ready(stream):
                    rec.decode_stream(stream)
                text = (rec.get_result(stream) or "").strip()
                if rec.is_endpoint(stream):
                    if text:
                        self.on_partial(with_punct(text), True, None)
                    rec.reset(stream)
                    self._last = ""
                elif text and text != self._last:
                    self._last = text
                    self.on_partial(with_punct(text), False, None)
            except Exception as e:
                self.on_error("本地识别出错：%s" % e)
                break
        self.running = False


class AudioEngine(threading.Thread):
    """抓系统输出，能量 VAD 切段"""

    SIL_DB = -40.0
    CUT_SIL = 0.20       # 停顿超过 200ms 就切，尽量跟着句子走
    MIN_SEG = 0.45
    MAX_SEG = 2.8        # 上限从 5.0 降到 2.8：延迟 = 段长 + 识别，段越长等得越久

    def __init__(self, on_audio, on_level, device=None):
        super().__init__(daemon=True)
        self.on_audio = on_audio
        self.on_level = on_level
        self.running = False
        self.device_name = None
        self.want_device = device

        self.buf = []
        self.buf_s = 0.0
        self.silence_s = 0.0
        self.speech_s = 0.0
        self.seg_start = 0.0     # 段开始时刻，用于算真实延迟
        self.noise = 1e-4        # 底噪估计，用来做相对判定

    def stop(self):
        self.running = False

    def run(self):
        try:
            if self.want_device:
                loop = sc.get_microphone(self.want_device, include_loopback=True)
                self.device_name = loop.name
            else:
                spk = sc.default_speaker()
                self.device_name = spk.name
                loop = sc.get_microphone(spk.name, include_loopback=True)
        except Exception as e:
            self.on_level(-999, str(e))
            return

        self.running = True
        try:
            with loop.recorder(samplerate=SR, channels=1) as rec:
                while self.running:
                    data = rec.record(numframes=int(SR * 0.1))
                    if data is None or len(data) == 0:
                        continue
                    mono = data[:, 0].astype(np.float32)
                    rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
                    db = 20 * math.log10(rms + 1e-9)
                    self.on_level(db, None)

                    self.on_audio(mono)
        except Exception as e:
            self.on_level(-998, str(e))
        finally:
            self.running = False

    def _flush(self):
        if not self.buf:
            return
        if self.speech_s < 0.22:
            self.buf = []
            self.buf_s = self.silence_s = self.speech_s = 0.0
            return
        merged = np.concatenate(self.buf)
        # 系统音量小的时候信号只有 -50dB 上下，归一化到正常电平再送识别
        peak = float(np.max(np.abs(merged))) if len(merged) else 0.0
        if peak > 1e-6:
            merged = np.clip(merged * (0.7 / peak), -1.0, 1.0)
        self.buf = []
        self.buf_s = self.silence_s = self.speech_s = 0.0
        try:
            self.on_segment(merged, self.seg_start)
        except Exception:
            pass


# ============================ 字幕浮窗 ============================
class LatencyBadge:
    """延迟徽标 —— 完全独立的窗口。

    刻意不与 SubtitleWindow 共享任何状态：位置、拖动、显隐、配色全自己管。
    动字幕窗不会影响它，动它也不会影响字幕窗。
    """

    def __init__(self, master, conf):
        self.conf = conf
        self.text = ""
        self._abs = list(conf["latency_abs"]) if conf.get("latency_abs") else None
        self._drag = None
        bg, fg = LAT_THEMES.get(conf.get("latency_theme", LAT_THEME_DEFAULT),
                                LAT_THEMES[LAT_THEME_DEFAULT])
        self.win = tk.Toplevel(master)
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        self.win.configure(bg=bg)
        self.label = tk.Label(self.win, text="", bg=bg, fg=fg, font=(FONT, -14))
        self.label.pack(padx=7, pady=2)
        for ev, fn in (("<Button-1>", self._press),
                       ("<B1-Motion>", self._motion),
                       ("<ButtonRelease-1>", self._release)):
            self.win.bind(ev, fn)
            self.label.bind(ev, fn)
        self.win.withdraw()

    # ---------------- 对外接口 ----------------
    def set_text(self, text):
        """有内容就显示，空内容就隐藏。位置与字幕窗无关。"""
        self.text = text or ""
        if not self.text:
            self.hide()
            return
        try:
            self.label.config(text=self.text)
            if not self._abs:
                # 默认位置：屏幕右上角（不参考字幕窗位置）
                self._abs = [self.win.winfo_screenwidth() - 130, 40]
            self._place()
            self.show()
        except Exception:
            pass

    def set_theme(self, name):
        bg, fg = LAT_THEMES.get(name, LAT_THEMES[LAT_THEME_DEFAULT])
        try:
            self.win.configure(bg=bg)
            self.label.configure(bg=bg, fg=fg)
        except Exception:
            pass

    def show(self):
        try:
            self.win.deiconify()
            self.win.lift()
        except Exception:
            pass

    def hide(self):
        try:
            self.win.withdraw()
        except Exception:
            pass

    # ---------------- 内部 ----------------
    def _place(self):
        sw = self.win.winfo_screenwidth()
        sh = self.win.winfo_screenheight()
        x = max(0, min(sw - 60, int(self._abs[0])))
        y = max(0, min(sh - 30, int(self._abs[1])))
        self._abs = [x, y]
        self.win.geometry("+%d+%d" % (x, y))

    def _press(self, e):
        # 捕获鼠标，否则鼠标移出小窗后拖动会被其他窗口接管
        try:
            self.win.grab_set()
        except Exception:
            pass
        self._drag = (e.x_root, e.y_root, self.win.winfo_x(), self.win.winfo_y())
        return "break"

    def _motion(self, e):
        if not self._drag:
            return "break"
        x0, y0, wx, wy = self._drag
        self._abs = [wx + (e.x_root - x0), wy + (e.y_root - y0)]
        try:
            self.win.geometry("+%d+%d" % tuple(self._abs))
        except Exception:
            pass
        return "break"

    def _release(self, e):
        if self._drag:
            self.conf["latency_abs"] = list(self._abs or [])
            save_conf(self.conf)
        self._drag = None
        try:
            self.win.grab_release()
        except Exception:
            pass
        return "break"


class SubtitleWindow:
    HANDLE = 24

    def __init__(self, master, conf, on_settings, on_hide):
        self.conf = conf
        self.on_settings = on_settings
        self.on_hide = on_hide

        self.win = tk.Toplevel(master)
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        self.win.geometry("%dx%d+%d+%d" % (conf["w"], conf["h"], conf["x"], conf["y"]))

        self.canvas = tk.Canvas(self.win, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)

        self.src_text = ""
        self.dst_text = "…"
        self.lines = []       # 已完成的句子（最多保留 2 条）
        self.active = ""      # 正在增长的那一句
        self.src_line = ""    # 当前句的原文（可选显示）
        self.show_source = conf.get("show_source", False)
        self._last_active_at = 0.0   # 最后一次收到新内容的时间
        self._line_shown_at = 0.0    # 当前这行是什么时候开始显示的
        self._pending_q = []         # 还没到最短显示时间，排队的新内容（FIFO，不丢）
        self._pending_job = None
        self._flushing = False
        self.latency = None
        self.state = "idle"

        self._drag = None
        self._resize = None
        self._q = []
        self._q_job = None
        self._fit_job = None
        self._fitting = False
        self.auto_width = conf.get("auto_width", True)
        self.adjusting = False

        self.canvas.bind("<Button-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._motion)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.canvas.bind("<Button-3>", self._menu)
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<Double-Button-1>", lambda e: self.on_hide())
        self.win.bind("<Configure>", self._on_configure)

        self._build_menu()
        self.redraw()
        # 必须等窗口真正映射后再设分层属性，否则不生效
        self.win.after(120, self.apply_style)
        # 【已撤】定时 SetWindowPos 强制置顶会让透明窗口反复重合成、导致闪烁，
        # 且对腾讯视频无效。待办见需求清单 F27。
        self.win.after(800, self._check_idle)

    # ---------- 外观 ----------
    def apply_style(self):
        """透明模式用 Tk 原生 -transparentcolor。
        注意：必须用 python.exe 启动，pythonw 下这个特性会让进程静默崩溃。"""
        mode = self.conf.get("bg_mode", "transparent")
        if mode == "transparent":
            try:
                self.win.attributes("-alpha", 1.0)
                self.win.attributes("-transparentcolor", TRANSPARENT_KEY)
            except Exception as e:
                log("transparent failed: %s" % e)
            self.canvas.configure(bg=TRANSPARENT_KEY)
        else:
            try:
                self.win.attributes("-transparentcolor", "#FE00FE")
            except Exception:
                pass
            try:
                self.win.attributes("-alpha", max(0.25, min(1.0, self.conf.get("alpha", 0.62))))
            except Exception as e:
                log("alpha failed: %s" % e)
            self.canvas.configure(bg="#000000")
        self.redraw()

    def set_alpha(self, a):
        self.conf["alpha"] = a
        save_conf(self.conf)
        self.apply_style()

    def set_bg_mode(self, mode):
        self.conf["bg_mode"] = mode
        save_conf(self.conf)
        self.apply_style()
        self.redraw()

    # ---------- 菜单 ----------
    def _build_menu(self):
        self.menu = tk.Menu(self.win, tearoff=0)
        self.menu.add_command(label="设置…", command=self.on_settings)
        self.menu.add_command(label="隐藏字幕（双击也行）", command=self.on_hide)

        self.menu.add_command(label="移到底部居中", command=self.move_bottom)
        self.menu.add_command(label="调整位置和大小", command=self.toggle_adjust)
        self.menu.add_command(label="宽度贴合文字", command=self._toggle_autowidth)

        style = tk.Menu(self.menu, tearoff=0)
        for n, label in [("outline", "白字黑边"), ("invert", "黑字白边"), ("yellow", "黄字强调")]:
            style.add_command(label=label, command=lambda x=n: self.set_preset(x))
        self.menu.add_cascade(label="字幕样式", menu=style)

        rate = tk.Menu(self.menu, tearoff=0)
        for v, label in [(1, "1.0x 完整"), (1.5, "1.5x 精简"), (2, "2.0x 压缩")]:
            rate.add_command(label=label, command=lambda x=v: self.set_rate(x))
        self.menu.add_cascade(label="倍速档位", menu=rate)

        self.menu.add_separator()
        self.menu.add_command(label="退出程序", command=self.on_quit)

    def _toggle_autowidth(self):
        self.auto_width = not self.auto_width
        self.conf["auto_width"] = self.auto_width
        save_conf(self.conf)
        if self.auto_width:
            self._schedule_fit()

    def _force_topmost(self):
        """【当前未启用】用 Win32 SetWindowPos 强制置顶。

        注意：定时调用会让透明窗口反复重合成 → 画面闪烁，且对腾讯视频无效，
        所以先停用。需要时手动调用一次即可。

        Tk 的 -topmost 对某些播放器（如腾讯视频）不够，
        它会把窗口重新压到下面；定时用系统 API 再顶一次。
        注意：播放器的「独占全屏」模式任何窗口都盖不住，需切窗口化。
        """
        try:
            if sys.platform == "win32":
                GA_ROOT, HWND_TOPMOST = 2, -1
                SWP_NOMOVE, SWP_NOSIZE, SWP_NOACTIVATE = 0x0002, 0x0001, 0x0010
                hwnd = ctypes.windll.user32.GetAncestor(self.win.winfo_id(), GA_ROOT)
                if hwnd:
                    ctypes.windll.user32.SetWindowPos(
                        hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                        SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE)
        except Exception:
            pass
    def move_bottom(self):
        """一键挪到屏幕底部居中，像播放器字幕那样"""
        try:
            sw = self.win.winfo_screenwidth()
            sh = self.win.winfo_screenheight()
            w = self.win.winfo_width()
            h = self.win.winfo_height()
            x = max(0, (sw - w) // 2)
            y = max(0, sh - h - 90)
            self.win.geometry("+%d+%d" % (x, y))
            self.conf["x"], self.conf["y"] = x, y
            save_conf(self.conf)
        except Exception as e:
            log("move bottom failed: %s" % e)

    def toggle_adjust(self):
        self.adjusting = not self.adjusting
        if self.adjusting:
            # 临时取消抠色，整块区域都能接住鼠标
            try:
                self.win.attributes("-transparentcolor", "#FE00FE")
            except Exception:
                pass
            self.win.attributes("-alpha", 0.5)
            self.canvas.configure(bg="#000000")
            self.redraw()
        else:
            self.apply_style()
            save_conf(self.conf)

    def _menu(self, e):
        try:
            self.menu.tk_popup(e.x_root, e.y_root)
        finally:
            self.menu.grab_release()

    # ---------- 交互 ----------
    def _on_configure(self, e):
        """窗口尺寸变化时防抖重绘。直接重绘会和分层设置互相触发，形成重绘风暴"""
        if getattr(self, "_cfg_job", None):
            try:
                self.win.after_cancel(self._cfg_job)
            except Exception:
                pass
        self._cfg_job = self.win.after(140, self._safe_redraw)

    def _schedule_fit(self):
        if not self.auto_width or self.adjusting:
            return
        if self._fit_job:
            try:
                self.win.after_cancel(self._fit_job)
            except Exception:
                pass
        self._fit_job = self.win.after(120, self._fit_width)

    def _fit_width(self):
        """窗口宽度贴合文字，像播放器自带字幕那样不占地方"""
        self._fit_job = None
        if self._fitting or not self.auto_width:
            return
        self._fitting = True
        try:
            h = self.win.winfo_height()
            fs = max(12, int(h * self.conf.get("ratio", 0.16)))
            fs_src = max(9, int(fs * 0.44))
            try:
                f_dst = tkfont.Font(family=FONT, size=-fs, weight="bold")
                # 必须在「所有已显示的行」里取最宽，
                # 只算正在说的那句会让上一行长句被挤成两行
                widths = [f_dst.measure(t) for t in self.lines if t]
                widths.append(f_dst.measure(self.active or ""))
                if self.show_source and self.src_line:
                    f_src = tkfont.Font(family=FONT, size=-fs_src)
                    widths.append(f_src.measure(self.src_line))
                w_max = max(widths) if widths else 0
            except Exception:
                return
            need = w_max + int(fs * 1.9)
            screen = self.win.winfo_screenwidth()
            need = max(260, min(need, int(screen * 0.86)))
            cur = self.win.winfo_width()
            if abs(need - cur) < 12:
                return
            cx = self.win.winfo_x() + cur // 2
            self.win.geometry("%dx%d+%d+%d" % (need, h, cx - need // 2, self.win.winfo_y()))
            self.conf["w"] = need
            # 让新宽度立刻生效，否则紧接着的重绘还是按旧宽度折行
            try:
                self.win.update_idletasks()
            except Exception:
                pass
        finally:
            self._fitting = False

    def _safe_redraw(self):
        self._cfg_job = None
        self.redraw()

    def _press(self, e):
        w, h = self.win.winfo_width(), self.win.winfo_height()
        if e.x >= w - self.HANDLE and e.y >= h - self.HANDLE:
            self._resize = (e.x_root, e.y_root, w, h)
        else:
            self._drag = (e.x_root - self.win.winfo_x(), e.y_root - self.win.winfo_y())

    def _motion(self, e):
        if self._resize:
            x0, y0, w0, h0 = self._resize
            nw = max(360, w0 + (e.x_root - x0))
            nh = max(90, h0 + (e.y_root - y0))
            self.win.geometry("%dx%d" % (nw, nh))
            self.conf["w"], self.conf["h"] = nw, nh
        elif self._drag:
            dx, dy = self._drag
            self.win.geometry("+%d+%d" % (e.x_root - dx, e.y_root - dy))
            self.conf["x"] = e.x_root - dx
            self.conf["y"] = e.y_root - dy

    def _release(self, e):
        if self._resize:
            # 用户手动调过尺寸，就不再多事去自动贴宽度
            self.auto_width = False
            self.conf["auto_width"] = False
        if self._drag or self._resize:
            save_conf(self.conf)
        self._drag = None
        self._resize = None
        self.redraw()

    def _wheel(self, e):
        step = 0.01 if e.delta > 0 else -0.01
        self.conf["ratio"] = max(0.06, min(0.34, self.conf.get("ratio", 0.15) + step))
        save_conf(self.conf)
        self.redraw()

    def set_preset(self, name):
        self.conf["preset"] = name
        save_conf(self.conf)
        self.redraw()

    def set_rate(self, v):
        self.conf["rate"] = v
        save_conf(self.conf)

    # ---------- 内容 ----------
    def push(self, src, dst, latency=None):
        """按句推进：一段里有多句就依次显示"""
        self.latency = latency
        self.state = "live"
        sents = split_sentences(dst) or [dst]
        self.src_text = src if not mostly_chinese(src) else ""

        for s in sents:
            self._q.append(s)
        if self._q_job is None:
            self._next()

    def _next(self):
        self._q_job = None
        if not self._q:
            return
        s = self._q.pop(0)
        self.dst_text = s
        self.redraw()
        self._schedule_fit()
        # 停留时间按字数估：每字 0.16s，最少 1.0s，最多 3.5s
        dwell = min(3.5, max(1.0, len(s) * 0.16))
        self._q_job = self.win.after(int(dwell * 1000), self._next)

    def clear_queue(self):
        if self._q_job:
            try:
                self.win.after_cancel(self._q_job)
            except Exception:
                pass
        self._q_job = None
        self._q = []

    def set_source(self, src):
        self.src_line = src or ""
        if src:
            self._last_active_at = time.time()
        self.redraw()

    def _ensure_width_now(self):
        """文字变长就立刻把窗口放宽一点，避免先折行再展开的跳动"""
        if not self.auto_width or self.adjusting:
            return
        if self._fitting:
            return
        self._fit_width()

    def set_stream_text(self, text):
        """流式累积文本 → 上面的行固定、最后一行实时增长

        关键约束：**上一行只能是「已经显示过」的内容**，
        没显示过的一律放最新行，新旧绝不能混在一行里。
        """
        raw_text = text

        def split_seg(t):
            segs = []
            # 停顿标记也是切分边界
            for chunk in t.replace("\u0001", "\n").split("\n"):
                chunk = chunk.strip()
                if not chunk:
                    continue
                # 先把句末的延迟标记「（x.xs）」摘下来，
                # 否则它会被逗号/句号切分成独立一行
                m_tail = re.search(r"(（\d+\.\d+s）)$", chunk)
                tail = m_tail.group(1) if m_tail else ""
                body = chunk[:m_tail.start()].strip() if m_tail else chunk
                if not body:
                    if tail:
                        segs.append(tail)
                    continue
                # 句末标点 + 逗号类都分段：逗号处也断开，避免堆成一大段
                sub = [x.strip() for x in re.split(r"(?<=[。！？!?，、,;；])", body)
                       if x.strip()]
                if sub and tail:
                    sub[-1] = sub[-1] + tail      # 标记贴回本句最后一段
                segs.extend(sub)
            return segs

        def refine(parts):
            # 纯标点的片段直接丢掉（如单独一个「。」「，」），它不是内容
            parts = [x for x in parts
                     if re.sub(r"[。！？!?，、,;；\s]", "", x)]
            # 一两个字的碎片不单独成行，攒着合并到下一行
            merged, pending = [], []
            for x in parts:
                xs = x.strip()
                # 一两个字但以标点结尾的是完整短句（「是。」「好，」），要显示；
                # 没标点的短碎片（「那么」）才攒着合并到下一行。
                # 否则短句全被吞掉，会连续跳过好几句。
                ends_punct = bool(xs) and xs[-1] in "。！？!?，、,;；"
                if len(xs) < 3 and not ends_punct:
                    pending.append(xs)
                else:
                    if pending:
                        merged.append("".join(pending) + x)
                        pending = []
                    else:
                        merged.append(x)
            if pending:
                merged.append("".join(pending))
            # 长句按字数断行换下一行
            MAXL = 20
            out = []
            for pp in merged:
                # 末尾的延迟标记（如「（2.0s）」）不参与断行，始终跟在同一句后面
                m_tail = re.search(r"(（\d+\.\d+s）)$", pp)
                tail = ""
                if m_tail:
                    tail = m_tail.group(1)
                    pp = pp[:m_tail.start()]
                while len(pp) > MAXL:
                    out.append(pp[:MAXL])
                    pp = pp[MAXL:]
                if pp or tail:
                    out.append(pp + tail)
            return out

        prev_text = getattr(self, "_prev_text", "") or ""

        if prev_text and text.startswith(prev_text) and len(text) > len(prev_text):
            # ═══════════════════════════════════════════════════════════
            # 【改进实验 · 可回退】以断句为准，整段一次性显示
            #   原行为：新增内容实时追加（打字机，一个字一个字蹦，看着累）
            #   新行为：累积到断句点（标点结尾）才一次性显示完整片段
            #   回退方法：把下面「if not complete: return」改成直接显示 new_txt
            # ═══════════════════════════════════════════════════════════
            new_txt = text[len(prev_text):].strip()
            new_txt = " ".join(x for x in new_txt.split("\n") if x.strip())

            if self.conf.get("punct_batch", True):
                # 【按断句整段显示】累积到断句点才一次性出完整片段，
                # 不再一个字一个字蹦。可在设置里关掉（关掉=打字机模式）
                segs = [x.strip() for x in re.split(r"(?<=[。！？!?，、,;；])", new_txt)
                        if x.strip()]
                complete = [x for x in segs
                            if x and x[-1] in "。！？!?，、,;；"
                            and re.sub(r"[。！？!?，、,;；\s]", "", x)]
                if not complete:
                    # 还没到断句点 → 不显示，继续累积
                    return
                shown = refine(split_seg(prev_text))
                self.lines = shown[-2:]
                if len(complete) > 1:
                    self.lines = (self.lines + complete[:-1])[-2:]
                self.active = complete[-1]
            else:
                # 【打字机模式】新增内容实时追加
                shown = refine(split_seg(prev_text))
                self.lines = shown[-2:]
                if re.sub(r"[。！？!?，、,;；\s]", "", new_txt):
                    self.active = new_txt
        else:
            # 新内容（换句了）—— 上一行还没看够的话先攒着，别急着顶掉
            now = time.time()
            try:
                base_min = float(self.conf.get("min_show", 2.5))
            except Exception:
                base_min = 2.5
            # 说话密集时加快刷新：换句间隔越短，最短显示时长越小，
            # 否则密集时会攒一堆、然后一下子冒出来，看着累
            gap = (now - self._line_shown_at) if self._line_shown_at else 999.0
            if gap < 2.0:
                min_show = max(0.8, base_min * 0.4)    # 很密集，快速刷
            elif gap < 3.5:
                min_show = max(1.2, base_min * 0.7)    # 较密集
            else:
                min_show = base_min                    # 正常节奏
            # 短句读得快，显示时间更短，别让它排长队
            if len(text.strip()) < 6:
                min_show = max(0.5, min_show * 0.45)
            if (not self._flushing and min_show > 0 and self._line_shown_at):
                elapsed = now - self._line_shown_at
                if 0 <= elapsed < min_show:
                    # 排队而不是覆盖：连续短句不会被跳过
                    self._pending_q.append(text)
                    if not self._pending_job:
                        self._pending_job = self.win.after(
                            int((min_show - elapsed) * 1000) + 40,
                            self._flush_pending)
                    return

            parts = refine(split_seg(text))
            if not parts:
                # 全是标点/空，没有有效内容 → 保持当前显示，别清空
                self._prev_text = raw_text
                return
            if len(parts) > 1:
                self.lines = parts[:-1][-2:]
            self.active = parts[-1]
            self._line_shown_at = now

        # 去重：上一行和本行不能一样
        if self.active and self.lines and self.lines[-1] == self.active:
            self.lines = self.lines[:-1]

        self.dst_text = self.active
        self.state = "live"
        self._last_active_at = time.time()
        self._ensure_width_now()
        self._prev_text = raw_text
        self.redraw()

    def _flush_pending(self):
        """最短显示时间到了，把排队的新内容按顺序放出来（不跳过）"""
        self._pending_job = None
        if self._pending_q:
            t = self._pending_q.pop(0)
            self._flushing = True
            try:
                self.set_stream_text(t)
            finally:
                self._flushing = False
            # 队列还有就继续，每条间隔短一点
            if self._pending_q:
                self._pending_job = self.win.after(220, self._flush_pending)

    def finalize(self):
        """一句结束：把它变成固定行，等下一句"""
        # 如果这一句还没翻译、屏幕停在上句，不要重复入列
        if self.active and not (self.lines and self.lines[-1] == self.active):
            self.lines = (self.lines + [self.active])[-2:]
            self.active = ""
            self.redraw()
        self._schedule_fit()

    def show_partial(self, text):
        """流式中间结果：直接替换当前行，像打字机"""
        self.dst_text = text
        self.state = "live"
        self.redraw()
        # 字还在往外冒的时候不要动窗口宽度，等它停下来再贴
        if getattr(self, "_settle_job", None):
            try:
                self.win.after_cancel(self._settle_job)
            except Exception:
                pass
        self._settle_job = self.win.after(650, self._settle_fit)

    def _settle_fit(self):
        self._settle_job = None
        self._schedule_fit()

    def commit(self):
        """一句结束：稳定下来后再贴宽度，避免逐字变化时窗口抖动"""
        self._schedule_fit()

    def _check_idle(self):
        """一段时间没有新内容就把字幕淡出，免得屏幕上一直挂着旧句子"""
        try:
            idle_for = time.time() - (self._last_active_at or 0)
            has_content = bool(self.lines or self.active or self.src_line)
            if has_content and idle_for > float(self.conf.get("idle_hide", 5.0)):
                self.lines = []
                self.active = ""
                self.src_line = ""
                self.dst_text = ""
                self.redraw()
        except Exception:
            pass
        try:
            self.win.after(500, self._check_idle)
        except Exception:
            pass

    def set_state(self, s):
        self.state = s
        self.redraw()

    def show(self):
        self.win.deiconify()
        self.win.attributes("-topmost", True)
        self.conf["show_sub"] = True
        save_conf(self.conf)
        self.win.after(120, self.apply_style)
        # 【已撤】定时 SetWindowPos 强制置顶会让透明窗口反复重合成、导致闪烁，
        # 且对腾讯视频无效。待办见需求清单 F27。
        self.win.after(800, self._check_idle)

    def hide(self):
        self.win.withdraw()
        self.conf["show_sub"] = False
        save_conf(self.conf)

    # ---------- 绘制 ----------
    def redraw(self):
        if getattr(self, "_in_redraw", False):
            return
        self._in_redraw = True
        c = self.canvas
        c.delete("all")
        w, h = max(1, self.win.winfo_width()), max(1, self.win.winfo_height())

        p = PRESETS.get(self.conf.get("preset", "outline"), PRESETS["outline"])

        # 滚动字幕：已完成的行固定在上，正在说的那句在最底部增长
        rows = [x for x in self.lines if x]
        if self.active:
            rows.append(self.active)
        if not rows and self.dst_text:
            rows = [self.dst_text]
        rows = rows[-3:]

        # 完全没内容时什么都别画，让窗口彻底透明
        has_any = any(t.strip() for t in rows) or (self.show_source and self.src_line.strip())
        if not has_any and not self.adjusting:
            self._in_redraw = False
            return

        # 字号只由「窗口高度 × 倍率」决定，不因行数变多而缩水。
        # 长句会折行，所以按每行的「实际渲染高度」从下往上堆。
        fs = max(12, int(h * self.conf.get("ratio", 0.16)))
        fs = min(fs, max(12, int(h * 0.86)))
        fs_src = max(9, int(fs * 0.58))

        try:
            fnt = tkfont.Font(family=FONT, size=-fs, weight="bold")
            fnt_src = tkfont.Font(family=FONT, size=-fs_src)
        except Exception:
            fnt = fnt_src = None

        # 文字区在窗口里左右留白对称：窗口居中时文字块看起来也是居中的。
        # 但左边界是固定的，所以文字变长只往右长，不会左右晃。
        wrap = int(w * 0.90)
        cx = max(8, (w - wrap) // 2)
        base = h - int(h * 0.04)

        def linespace(f, fallback):
            try:
                return f.metrics("linespace")
            except Exception:
                return fallback

        ls = linespace(fnt, int(fs * 1.35))
        ls_src = linespace(fnt_src, int(fs_src * 1.35))
        gap = max(2, int(fs * 0.16))

        # 组装条目：最上面是可选的原文行，下面是译文行
        entries = []
        if self.show_source and self.src_line:
            entries.append((self.src_line, True))
        for t in rows:
            entries.append((t, False))
        entries = entries[-4:]

        def h_of(t, is_src):
            f, l = (fnt_src, ls_src) if is_src else (fnt, ls)
            if f is None:
                return l
            try:
                tw = f.measure(t)
            except Exception:
                return l
            return max(1, -(-tw // max(1, wrap))) * l

        heights = [h_of(t, is_src) for t, is_src in entries]

        avail = h * 0.94
        total, keep = 0, len(entries)
        for k in range(len(entries) - 1, -1, -1):
            add = heights[k] + (gap if total else 0)
            if total + add > avail:
                break
            total += add
            keep = k

        entries = entries[keep:]
        heights = heights[keep:]
        if not entries:
            entries = [(self.active or self.dst_text or "", False)]
            heights = [min(h_of(*entries[0]), int(avail))]
        n = len(entries)

        y = base
        active_item = None
        for i in range(n - 1, -1, -1):
            txt, is_src = entries[i]
            f_size = fs_src if is_src else fs
            color = "#AEB8C4" if is_src else (p["text"] if i == n - 1 else "#C3CCD8")
            off = max(1, int(f_size * 0.055))
            for dx, dy in ((-off, 0), (off, 0), (0, -off), (0, off),
                           (-off, -off), (off, -off), (-off, off), (off, off)):
                c.create_text(cx + dx, y + dy, text=txt, fill=p["outline"],
                              font=(FONT, -f_size), width=wrap,
                              justify="left", anchor="sw")
            _it = c.create_text(cx, y, text=txt, fill=color,
                                font=(FONT, -f_size), width=wrap,
                                justify="left", anchor="sw")
            if i == n - 1:
                active_item = _it
            y -= heights[i] + gap

        # 延迟徽标是独立窗口，这里不画


        # 调整模式提示
        if self.adjusting:
            c.create_rectangle(1, 1, w - 2, h - 2, outline="#3DDC97", width=2, dash=(6, 4))
            c.create_text(w // 2, 12, text="拖动移动 · 右下角调大小 · 右键结束调整",
                          fill="#3DDC97", font=(FONT, 9), anchor="n")

        if self._resize:
            for i in range(3):
                c.create_line(w - 6 - i * 5, h - 6, w - 6, h - 6 - i * 5, fill="#3DDC97", width=1)

        self._in_redraw = False

    def on_quit(self):
        log("quit from subtitle menu")
        save_conf(self.conf)
        os._exit(0)


# ============================ 界面组件 ============================
BG = "#0B0D11"
CARD = "#12161C"
CARD2 = "#151A20"
LINE = "#232A33"
TXT = "#EDF1F5"
TXT2 = "#A2ACB9"
TXT3 = "#6B7684"
ACC = "#3DDC97"


def card(parent, title):
    f = tk.Frame(parent, bg=CARD, highlightbackground=LINE, highlightthickness=1)
    f.pack(fill="x", padx=14, pady=(0, 5))
    if title:
        tk.Label(f, text=title, bg=CARD, fg=TXT3, font=(FONT, 9)).pack(
            anchor="w", padx=12, pady=(6, 0))
    return f


# ============================ 设置窗口 ============================
class Settings:
    def __init__(self, root, conf, app):
        self.root = root
        self.conf = conf
        self.app = app

        root.title("字幕助手")
        root.geometry("500x700")
        root.configure(bg=BG)
        root.resizable(False, False)

        # ---- 顶部 ----
        head = tk.Frame(root, bg=BG)
        head.pack(fill="x", padx=14, pady=(10, 6))
        tk.Label(head, text="字幕助手", bg=BG, fg=TXT, font=(FONT, 16, "bold")).pack(anchor="w")
        tk.Label(head, text="把听到的，变成看得懂的", bg=BG, fg=TXT3,
                 font=(FONT, 9)).pack(anchor="w", pady=(2, 0))

        # ---- Key 卡片 ----
        c1 = card(root, "模型密钥")
        row = tk.Frame(c1, bg=CARD)
        row.pack(fill="x", padx=12, pady=(4, 3))

        self.key_names = [k.get("name", "默认") for k in conf["keys"]] or ["（未配置）"]
        self.key_menu_var = tk.StringVar(value=self.key_names[min(conf["active"], len(self.key_names) - 1)])
        self.key_menu = tk.OptionMenu(row, self.key_menu_var, *self.key_names,
                                      command=self._on_pick_key)
        self.key_menu.config(bg=CARD2, fg=TXT2, activebackground=CARD2, activeforeground=TXT,
                             relief="flat", highlightthickness=0, font=(FONT, 10),
                             width=10, anchor="w")
        self.key_menu["menu"].config(bg=CARD2, fg=TXT2, activebackground=LINE,
                                     activeforeground=TXT, font=(FONT, 10), bd=0)
        self.key_menu.pack(side="left")

        tk.Button(row, text="＋", command=self._add_key, bg=CARD2, fg=ACC, relief="flat",
                  font=(FONT, 11), cursor="hand2", width=3,
                  activebackground=LINE).pack(side="left", padx=(6, 0))
        tk.Button(row, text="－", command=self._del_key, bg=CARD2, fg="#FF9C88", relief="flat",
                  font=(FONT, 11), cursor="hand2", width=3,
                  activebackground=LINE).pack(side="left", padx=(4, 0))

        self.key_var = tk.StringVar(value=active_key(conf))
        e = tk.Entry(c1, textvariable=self.key_var, show="•", bg=CARD2, fg=TXT,
                     insertbackground=TXT, relief="flat", font=("Consolas", 10))
        e.pack(fill="x", padx=12, pady=(0, 3), ipady=6)

        self.key_status = tk.Label(c1, text="", bg=CARD, fg=ACC, font=(FONT, 9), anchor="w")
        self.key_status.pack(fill="x", padx=12, pady=(0, 7))

        self._key_job = None
        self.key_var.trace_add("write", lambda *a: self._on_key_change())
        self._refresh_key_status()

        # ---- 字幕卡片 ----
        c2 = card(root, "字幕")
        r2 = tk.Frame(c2, bg=CARD)
        r2.pack(fill="x", padx=12, pady=(5, 4))
        tk.Label(r2, text="显示", bg=CARD, fg=TXT2, font=(FONT, 10), width=6,
                 anchor="w").pack(side="left")
        self.btn_show = tk.Button(r2, text="", command=self._toggle_sub, bg=CARD2, fg=TXT2,
                                  relief="flat", font=(FONT, 10), cursor="hand2", width=10,
                                  activebackground=LINE)
        self.btn_show.pack(side="left")
        self._refresh_sub_btn()

        r3 = tk.Frame(c2, bg=CARD)
        r3.pack(fill="x", padx=12, pady=(0, 4))
        tk.Label(r3, text="背景", bg=CARD, fg=TXT2, font=(FONT, 10), width=6,
                 anchor="w").pack(side="left")
        self.bg_var = tk.StringVar(value=conf.get("bg_mode", "transparent"))
        tk.Radiobutton(r3, text="透明（只显示字）", variable=self.bg_var, value="transparent",
                       command=self._on_bg, bg=CARD, fg=TXT2, selectcolor=CARD2,
                       activebackground=CARD, activeforeground=TXT, font=(FONT, 10),
                       bd=0, highlightthickness=0).pack(side="left", padx=(0, 10))
        tk.Radiobutton(r3, text="半透明底", variable=self.bg_var, value="plate",
                       command=self._on_bg, bg=CARD, fg=TXT2, selectcolor=CARD2,
                       activebackground=CARD, activeforeground=TXT, font=(FONT, 10),
                       bd=0, highlightthickness=0).pack(side="left")

        r3b = tk.Frame(c2, bg=CARD)
        self.r3b = r3b
        r3b.pack(fill="x", padx=12, pady=(0, 8))
        tk.Label(r3b, text="深浅", bg=CARD, fg=TXT2, font=(FONT, 10), width=6,
                 anchor="w").pack(side="left")
        self.alpha_var = tk.DoubleVar(value=conf.get("alpha", 0.62))
        self.alpha_scale = tk.Scale(r3b, from_=0.25, to=1.0, resolution=0.05,
                 orient="horizontal", variable=self.alpha_var, bg=CARD, fg=TXT2,
                 troughcolor=CARD2, highlightthickness=0, bd=0, sliderlength=14,
                 width=10, font=(FONT, 8), command=self._on_alpha)
        self.alpha_scale.pack(side="left", fill="x", expand=True)
        self._refresh_bg_state()

        # ---- 倍速（下拉，省一行）----
        r4 = tk.Frame(c2, bg=CARD)
        r4.pack(fill="x", padx=12, pady=(0, 4))
        tk.Label(r4, text="倍速", bg=CARD, fg=TXT2, font=(FONT, 10), width=6,
                 anchor="w").pack(side="left")
        _RATE = {"1.0x 完整": 10, "1.5x 精简": 15, "2.0x 压缩": 20}
        _cur_rate = {v: k for k, v in _RATE.items()}.get(
            int(conf.get("rate", 1) * 10), "1.0x 完整")
        self.rate_var = tk.IntVar(value=_RATE[_cur_rate])
        self.rate_name = tk.StringVar(value=_cur_rate)
        _rm = tk.OptionMenu(r4, self.rate_name, *_RATE.keys(), command=self._on_rate_name)
        _rm.config(bg=CARD2, fg=TXT2, activebackground=CARD2, activeforeground=TXT,
                   relief="flat", highlightthickness=0, font=(FONT, 9), anchor="w")
        _rm["menu"].config(bg=CARD2, fg=TXT2, activebackground=LINE,
                           activeforeground=TXT, font=(FONT, 9), bd=0)
        _rm.pack(side="left")

        # ---- 语言 ----
        r5 = tk.Frame(c2, bg=CARD)
        r5.pack(fill="x", padx=12, pady=(0, 4))

        # ---- 引擎 + 本地模型（合并一行，省空间）----
        r_eng = tk.Frame(c2, bg=CARD)
        r_eng.pack(fill="x", padx=12, pady=(0, 4))
        tk.Label(r_eng, text="引擎", bg=CARD, fg=TXT2, font=(FONT, 10), width=6,
                 anchor="w").pack(side="left")
        self.engine_var = tk.StringVar(value=conf.get("asr_engine", "cloud"))
        for _val, _label in [("cloud", "在线"), ("local", "本地")]:
            tk.Radiobutton(r_eng, text=_label, variable=self.engine_var, value=_val,
                           command=self._on_engine, bg=CARD, fg=TXT2,
                           selectcolor=CARD2, activebackground=CARD,
                           activeforeground=TXT, font=(FONT, 10), bd=0,
                           highlightthickness=0).pack(side="left", padx=(0, 8))

        # 本地模型：显示短名，避免长名把窗口撑宽
        _base = os.path.dirname((conf.get("local_model_dir") or "D:/sherpa-models").rstrip("/"))
        _full = []
        try:
            for _d in sorted(os.listdir(_base)):
                _p = os.path.join(_base, _d)
                if os.path.isdir(_p) and os.path.exists(os.path.join(_p, "tokens.txt")):
                    _full.append(_d)
        except Exception:
            pass
        _cur_full = os.path.basename((conf.get("local_model_dir") or "").rstrip("/")) or ""
        if _cur_full and _cur_full not in _full:
            _full = [_cur_full] + _full
        if not _full:
            _full = ["（未发现模型）"]

        def _short(n):
            low = n.lower()
            if "paraformer" in low:
                return "Paraformer"
            if "x-asr" in low:
                c = ("160" if "160ms" in low else "960" if "960ms" in low
                     else "1920" if "1920ms" in low else "480")
                return "X-ASR %s %s" % (c, "int8" if "int8" in low else "fp32")
            return n[:16]

        self._model_map = {}
        _labels = []
        for _f in _full:
            _s = _short(_f)
            self._model_map[_s] = _f
            _labels.append(_s)
        _cur_label = _short(_cur_full) if _cur_full else _labels[0]
        self.local_model_var = tk.StringVar(value=_cur_label)
        _ml = tk.OptionMenu(r_eng, self.local_model_var, *_labels,
                            command=self._on_local_model)
        _ml.config(bg=CARD2, fg=TXT2, activebackground=CARD2, activeforeground=TXT,
                   relief="flat", highlightthickness=0, font=(FONT, 9), anchor="w")
        _ml["menu"].config(bg=CARD2, fg=TXT2, activebackground=LINE,
                           activeforeground=TXT, font=(FONT, 9), bd=0)
        _ml.pack(side="left", padx=(4, 0))

        tk.Label(r5, text="语言", bg=CARD, fg=TXT2, font=(FONT, 10), width=6,
                 anchor="w").pack(side="left")

        src_names = [n for _, n in SRC_LANGS]
        cur_src = dict((c, n) for c, n in SRC_LANGS).get(conf.get("src_lang", "auto"), "自动")
        self.src_var = tk.StringVar(value=cur_src)
        m1 = tk.OptionMenu(r5, self.src_var, *src_names, command=self._on_src_lang)
        m1.config(bg=CARD2, fg=TXT2, activebackground=CARD2, activeforeground=TXT,
                  relief="flat", highlightthickness=0, font=(FONT, 10), width=5, anchor="w")
        m1["menu"].config(bg=CARD2, fg=TXT2, activebackground=LINE,
                          activeforeground=TXT, font=(FONT, 10), bd=0)
        m1.pack(side="left")

        tk.Label(r5, text="→", bg=CARD, fg=TXT3, font=(FONT, 11)).pack(side="left", padx=5)

        dst_names = [n for _, n in DST_LANGS]
        cur_dst = dict((c, n) for c, n in DST_LANGS).get(conf.get("dst_lang", "zh"), "中文")
        self.dst_var = tk.StringVar(value=cur_dst)
        m2 = tk.OptionMenu(r5, self.dst_var, *dst_names, command=self._on_dst_lang)
        m2.config(bg=CARD2, fg=TXT2, activebackground=CARD2, activeforeground=TXT,
                  relief="flat", highlightthickness=0, font=(FONT, 10), width=5, anchor="w")
        m2["menu"].config(bg=CARD2, fg=TXT2, activebackground=LINE,
                          activeforeground=TXT, font=(FONT, 10), bd=0)
        m2.pack(side="left")



        r6 = tk.Frame(c2, bg=CARD)
        r6.pack(fill="x", padx=12, pady=(0, 4))
        tk.Label(r6, text="", bg=CARD, width=6).pack(side="left")
        self.show_src_var = tk.BooleanVar(value=conf.get("show_source", False))
        tk.Checkbutton(r6, text="显示原文（双语字幕）", variable=self.show_src_var,
                       command=self._on_show_source, bg=CARD, fg=TXT2,
                       selectcolor=CARD2, activebackground=CARD,
                       activeforeground=TXT, font=(FONT, 10), bd=0,
                       highlightthickness=0).pack(side="left")

        r6b = tk.Frame(c2, bg=CARD)
        r6b.pack(fill="x", padx=12, pady=(0, 4))
        tk.Label(r6b, text="", bg=CARD, width=6).pack(side="left")
        self.punct_batch_var = tk.BooleanVar(value=conf.get("punct_batch", True))
        tk.Checkbutton(r6b, text="按断句整段显示（关=打字机）", variable=self.punct_batch_var,
                       command=self._on_punct_batch, bg=CARD, fg=TXT2,
                       selectcolor=CARD2, activebackground=CARD,
                       activeforeground=TXT, font=(FONT, 10), bd=0,
                       highlightthickness=0).pack(side="left")

        r6c = tk.Frame(c2, bg=CARD)
        r6c.pack(fill="x", padx=12, pady=(0, 8))
        tk.Label(r6c, text="", bg=CARD, width=6).pack(side="left")
        self.show_latency_var = tk.BooleanVar(value=conf.get("show_latency", False))
        tk.Checkbutton(r6c, text="显示延迟", variable=self.show_latency_var,
                       command=self._on_show_latency, bg=CARD, fg=TXT2,
                       selectcolor=CARD2, activebackground=CARD,
                       activeforeground=TXT, font=(FONT, 10), bd=0,
                       highlightthickness=0).pack(side="left")

        # 延迟底色主题（跟勾选框同一行，不额外占高度）
        self.lat_theme_var = tk.StringVar(
            value=conf.get("latency_theme", LAT_THEME_DEFAULT))
        _lt = tk.OptionMenu(r6c, self.lat_theme_var, *LAT_THEMES.keys(),
                            command=self._on_lat_theme)
        _lt.config(bg=CARD2, fg=TXT2, activebackground=CARD2, activeforeground=TXT,
                   relief="flat", highlightthickness=0, font=(FONT, 9), anchor="w")
        _lt["menu"].config(bg=CARD2, fg=TXT2, activebackground=LINE,
                           activeforeground=TXT, font=(FONT, 9), bd=0)
        _lt.pack(side="left", padx=(6, 0))

        # ---- 状态卡片 ----
        c3 = card(root, "状态")
        rdev = tk.Frame(c3, bg=CARD)
        rdev.pack(fill="x", padx=12, pady=(4, 2))
        tk.Label(rdev, text="音频源", bg=CARD, fg=TXT2, font=(FONT, 10), width=6,
                 anchor="w").pack(side="left")
        devs = list_loopback_devices()
        cur = conf.get("audio_device") or "（默认输出）"
        self.dev_var = tk.StringVar(value=cur)
        self.dev_menu = tk.OptionMenu(rdev, self.dev_var,
                                      "（默认输出）", *devs, command=self._on_pick_dev)
        self.dev_menu.config(bg=CARD2, fg=TXT2, activebackground=CARD2, activeforeground=TXT,
                             relief="flat", highlightthickness=0, font=(FONT, 9),
                             anchor="w", width=26)
        self.dev_menu["menu"].config(bg=CARD2, fg=TXT2, activebackground=LINE,
                                     activeforeground=TXT, font=(FONT, 9), bd=0)
        self.dev_menu.pack(side="left")
        tk.Button(rdev, text="↻", command=self._reload_devices, bg=CARD2, fg=TXT2,
                  relief="flat", font=(FONT, 10), cursor="hand2", width=2,
                  activebackground=LINE).pack(side="left", padx=(4, 0))

        self.dev_label = tk.Label(c3, text="", bg=CARD, fg=TXT3,
                                  font=(FONT, 9), anchor="w", justify="left")
        self.dev_label.pack(fill="x", padx=12, pady=(0, 1))
        self.status = tk.Label(c3, text="", bg=CARD, fg="#F0A93B", font=(FONT, 9),
                               anchor="w", justify="left", wraplength=440)
        self.status.pack(fill="x", padx=12, pady=(0, 8))

        # ---- 主按钮 ----
        self.btn = tk.Button(root, text="开始听", command=self._toggle,
                             bg=ACC, fg="#04160D", relief="flat",
                             font=(FONT, 13, "bold"), cursor="hand2",
                             activebackground="#54EDAA")
        self.btn.pack(fill="x", padx=14, pady=(2, 4), ipady=8)

        tk.Label(root, text="点开始后可以关掉这个窗口，字幕会留在屏幕上。双击字幕条可隐藏。",
                 bg=BG, fg=TXT3, font=(FONT, 8)).pack(anchor="w", padx=14)

        root.protocol("WM_DELETE_WINDOW", self._on_close)

        # 按实际内容收缩高度。必须等一轮布局完成再量，否则量到的是半成品
        root.after(120, self._fit_height)

    # ---------- key ----------
    def _on_pick_key(self, name):
        names = [k.get("name", "") for k in self.conf["keys"]]
        if name in names:
            i = names.index(name)
            self.conf["active"] = i
            self.key_var.set(self.conf["keys"][i]["key"])
            save_conf(self.conf)

    def _add_key(self):
        k = self.key_var.get().strip()
        if not is_valid_key(k):
            messagebox.showwarning("Key 不合法",
                                   "要以 sk- 开头，只含字母数字和 . _ -\n\n当前输入：%s"
                                   % (k[:40] or "（空）"))
            return
        for item in self.conf["keys"]:
            if item["key"] == k:
                messagebox.showinfo("已存在", "这个 key 已经保存过了")
                return
        name = "key%d" % (len(self.conf["keys"]) + 1)
        self.conf["keys"].append({"name": name, "key": k})
        self.conf["active"] = len(self.conf["keys"]) - 1
        save_conf(self.conf)
        self._rebuild_key_menu(name)
        self._refresh_key_status()

    def _del_key(self):
        if len(self.conf["keys"]) <= 1:
            messagebox.showinfo("删不了", "至少保留一个 key")
            return
        i = self.conf.get("active", 0)
        name = self.conf["keys"][i].get("name", "")
        if not messagebox.askyesno("确认", "删除 %s？" % name):
            return
        self.conf["keys"].pop(i)
        self.conf["active"] = 0
        save_conf(self.conf)
        self._rebuild_key_menu(self.conf["keys"][0].get("name", ""))
        self.key_var.set(self.conf["keys"][0]["key"])
        self._refresh_key_status()

    def _rebuild_key_menu(self, select):
        names = [k.get("name", "") for k in self.conf["keys"]] or ["（未配置）"]
        m = self.key_menu["menu"]
        m.delete(0, "end")
        for n in names:
            m.add_command(label=n, command=lambda x=n: (self.key_menu_var.set(x), self._on_pick_key(x)))
        self.key_menu_var.set(select if select in names else names[0])

    def _refresh_key_status(self):
        k = self.key_var.get().strip()
        if is_valid_key(k):
            self.key_status.config(text="✓ 已保存 %s   下次打开自动带出" % mask_key(k), fg=ACC)
        elif k:
            self.key_status.config(text="格式不对：应以 sk- 开头，只含字母数字和 . _ -", fg="#F0A93B")
        else:
            self.key_status.config(text="未设置", fg=TXT3)

    def _on_key_change(self):
        self._refresh_key_status()
        if self._key_job:
            self.root.after_cancel(self._key_job)
        self._key_job = self.root.after(700, self._persist_key)

    def _persist_key(self):
        k = self.key_var.get().strip()
        if not is_valid_key(k):
            return
        i = self.conf.get("active", 0)
        if 0 <= i < len(self.conf["keys"]):
            if self.conf["keys"][i]["key"] != k:
                self.conf["keys"][i]["key"] = k
                save_conf(self.conf)
                log("key: 更新 %s" % mask_key(k))

    # ---------- 字幕控制 ----------
    def _refresh_sub_btn(self):
        if self.app.sub.win.state() == "withdrawn" or not self.conf.get("show_sub", True):
            self.btn_show.config(text="显示字幕", fg=ACC)
        else:
            self.btn_show.config(text="隐藏字幕", fg=TXT2)

    def _toggle_sub(self):
        if self.conf.get("show_sub", True):
            self.app.sub.hide()
        else:
            self.app.sub.show()
        self._refresh_sub_btn()

    def _on_alpha(self, v):
        self.conf["alpha"] = float(v)
        try:
            self.app.sub.win.attributes("-alpha", float(v))
        except Exception:
            pass
        if getattr(self, "_alpha_job", None):
            try:
                self.root.after_cancel(self._alpha_job)
            except Exception:
                pass
        self._alpha_job = self.root.after(600, lambda: save_conf(self.conf))

    def _on_pick_dev(self, name):
        self.conf["audio_device"] = "" if name.startswith("（默认") else name
        save_conf(self.conf)
        if self.app.running:
            self.status.config(text="音频源已改，停止后再开始才生效")

    def _reload_devices(self):
        devs = list_loopback_devices()
        m = self.dev_menu["menu"]
        m.delete(0, "end")
        m.add_command(label="（默认输出）",
                      command=lambda: (self.dev_var.set("（默认输出）"),
                                       self._on_pick_dev("（默认输出）")))
        for d in devs:
            m.add_command(label=d, command=lambda x=d: (self.dev_var.set(x), self._on_pick_dev(x)))

    def _fit_height(self):
        try:
            self.root.update_idletasks()
            need = self.root.winfo_reqheight() + 8
            _sh = self.root.winfo_screenheight()
            _maxh = int(_sh * 0.92)          # 不超出屏幕（1080p 下约 994）
            self.root.geometry("500x%d" % min(max(need, 420), _maxh))
        except Exception as e:
            log("fit height failed: %s" % e)

    def _on_src_lang(self, name):
        code = dict((n, c) for c, n in SRC_LANGS).get(name, "auto")
        self.conf["src_lang"] = code
        save_conf(self.conf)
        if self.app.running:
            self.status.config(text="源语言改了，停止后再开始才生效")

    def _on_dst_lang(self, name):
        code = dict((n, c) for c, n in DST_LANGS).get(name, "zh")
        self.conf["dst_lang"] = code
        save_conf(self.conf)

    def _on_local_model(self, name):
        if name.startswith("（"):
            return
        full = getattr(self, "_model_map", {}).get(name, name)
        _base = os.path.dirname((self.conf.get("local_model_dir") or "D:/sherpa-models").rstrip("/"))
        self.conf["local_model_dir"] = _base + "/" + full
        save_conf(self.conf)
        if self.app.running:
            self.status.config(text="本地模型切换需停止后重新开始")

    def _on_engine(self):
        v = self.engine_var.get()
        self.conf["asr_engine"] = v
        save_conf(self.conf)
        if v == "local":
            md = self.conf.get("local_model_dir", "")
            if not os.path.isdir(md):
                self.status.config(text="本地模型未下载：%s" % md, fg="#e0a030")
            elif self.app.running:
                self.status.config(text="引擎切换需停止后重新开始")
        else:
            if self.app.running:
                self.status.config(text="引擎切换需停止后重新开始")

    def _on_lat_theme(self, name):
        self.conf["latency_theme"] = name
        save_conf(self.conf)
        try:
            if getattr(self.app, "sub", None):
                self.app.badge.set_theme(name)
        except Exception:
            pass

    def _on_show_latency(self):
        self.conf["show_latency"] = bool(self.show_latency_var.get())
        save_conf(self.conf)

    def _on_punct_batch(self):
        self.conf["punct_batch"] = bool(self.punct_batch_var.get())
        save_conf(self.conf)

    def _on_show_source(self):
        v = self.show_src_var.get()
        self.conf["show_source"] = v
        save_conf(self.conf)
        self.app.sub.show_source = v
        self.app.sub.redraw()

    def _refresh_bg_state(self):
        # 透明模式下「深浅」这一行没意义，直接收起来，省一大块空间
        if self.bg_var.get() == "transparent":
            try:
                self.r3b.pack_forget()
            except Exception:
                pass
        else:
            try:
                if not self.r3b.winfo_ismapped():
                    self.r3b.pack(fill="x", padx=12, pady=(0, 8))
            except Exception:
                pass

    def _on_bg(self):
        self.conf["bg_mode"] = self.bg_var.get()
        save_conf(self.conf)
        self._refresh_bg_state()
        self.app.sub.apply_style()

    def _on_rate_name(self, name):
        v = {"1.0x 完整": 10, "1.5x 精简": 15, "2.0x 压缩": 20}.get(name, 10)
        self.rate_var.set(v)
        self._on_rate()

    def _on_rate(self):
        self.conf["rate"] = self.rate_var.get() / 10
        save_conf(self.conf)

    # ---------- 运行 ----------
    def _toggle(self):
        if self.app.running:
            self.app.stop()
            self.btn.config(text="开始听", bg=ACC, fg="#04160D")
            self.dev_label.config(text="音频设备：已停止")
        else:
            k = self.key_var.get().strip()
            if not is_valid_key(k):
                messagebox.showwarning("Key 不对",
                                       "要以 sk- 开头，只含字母数字和 . _ -\n\n当前：%s"
                                       % (k[:40] or "（空）"))
                return
            self._persist_key()
            ok, msg = self.app.start()
            if ok:
                self.btn.config(text="停止", bg=CARD2, fg="#FF9C88")
                self.status.config(text="")
                self.dev_label.config(text="音频设备：%s" % (self.app.engine.device_name or "-"))
            else:
                self.status.config(text=msg)

    def _on_close(self):
        if self.app.running:
            self.root.withdraw()
        else:
            save_conf(self.conf)
            self.root.quit()
            self.root.destroy()

    def show(self):
        self.root.deiconify()
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(300, lambda: self.root.attributes("-topmost", False))
        if self.app.running:
            self.btn.config(text="停止", bg=CARD2, fg="#FF9C88")
        self._refresh_sub_btn()


# ============================ 主程序 ============================
class App:
    def __init__(self):
        self.conf = load_conf()
        log("conf loaded, keys=%d" % len(self.conf.get("keys", [])))
        self.root = tk.Tk()
        self.root.withdraw()
        log("Tk created")

        self.root.report_callback_exception = self._tk_exc

        self.running = False
        self.engine = None
        self.ui_q = queue.Queue()
        self.work_q = queue.Queue()
        self.worker = None
        self._pending = ""
        self._pending_since = None
        self.tr_cache = {}        # 原文 -> 译文
        self.tr_ms = {}           # 原文 -> 翻译耗时(ms)
        self._fed_sec = 0.0       # 已喂入音频累计秒数（在线时用于算「从发声到上屏」）
        self._last_begin_ms = None  # 当前句在音频里的起始毫秒（云端 fun-asr 提供）
        self.unit_seen = {}       # 单元 -> 首次识别出字的时间（端到端起点）
        self.unit_e2e = {}        # 单元 -> 端到端耗时（同一句共用）
        self._sent_e2e = None     # 当前句的端到端耗时（一句只算一次）
        self._sent_key = None     # 当前句标识（begin_ms 或句序号）
        self._sent_seq = 0        # 句子序号（本地引擎用）
        self.tr_busy = set()      # 正在翻译、还没回来的句子
        self._last_raw = ""       # 最近一次原始累积文本

        self.sub = SubtitleWindow(self.root, self.conf,
                                  on_settings=self.open_settings,
                                  on_hide=self.hide_sub)
        # 延迟徽标是独立窗口（自己管位置/拖动/显隐），与字幕窗互不影响
        self.badge = LatencyBadge(self.root, self.conf)
        log("subtitle window + latency badge built")
        self.settings = Settings(self.root, self.conf, self)
        log("settings built")

        if not self.conf.get("show_sub", True):
            self.sub.win.withdraw()

        self.root.deiconify()
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(400, lambda: self.root.attributes("-topmost", False))
        log("deiconified, state=%s" % self.root.state())

        self.poll()

    def _tk_exc(self, exc, val, tb):
        import traceback
        log("TK CALLBACK: " + "".join(traceback.format_exception(exc, val, tb)))

    def start(self):
        if not active_key(self.conf):
            return False, "没有可用的 key"
        active_key_holder["k"] = active_key(self.conf)
        _sh = self.conf.get("src_lang", "auto")
        if self.conf.get("asr_engine", "cloud") == "local":
            # 本地引擎：sherpa-onnx + 中英 Zipformer（带标点）
            self.stream = LocalAsr(
                self.on_partial, self.on_final, self.on_stream_error,
                self.conf.get("local_model_dir", ""),
                punct_dir=self.conf.get("local_punct_dir", ""))
            log("识别引擎：本地（sherpa-onnx）")
        else:
            self.stream = AsrStream(self.on_partial, self.on_final, self.on_stream_error,
                                    lang_hint=(None if _sh == "auto" else _sh))
            log("识别引擎：在线（百炼）")
        self.stream.start()
        time.sleep(0.4)

        try:
            self.engine = AudioEngine(self.on_audio_chunk, self.on_level,
                                      device=self.conf.get("audio_device") or None)
            self.engine.start()
            time.sleep(0.3)
            if self.engine.device_name is None:
                return False, "抓不到系统声音设备，检查是否接了输出设备"
        except Exception as e:
            return False, "启动音频失败：%s" % e

        # work_loop 是旧的 VAD 分段识别（work_q 从未被填充），已废弃不再启动

        self.sub.show()
        self.sub.set_state("live")
        self.sub.dst_text = "…"
        self.sub.redraw()
        self.settings._refresh_sub_btn()
        self.running = True
        log("start, device=%s" % self.engine.device_name)
        return True, ""

    def stop(self):
        self.running = False
        if self.engine:
            self.engine.stop()
        if getattr(self, "stream", None):
            self.stream.running = False
        self.sub.set_state("idle")
        self.sub.clear_queue()
        self.sub.dst_text = "已停止"
        self.sub.redraw()
        # 停止时收起延迟徽标（它自己管显隐，但程序停了不该残留）
        try:
            self.badge.set_text("")
        except Exception:
            pass

    def hide_sub(self):
        self.sub.hide()
        self.settings._refresh_sub_btn()

    def open_settings(self):
        self.settings.show()

    def on_level(self, db, err):
        if err:
            self.ui_q.put(("level", db, err))
            return
        # 长时间静音时提示一次（多半是系统音量被关了）
        if db < -55:
            self._silent_s = getattr(self, "_silent_s", 0) + 0.1
            if self._silent_s > 10 and not getattr(self, "_silent_warned", False):
                self._silent_warned = True
                self.ui_q.put(("warn", "连续 10 秒没听到声音。如果系统静音了，"
                                       "把音频源换成 VB-Cable 再试（见设置里的说明）。"))
        else:
            self._silent_s = 0
            self._silent_warned = False
        self.ui_q.put(("level", db, None))

    def on_audio_chunk(self, mono):
        try:
            self._fed_sec += len(mono) / float(SR)
        except Exception:
            pass
        if self.stream:
            pcm = (np.clip(mono, -1, 1) * 32767).astype("<i2").tobytes()
            self.stream.feed(pcm)

    def on_partial(self, text, is_end, begin_ms=None):
        self._last_begin_ms = begin_ms
        self.ui_q.put(("partial", text, is_end, begin_ms))

    def on_final(self, text):
        self.ui_q.put(("final", text))

    def on_stream_error(self, msg):
        self.ui_q.put(("error", msg))

    def _translate_sentence(self, sent):
        """后台翻译一句，结果回主线程"""
        if not sent or sent in self.tr_busy or sent in self.tr_cache:
            return
        self.tr_busy.add(sent)

        def work():
            t0 = time.perf_counter()
            try:
                zh = do_translate(sent, active_key(self.conf), self.conf.get("rate", 1),
                                  self.conf.get("dst_lang", "zh"))
                ms = (time.perf_counter() - t0) * 1000
                if zh and zh != sent:
                    self.ui_q.put(("trans", sent, zh, ms))
                else:
                    # 翻译没成功也要有东西显示，退回原文
                    self.ui_q.put(("trans", sent, sent, ms))
            except Exception as e:
                ms = (time.perf_counter() - t0) * 1000
                log("翻译失败：%s" % str(e)[:80])
                self.ui_q.put(("trans", sent, sent, ms))
            finally:
                self.ui_q.put(("trans_done", sent))

        threading.Thread(target=work, daemon=True).start()

    def _cur_e2e(self, now):
        """当前句的端到端耗时。一句只算一次，避免随识别增长反复变化（会闪烁）。"""
        # 句子标识：在线用 begin_ms，本地用句序号
        key = ("b", self._last_begin_ms) if self._last_begin_ms is not None \
              else ("s", self._sent_seq)
        if key != self._sent_key:
            self._sent_key = key
            self._sent_e2e = None
        if self._sent_e2e is None:
            if self._last_begin_ms is not None and self._fed_sec > 0:
                # 在线：真正的「从发声到上屏」
                self._sent_e2e = max(0.0, self._fed_sec - self._last_begin_ms / 1000.0)
            else:
                # 本地：从当前句首个单元首次出字算
                vals = [v for k, v in self.unit_seen.items()]
                self._sent_e2e = max(0.0, now - min(vals)) if vals else 0.0
        return self._sent_e2e

    def _render_stream(self):
        """把最近一次识别文本按句替换成中文后再交给字幕窗"""
        raw = self._last_raw
        if not raw:
            return
        parts = split_units(raw)
        if not parts:
            return

        show_ms = bool(self.conf.get("show_latency", False))
        now = time.time()
        # 记录每个单元「首次识别出字」的时刻（端到端起点），只保留当前句的
        for u in parts:
            if u not in self.unit_seen:
                self.unit_seen[u] = now
        self.unit_seen = {k: v for k, v in self.unit_seen.items() if k in parts}
        self.unit_e2e = {k: v for k, v in self.unit_e2e.items() if k in parts}

        if mostly_chinese(raw):
            # 中文直接转写显示，不经过翻译缓存
            self.badge.set_text("%.1fs" % self._cur_e2e(now) if show_ms else "")
            self.sub.set_stream_text("\n".join(parts))
            return

        # 外语只显示已经有译文的单元；没翻完的不能拿原文顶替
        shown = [self.tr_cache[x] for x in parts if x in self.tr_cache]
        # 延迟单独交给字幕窗绘制，绝不混进文本（混进去会让断行变化 → 字跳动）
        self.badge.set_text(("%.1fs" % self._cur_e2e(now))
                            if (show_ms and shown) else "")

        # 一两个字的碎片先攒着，等下一块一起出现
        merged, pending = [], []
        for x in shown:
            if len(x.strip()) < 3:
                pending.append(x.strip())
            else:
                if pending:
                    merged.append("".join(pending) + x)
                    pending = []
                else:
                    merged.append(x)
        if pending:
            merged.append("".join(pending))
        self.sub.set_stream_text("\n".join(merged))

        # 原文只在「开着双语」时才喂进去
        if self.conf.get("show_source"):
            self.sub.set_source(parts[-1])
        else:
            self.sub.set_source("")

    def work_loop(self):
        while True:
            item = self.work_q.get()
            if item is None:
                break
            mono, seg_start = item
            try:
                key = active_key(self.conf)
                seg_s = len(mono) / SR
                t0 = time.time()
                src = do_asr(wav_data_url(mono), key)
                t_asr = time.time() - t0
                if not src:
                    log("seg %.1fs | ASR 空结果" % seg_s)
                    continue
                t1 = time.time()
                dst = do_translate(src, key, self.conf.get("rate", 1))
                t_mt = time.time() - t1
                lag = time.time() - seg_start
                log("seg %.1fs | asr %.2fs | mt %.2fs | lag %.1fs | %s | %s"
                    % (seg_s, t_asr, t_mt, lag, src[:60], dst[:60]))
                self.ui_q.put(("text", src, dst, lag))
            except urllib.error.HTTPError as e:
                body = ""
                try:
                    body = e.read().decode("utf-8", "ignore")[:120]
                except Exception:
                    pass
                log("HTTP %s %s" % (e.code, body))
                self.ui_q.put(("error", "HTTP %s %s" % (e.code, body)))
            except Exception as e:
                log("ERR %s" % e)
                self.ui_q.put(("error", str(e)[:120]))

    def poll(self):
        if not getattr(self, "_polled", False):
            self._polled = True
            log("first poll tick")
        try:
            while True:
                msg = self.ui_q.get_nowait()
                if msg[0] == "partial":
                    _, text, is_end = msg[0:3]
                    if len(msg) > 3:
                        self._last_begin_ms = msg[3]
                    self._last_raw = text

                    # 识别出来不是中文，就把已完整的句子送去翻
                    if not mostly_chinese(text):
                        # 按单元（而非整句）提交翻译，长句能分块陆续出
                        units = split_units(text)
                        for done in units[:-1]:
                            self._translate_sentence(done)
                        if is_end and units:
                            self._translate_sentence(units[-1])

                    self._render_stream()
                    if is_end:
                        log("ASR [end] %s" % text[-60:])
                        self.sub.finalize()
                        self._sent_seq += 1      # 下一句重新计时
                        self._sent_key = None

                elif msg[0] == "trans":
                    _, src, zh, ms = msg
                    self.tr_cache[src] = zh
                    self.tr_ms[src] = ms
                    self._render_stream()

                elif msg[0] == "trans_done":
                    self.tr_busy.discard(msg[1])
                elif msg[0] == "final":
                    self.sub.show_partial(msg[1])
                    self.sub.commit()
                elif msg[0] == "text":
                    _, src, dst, lag = msg
                    self.sub.push(src, dst, lag)
                elif msg[0] == "error":
                    self.sub.set_state("error")
                    self.settings.status.config(text=msg[1])
                elif msg[0] == "warn":
                    self.settings.status.config(text=msg[1])
                    log("WARN " + msg[1])
                elif msg[0] == "level":
                    _, db, err = msg
                    if err and self.running:
                        self.sub.set_state("error")
                        self.settings.status.config(text="音频：" + err[:100])
        except queue.Empty:
            pass
        self.root.after(60, self.poll)

    def run(self):
        log("entering mainloop")
        self.root.mainloop()
        log("mainloop exited")


if __name__ == "__main__":
    import traceback

    def _hook(t, v, tb):
        log("UNCAUGHT: " + "".join(traceback.format_exception(t, v, tb)))

    sys.excepthook = _hook
    try:
        App().run()
    except Exception:
        log("FATAL: " + traceback.format_exc())
        try:
            messagebox.showerror("字幕助手启动失败",
                                 "详情已写入 session.log\n\n" + traceback.format_exc()[-400:])
        except Exception:
            pass
        sys.exit(1)
