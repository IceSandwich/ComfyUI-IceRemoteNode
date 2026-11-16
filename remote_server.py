import io
import asyncio
import logging
import gc
import comfy.utils
import torch
import folder_paths
import comfy.sd
import comfy.model_management
from aiohttp import web
from server import PromptServer
import functools
import typing
import time
from aiohttp.web import Request, StreamResponse, Response
from .split_clip_to_multigpu import SplitClipToMultiGPU

# 常驻显存的 CLIP 缓存：{(clip_name, clip_type, device_map): split_clip}
# ============================================================
# 全局 registry（CLIP / VAE 共用）
# ============================================================
REMOTE_MODEL_REGISTRY = {}

# 全局推理锁：同一时刻只允许一个请求占用多卡做编码，避免显存/上下文冲突
GPU_INFERENCE_LOCK = asyncio.Lock()

def to_cpu(obj):
    """递归地把张量搬到 CPU 并转换成可序列化结构，用于网络传输"""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().contiguous()
    if isinstance(obj, tuple):
        return tuple(to_cpu(x) for x in obj)
    if isinstance(obj, list):
        return [to_cpu(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    return obj

def clear_model_registry():
    """卸载所有常驻多卡模型，并释放全部 GPU 显存"""
    if not REMOTE_MODEL_REGISTRY:
        return

    logging.info("[IceRemoteNode] 检测到卸载模型指令...")

    for clip in list(REMOTE_MODEL_REGISTRY.values()):
        remove_hooks = getattr(clip, "icern_remove_hooks", None)
        if remove_hooks: remove_hooks()
        cond_stage = getattr(clip, "cond_stage_model", None)
        if cond_stage is not None:
            # 移到 meta 设备彻底断开对 GPU 显存张量的引用
            cond_stage.to("meta")

    REMOTE_MODEL_REGISTRY.clear()
    gc.collect()

    # 逐个设备清空缓存
    for dev_id in range(torch.cuda.device_count()):
        with torch.cuda.device(dev_id):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    logging.info("[IceRemoteNode] 多卡显存已彻底释放！")


# Hook ComfyUI 原生的 unload_all_models，让网页"卸载模型"按钮也能清掉我们的常驻缓存
if not getattr(comfy.model_management, "icern_unload_patched", False):
    _orig_unload_all_models = comfy.model_management.unload_all_models

    def _patched_unload_all_models(*args, **kwargs):
        res = _orig_unload_all_models(*args, **kwargs)
        clear_model_registry()
        return res

    comfy.model_management.unload_all_models = _patched_unload_all_models
    # 注：该标志位保留为注释，沿用原逻辑；如需要防止热重载重复 patch 可以打开下面这行
    comfy.model_management.icern_unload_patched = True


# ============================================================
# 通用装饰器：把同步业务函数包装成二进制 HTTP 端点
# ============================================================

# aiohttp 路由期望的 handler 类型
_AsyncHandler = typing.Callable[[Request], typing.Awaitable[StreamResponse]]

def remote_binary_endpoint(route_method: typing.Literal["GET", "POST"], route_url: str, full_mm_precision: bool = True):
    def inner(func: typing.Callable[[dict], typing.Any]) -> _AsyncHandler:
        """把一个 sync 函数 func(payload: dict) -> 可 torch.save 的对象 包装成 aiohttp handler。

        公共管道：
        1. 读 request 原始字节 -> torch.load 成 payload
        2. 加 GPU_INFERENCE_LOCK 串行化
        3. 在 executor 里以 inference_mode 执行 func(payload)
        4. torch.save(func 返回值) -> application/octet-stream 响应
        """
        @functools.wraps(func)
        def _sync_runner(payload):
            with torch.inference_mode():
                return func(payload)

        async def handler(request: web.Request):
            logging.info(f"[IceRemoteNode] {route_method} {route_url}")
            try:
                raw_bytes = await request.read()
                payload = torch.load(
                    io.BytesIO(raw_bytes), map_location="cpu", weights_only=False
                )
            except Exception as e:
                logging.exception(f"[IceRemoteNode] {func.__name__} 反序列化失败")
                return web.json_response({"error": str(e)}, status=400)

            default_precision = torch.get_float32_matmul_precision()
            if full_mm_precision:
                # 强制使用纯 FP32 计算，确保跨机器行为一致
                torch.set_float32_matmul_precision("highest")
            t0 = time.perf_counter()
            try:
                async with GPU_INFERENCE_LOCK:
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(None, _sync_runner, payload)
            except Exception as e:
                logging.exception(f"[IceRemoteNode] {func.__name__} 执行失败")
                return web.json_response({"error": str(e)}, status=500)
            finally:
                torch.set_float32_matmul_precision(default_precision)
            elapsed = time.perf_counter() - t0
            logging.info(f"[IceRemoteNode] costs {elapsed * 1000:.1f} ms")

            buf_out = io.BytesIO()
            torch.save(to_cpu(result), buf_out)
            resp = web.Response(
                body=buf_out.getvalue(),
                content_type="application/octet-stream",
            )
            resp.enable_compression(web.ContentCoding.gzip)
            return resp

        return handler
    return inner

# ============================================================
# 路由: 远程CLIP
# ============================================================

def load_and_split_clip_sync(clip_name: str, clip_type_str: str, device_map: str):
    """首次请求时自动加载 CLIP，并按 device_map 切分到多张 GPU"""
    logging.info(
        f"[IceRemoteNode] 首次请求触发自动加载: {clip_name} "
        f"(type={clip_type_str}, device_map={device_map})..."
    )

    clip_path = folder_paths.get_full_path_or_raise("clip", clip_name)
    clip_type = getattr(
        comfy.sd.CLIPType, clip_type_str.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION
    )

    # 显式让 ComfyUI 在 CPU 上加载（避免默认调度提前把权重塞进 GPU）
    clip = comfy.sd.load_clip(
        ckpt_paths=[clip_path],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=clip_type,
        model_options={
            "load_device": torch.device("cpu"),
            "offload_device": torch.device("cpu"),
        },
    )

    # 交给切分节点按 device_map 分卡并注册钩子
    splitter = SplitClipToMultiGPU()
    (split_clip,) = splitter.apply_dual_gpu_split(clip, device_map)

    logging.info("[IceRemoteNode] 模型已自动加载并完成多卡切分常驻！")
    return split_clip

def get_or_load_clip_from_cache(clip_name: str, clip_type: str, device_map: str) -> comfy.sd.CLIP:
    # 缓存键包含 device_map：不同切分方案会缓存成不同的常驻实例
    cache_key = (clip_name, clip_type, device_map)
    cache_hit = cache_key in REMOTE_MODEL_REGISTRY
    logging.info(f"[IceRemoteNode] load clip, cache_hit={cache_hit} key={cache_key}")
    if not cache_hit:
        REMOTE_MODEL_REGISTRY[cache_key] = load_and_split_clip_sync(
            clip_name, clip_type, device_map
        )
    clip = REMOTE_MODEL_REGISTRY[cache_key]
    return clip

@PromptServer.instance.routes.post("/icern/clip__encode_from_tokens_scheduled")
@remote_binary_endpoint("POST", "/icern/clip__encode_from_tokens_scheduled")
def remote_clip_encode_from_tokens_scheduled(payload):
    """远程 token 编码入口：
    反序列化请求参数 -> 加载/复用多卡模型 -> 编码 -> 把结果搬到 CPU 返回
    """
    clip_name = payload["clip_name"]
    clip_type = payload.get("clip_type", "minimax")
    device_map = payload.get("device_map", "cuda:0@0-40,cuda:1")
    call_args = payload["args"]
    call_kwargs = payload["kwargs"]

    clip = get_or_load_clip_from_cache(clip_name, clip_type, device_map)
    # 多卡切分节点已内置了 per-layer 的 dtype/设备对齐钩子，无需再外套 autocast
    res = clip.encode_from_tokens_scheduled(*call_args, **call_kwargs)
    return res

def get_or_load_vae_from_cache(vae_path: str, device: str) -> comfy.sd.VAE:
    cache_key = (vae_path, "vae", device)
    cache_hit = cache_key in REMOTE_MODEL_REGISTRY
    logging.info(f"[IceRemoteNode] load vae, cache_hit={cache_hit} key={cache_key}")
    if not cache_hit:
        full_path = folder_paths.get_full_path_or_raise("vae", vae_path)
        sd, metadata = comfy.utils.load_torch_file(full_path, return_metadata=True)
        REMOTE_MODEL_REGISTRY[cache_key] = comfy.sd.VAE(sd=sd, metadata=metadata, device=torch.device(device))
    vae = REMOTE_MODEL_REGISTRY[cache_key]
    return vae

@PromptServer.instance.routes.get("/icern/vae__getattr")
async def remote_vae_getattr(request):
    logging.info("[IceRemoteNode] GET /icern/vae__getattr")
    try:
        data     = await request.json()
        device   = data["device"]
        vae_path = data["vae_path"]
        attrname = data["attr_name"]

        vae = get_or_load_vae_from_cache(vae_path, device)
        attr = getattr(vae, attrname)
        return web.json_response({"value": attr})
    except Exception as e:
        logging.exception("[VAE] load failed")
        return web.json_response({"error": str(e)}, status=500)

@PromptServer.instance.routes.post("/icern/vae__encode")
@remote_binary_endpoint("POST", "/icern/vae__encode")
def remote_vae_encode(payload):
    """远程 VAE 编码入口：
    payload = {"vae_path": str, "device": str, "args": tuple, "kwargs": dict}
    """
    vae_path = payload["vae_path"]
    device   = payload["device"]
    call_args   = payload.get("args", ())
    call_kwargs = payload.get("kwargs", {})

    vae = get_or_load_vae_from_cache(vae_path, device)
    res = vae.encode(*call_args, **call_kwargs)
    return res

@PromptServer.instance.routes.post("/icern/vae__decode")
@remote_binary_endpoint("POST", "/icern/vae__decode", full_mm_precision=False)
def remote_vae_decode(payload):
    vae_path    = payload["vae_path"]
    device      = payload["device"]
    call_args   = payload.get("args", ())
    call_kwargs = payload.get("kwargs", {})

    vae = get_or_load_vae_from_cache(vae_path, device)
    res = vae.decode(*call_args, **call_kwargs)
    return res

# Re-use the exact helpers/constants the built-in H3 nodes use.
from comfy_extras.nodes_minimax_h3 import (
    _empty_av_latent,
    _resize,
    _encode_ref_audio,
    adapt_canvas,
    CANVAS_MULTIPLE,
    REF_IMAGE_SHORT_EDGE,
    FPS,
)
import math
@PromptServer.instance.routes.post("/icern/minimaxh3__image_and_reference_to_video")
@remote_binary_endpoint("POST", "/icern/minimaxh3__image_and_reference_to_video")
def remote_minimaxh3_image_and_reference_to_video(payload):
    clip_icern_info = payload["clip"]
    clip = get_or_load_clip_from_cache(
        clip_icern_info["clip_name"],
        clip_icern_info["clip_type"],
        clip_icern_info["device_map"]
    )
    vae_icern_info = payload["vae"]
    vae = get_or_load_vae_from_cache(
        vae_icern_info["vae_name"],
        vae_icern_info["device"]
    )
    audio_vae_icern_info = payload["audio_vae"]
    audio_vae = get_or_load_vae_from_cache(
        audio_vae_icern_info["vae_name"],
        audio_vae_icern_info["device"]
    )

    prompt = payload["prompt"]
    width = payload["width"]
    height = payload["height"]
    length = payload["length"]
    first_frame = payload["first_frame"]
    last_frame = payload["last_frame"]
    ref_image_size = payload["ref_image_size"]
    ref_images = payload["ref_images"]
    ref_videos = payload["ref_videos"]
    ref_video_audios = payload["ref_video_audios"]
    ref_audios = payload["ref_audios"]

    latent, frame_count = _empty_av_latent(width, height, length)

    # ---------- fl2va: first/last keyframes ----------
    images = []
    keyframes = []
    if first_frame is not None:
        # geometry anchor: plain stretch to canvas
        img = _resize(first_frame[:1], width, height, "disabled")
        images.append(img)
        keyframes.append({"resolved_frame_index": 0, "image": img})
    if last_frame is not None:
        # follower: aspect-preserving cover-crop
        img = _resize(last_frame[:1], width, height, "center")
        images.append(img)
        keyframes.append({"resolved_frame_index": frame_count - 1, "image": img})

    # ---------- ref2va: images / videos / audio references ----------
    ref_items = []   # for the tokenizer presentation, in request order
    ref_blocks = []  # for the DiT payload, same order

    for img in (ref_images or {}).values():
        if img is None:
            continue
        h, w = img.shape[1], img.shape[2]
        if ref_image_size == "match":
            scale = min(1.0, math.sqrt((width * height) / (w * h)))
        else:
            scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
        tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        resized = _resize(img[:1], tw, th, "disabled")
        z = vae.encode(resized)
        ref_items.append({"type": "image", "data": resized})
        ref_blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": z})

    ref_video_audios = ref_video_audios or {}
    for name, video_frames in (ref_videos or {}).items():
        if video_frames is None:
            continue
        soundtrack = ref_video_audios.get("ref_video_audio_" + name.rsplit("_", 1)[-1])
        vh, vw = video_frames.shape[1], video_frames.shape[2]
        cw, ch = adapt_canvas(vw, vh)
        if vw * vh < cw * ch:
            cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        frames = _resize(video_frames, cw, ch, "disabled")
        if frames.shape[0] > frame_count:
            frames = frames[:frame_count]
        n = frames.shape[0]
        if n < 5:
            raise ValueError("MiniMax H3 reference videos need at least 5 frames (~0.2s at 24 fps)")
        while n % 17 != 5:
            n -= 1
        frames = frames[:n]
        z = vae.encode(frames)
        audio_latent, ref_audio_t = (None, 0)
        if soundtrack is not None:
            audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, soundtrack)
            ref_items.append({"type": "audio"})
        sample_idx = list(range(0, frames.shape[0], FPS // 2))
        qwen_frames = frames[sample_idx]
        ref_items.append({
            "type": "video", "data": qwen_frames,
            "timestamps": [i / 2.0 for i in range(len(sample_idx))],
        })
        ref_blocks.append({
            "kind": "video_audio" if ref_audio_t else "video",
            "latent_t": z.shape[2], "latent_h": ch // 16, "latent_w": cw // 16,
            "ref_audio_t": ref_audio_t, "latent": z, "audio_latent": audio_latent,
        })

    for audio in (ref_audios or {}).values():
        if audio is None:
            continue
        audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, audio)
        ref_items.append({"type": "audio"})
        ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t, "audio_latent": audio_latent})

    # ---------- tokenize + encode (both keyframe images and refs at once) ----------
    # Keep reference numbering stable, then include keyframes in the vision context.
    encoder_items = ref_items + [{"type": "image", "data": image} for image in images]
    tokens = clip.tokenize(prompt, minimax_ref_items=encoder_items)
    cond = clip.encode_from_tokens_scheduled(tokens)

    ret_payload = {}
    if keyframes:
        for kf in keyframes:
            kf["latent"] = vae.encode(kf.pop("image"))
        ret_payload["minimax_keyframes"] = keyframes
    if ref_blocks:
        ret_payload["minimax_refs"] = ref_blocks

    ret_payload["icern_cond"] = cond
    return ret_payload

@PromptServer.instance.routes.post("/icern/minimaxh3__vae_decode_video")
@remote_binary_endpoint("POST", "/icern/minimaxh3__vae_decode_video", full_mm_precision=False)
def remote_minimaxh3_vae_decode_video(payload):
    vae_icern_info = payload["vae"]
    vae = get_or_load_vae_from_cache(
        vae_icern_info["vae_name"],
        vae_icern_info["device"]
    )
    audio_vae_icern_info = payload["audio_vae"]
    audio_vae = get_or_load_vae_from_cache(
        audio_vae_icern_info["vae_name"],
        audio_vae_icern_info["device"]
    )
    samples = payload["samples"]

    # From nodes.py:VAEDecode.decode
    latent = samples["samples"]
    assert latent.is_nested
    img_latent = latent.unbind()[0]
    audio_latent = latent.unbind()[1]

    images = vae.decode(img_latent)
    if len(images.shape) == 5: #Combine batches
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    ret_images = images

    # From nodes_audio.py:vae_decode_audio
    audio = audio_vae.decode(audio_latent).movedim(-1, 1)

    std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    audio /= std
    vae_sample_rate = getattr(audio_vae, "audio_sample_rate_output", getattr(audio_vae, "audio_sample_rate", 44100))
    logging.info(f"Using sample rate: {vae_sample_rate}")
    ret_audio = {"waveform": audio, "sample_rate": vae_sample_rate if "sample_rate" not in samples else samples["sample_rate"]}

    # direct send images and audio require ~500MiB for 5s 640x640 video
    # so we convert to video to send
    fps = payload["fps"]
    bit_depth = payload["bit_depth"]
    saved_metadata = payload["saved_metadata"]
    from comfy_api.latest import InputImpl, Types
    from fractions import Fraction
    video_component = InputImpl.VideoFromComponents(
        Types.VideoComponents(images=ret_images, audio=ret_audio, frame_rate=Fraction(fps)),
        bit_depth=bit_depth,
    )
    width, height = video_component.get_dimensions()
    buf = io.BytesIO()
    video_component.save_to(buf, metadata=saved_metadata)
    data = buf.getvalue()
    
    return {
        "width": width,
        "height": height,
        "data": data,
    }

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
