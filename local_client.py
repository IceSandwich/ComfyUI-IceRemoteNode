import io as Bio
import types
import time
import torch
import requests
import comfy.hooks
import comfy.sd
import comfy.sd1_clip
import comfy.text_encoders.minimax
import comfy.model_patcher
import comfy.model_management
from dataclasses import dataclass
from comfy_api.latest import io
from comfy_api.latest._util.video_types import VideoCodec, VideoContainer
import nodes
import node_helpers
import logging

class PatchedCondStageModel:
    def reset_clip_options(self, *args, **kwargs):
        pass

    def set_clip_options(self, *args, **kwargs):
        pass

def to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu().contiguous()
    if isinstance(obj, tuple):
        return tuple(to_cpu(x) for x in obj)
    if isinstance(obj, list):
        return [to_cpu(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    return obj

class TinyDummyModule(torch.nn.Module):
    """仅占 4 字节的合法 PyTorch 模块，用于走通原生 ModelPatcher 全部状态机"""
    def __init__(self):
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.zeros(1, dtype=torch.float16), requires_grad=False)

class MiniMaxH3ClipModel(comfy.sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, model_options={}):
        super().__init__(device=device, layer="last", layer_idx=None, textmodel_json_config={},
                         dtype=dtype, special_tokens={"pad": 151643}, layer_norm_hidden_state=False,
                         model_class=TinyDummyModule, enable_attention_masks=False,
                         return_attention_masks=False, model_options=model_options)

class PatchedTEModel(comfy.sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, name="qwen3vl_32b",
                         clip_model=MiniMaxH3ClipModel, model_options=model_options)


# ---------- 张量体积工具（定位"是谁把请求撑大了"） ----------

def tensor_nbytes(obj) -> int:
    """递归统计 payload 里所有张量的字节数（不含 Python 对象/序列化开销）。"""
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, (list, tuple)):
        return sum(tensor_nbytes(x) for x in obj)
    if isinstance(obj, dict):
        return sum(tensor_nbytes(v) for v in obj.values())
    return 0


# ---------- 传输统计 ----------

@dataclass
class TransferStat:
    url: str
    req_bytes: int          # torch.save 后的实际字节数
    resp_bytes: int         # 响应 body 字节数
    elapsed: float          # 秒
    decompressed_resp_bytes: bool
    payload_tensor_bytes: int = 0   # 仅统计 payload 里张量的原始字节

    @property
    def req_mib(self):  return self.req_bytes / 1024 ** 2
    @property
    def resp_mib(self): return self.resp_bytes / 1024 ** 2

    def log(self):
        # total = self.req_bytes + self.resp_bytes
        # bw = total / max(self.elapsed, 1e-6) / 1024 ** 2
        logging.info(
            f"[{self.url}] "
            f"↑ {self.req_mib:.2f} MiB  ↓ {self.resp_mib:.2f} MiB{'(decompressed)' if self.decompressed_resp_bytes else ''}  "
            f"{self.elapsed * 1000:.1f} ms  "
            # f"~{bw:.1f} MiB/s"
            + (f"  (tensor_raw={self.payload_tensor_bytes / 1024**2:.2f} MiB)"
               if self.payload_tensor_bytes else "")
        )

import gzip
def gzip_compress(data: bytes) -> bytes:
    """用 gzip 压缩字节数据"""
    buf = Bio.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as f:
        f.write(data)
    return buf.getvalue()

def request(url: str, payload: dict, timeout_seconds: int, verbose: bool = True):
    # 1. 序列化为字节
    buf_out = Bio.BytesIO()
    torch.save(to_cpu(payload), buf_out)
    raw_body = buf_out.getvalue()

    # 2. 压缩（建议对过小数据跳过压缩，因为压缩本身有开销）
    if len(raw_body) > 10 * 1024 * 1024:  # 阈值可调
        body = gzip_compress(raw_body)
        headers = {
            "Content-Type": "application/octet-stream",
            "Content-Encoding": "gzip",   # ⚠️ 关键：必须显式声明
        }
    else:
        body = raw_body
        headers = {"Content-Type": "application/octet-stream"}

    t0 = time.perf_counter()
    res = requests.post(
        url,
        data=body,
        headers=headers,
        timeout=(3, timeout_seconds)
    )
    if res.status_code != 200:
        raise RuntimeError(f"B 机器远程编码失败 ({res.status_code}): {res.text}")
    elapsed = time.perf_counter() - t0

    # logging.info(f"Content-Encoding: {res.headers.get('Content-Encoding')}")   # 期望 gzip
    # logging.info(f"Transfer-Encoding: {res.headers.get('Transfer-Encoding')}") # 期望 chunked
    # logging.info(f"Content-Length: {res.headers.get('Content-Length')}")       # 压缩后会变 None
    resp_bytes = res.headers.get("Content-Length", None)
    decompressed_resp_bytes = False
    if resp_bytes is None:
        resp_bytes = str(len(res.content))
        decompressed_resp_bytes = True

    if verbose:
        stat = TransferStat(
            url=url,
            req_bytes=len(body),
            resp_bytes=int(resp_bytes),          # 服务端没开 gzip，等价于 Content-Length
            elapsed=elapsed,
            decompressed_resp_bytes=decompressed_resp_bytes,
            payload_tensor_bytes=0,
        )
        stat.log()

    ret = torch.load(Bio.BytesIO(res.content), map_location="cpu", weights_only=False)
    return ret

class RemoteCLIPLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server_url": ("STRING", {"default": "http://127.0.0.1:8189"}),
                "server_clip_name": ("STRING", {"default": "dir/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"}),
                "server_clip_type": ("STRING", {"default": "minimax"}),
                "server_device_map": ("STRING", {"default": "cuda:0@0-40,cuda:1"}),
                "timeout_seconds": ("INT", {"default": 300, "min": 1, "max": 3600}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "create_remote_clip"
    CATEGORY = "IceRemoteNode"

    def create_remote_clip(self, server_url, server_clip_name, server_clip_type, server_device_map, timeout_seconds):
        server_url = server_url.rstrip("/")

        # # 1. 创建零权重空壳 CLIP 对象（不读硬盘、不占内存、不占显存）
        # clip = comfy.sd.CLIP(no_init=True)
        # clip.tokenizer_options = {}
        
        # # 2. 仅初始化官方内置的 MiniMax Tokenizer（词表内置在 ComfyUI 源码中）
        # tokenizer_cls = getattr(comfy.text_encoders.minimax, "MiniMaxH3Tokenizer", None) or getattr(comfy.text_encoders.minimax, "MiniMaxTokenizer")
        # clip.tokenizer = tokenizer_cls()
        # clip.cond_stage_model = PatchedCondStageModel()
        # clip.patcher = DummyModelPatcher()
        # clip.layer_idx = None
        # clip.use_clip_schedule = False

        # 3. 拦截本地模型加载请求
        # clip.load_model = lambda *args, **kwargs: None
        # 1. 创建零权重 CLIP 实例，使用原生 ModelPatcher 包装 4 字节真实模块
        #    完全兼容 Dynamic VRAM、模型预取队列与卸载机制，无需 --disable-dynamic-vram
        
        clip = comfy.sd.CLIP(no_init=True)
        clip.tokenizer_options = {}
        clip.cond_stage_model = TinyDummyModule()
        clip.patcher = comfy.model_patcher.ModelPatcher(
            clip.cond_stage_model,
            load_device=comfy.model_management.text_encoder_device(),
            offload_device=comfy.model_management.text_encoder_offload_device(),
        )
        clip.patcher.set_model_compute_dtype(torch.float32)
        clip.patcher.hook_mode = comfy.hooks.EnumHookMode.MinVram
        clip.patcher.is_clip = True
        clip.apply_hooks_to_conds = None
        clip.layer_idx = None
        clip.use_clip_schedule = False

        # 2. 初始化官方内置的 MiniMax Tokenizer
        tokenizer_cls = getattr(comfy.text_encoders.minimax, "MiniMaxH3Tokenizer", None) or getattr(comfy.text_encoders.minimax, "MiniMaxTokenizer")
        clip.tokenizer = tokenizer_cls()

        # 4. 绑定远程编码方法，直接将分词后的 tokens 发给 B 机器计算
        def remote_encode_from_tokens_scheduled(self, *args, **kwargs):
            payload = {
                "clip_name": server_clip_name,
                "clip_type": server_clip_type,
                "device_map": server_device_map,
                "args": args,
                "kwargs": kwargs,
            }

            res = request(
                f"{server_url}/icern/clip__encode_from_tokens_scheduled",
                payload,
                timeout_seconds=timeout_seconds
            )

            # buf_out = io.BytesIO()
            # torch.save(payload, buf_out)

            # res = requests.post(
            #     ,
            #     data=buf_out.getvalue(),
            #     headers={"Content-Type": "application/octet-stream"},
            #     timeout=(3, timeout_seconds)
            # )
            # if res.status_code != 200:
            #     raise RuntimeError(f"B 机器远程编码失败 ({res.status_code}): {res.text}")

            # def to_cuda0(obj):
            #     if isinstance(obj, torch.Tensor):
            #         return obj.to("cuda:0").contiguous()
            #     if isinstance(obj, tuple):
            #         return tuple(to_cuda0(x) for x in obj)
            #     if isinstance(obj, list):
            #         return [to_cuda0(x) for x in obj]
            #     if isinstance(obj, dict):
            #         return {k: to_cuda0(v) for k, v in obj.items()}
            #     return obj

            # ret = torch.load(io.BytesIO(res.content), map_location="cpu", weights_only=False)
            # ret = to_cuda0(ret)
            logging.info(f"[IceRemoteNode] 获得prompt")
            return res

        clip.encode_from_tokens_scheduled = types.MethodType(remote_encode_from_tokens_scheduled, clip)
        clip.icern_info = {
            "url": server_url,
            "clip_name": server_clip_name,
            "clip_type": server_clip_type,
            "device_map": server_device_map,
        }
        logging.info(f"[IceRemoteNode] 零权重远程 CLIP 初始化成功 -> 目标服务器: {server_url}")
        return (clip,)

class RemoteVAELoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server_url": ("STRING", {"default": "http://127.0.0.1:8189"}),
                "server_vae_name": ("STRING", {"default": "dir/vae/minimax_h3_video_vae_int8_convrot.safetensors"}),
                "server_device": ("STRING", {"default": "cuda:1"}),
                "timeout_seconds": ("INT", {"default": 300, "min": 1, "max": 3600}),
            }
        }
    
    RETURN_TYPES = ("VAE",)
    FUNCTION = "create_remote_vae"
    CATEGORY = "IceRemoteNode"

    def create_remote_vae(self, server_url, server_vae_name, server_device, timeout_seconds):
        server_url = server_url.rstrip("/")

        class RemoteVAE:
            def __init__(self):
                self.first_stage_model = TinyDummyModule()
                self.patcher = comfy.model_patcher.ModelPatcher(
                    self.first_stage_model,
                    load_device="cpu",
                    offload_device="cpu",
                )
                self.icern_info = {
                    "url": server_url,
                    "vae_name": server_vae_name,
                    "device": server_device,
                }

            def __getattr__(self, name):
                if name == "audio_sample_rate":
                    res = requests.get(
                        f"{server_url}/icern/vae__getattr",
                        json={
                            "vae_path": server_vae_name,
                            "device": server_device,
                            "attr_name": name,
                        },
                        timeout=(3, timeout_seconds)
                    )
                    if res.status_code != 200:
                        raise RuntimeError(f"B 机器远程编码失败 ({res.status_code}): {res.text}")
                    return res.json()["value"]
                raise AttributeError(
                    f"'{type(self).__name__}' object has no attribute '{name}'"
                )

            def encode(self, *args, **kwargs):
                payload = {
                    "vae_path": server_vae_name,
                    "device": server_device,
                    "args": to_cpu(args),
                    "kwargs": to_cpu(kwargs),
                }
                
                ret = request(
                    f"{server_url}/icern/vae__encode",
                    payload,
                    timeout_seconds
                )

                return ret
            
            def decode(self, *args, **kwargs):
                payload = {
                    "vae_path": server_vae_name,
                    "device": server_device,
                    "args": to_cpu(args),
                    "kwargs": to_cpu(kwargs),
                }
                
                ret = request(
                    f"{server_url}/icern/vae__decode",
                    payload,
                    timeout_seconds
                )

                return ret
            
        vae = RemoteVAE()
        return (vae, )

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

class RemoteMiniMaxH3ImageAndReferenceToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RemoteMiniMaxH3ImageAndReferenceToVideo",
            display_name="Remote/Local MiniMax H3 Image + Reference to Video",
            category="IceRemoteNode",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.Int.Input("timeout_seconds", default=300),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input(
                    "length", default=124, min=5, max=3600, step=17,
                    tooltip="Frame count at 24 fps, snapped up to the model's 17k+5 grid "
                            "(124 = ~5s; trained range is ~124-362, longer is untested)",
                ),

                # --- from MiniMaxH3ImageToVideo (fl2va) ---
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),

                # --- from MiniMaxH3ReferenceToVideo (ref2va) ---
                io.Combo.Input(
                    "ref_image_size", options=["match", "max"], default="match",
                    tooltip="Reference image sizing. 'match' scales each ref (down only, "
                            "keeping aspect) to the generation's pixel area; 'max' uses the "
                            "reference pipeline's 2048px short edge for best identity "
                            "fidelity. Reference tokens ride through every sampling step, "
                            "so 'max' can be several times slower.",
                ),
                io.Autogrow.Input(
                    "ref_images", optional=True,
                    template=io.Autogrow.TemplateNames(
                        input=io.Image.Input(
                            "ref_image",
                            tooltip="Reference image (downscaled to 2048 short edge if "
                                    "larger, never upscaled)",
                        ),
                        names=[f"ref_image_{i}" for i in range(1, 11)], # 1 through 10
                        min=0, 
                    ),
                ),
                io.Autogrow.Input(
                    "ref_videos", optional=True,
                    template=io.Autogrow.TemplateNames(
                        input=io.Image.Input(
                            "ref_video", tooltip="Reference video frames at 24 fps (2-15s)"
                        ),
                        names=[f"ref_video_{i}" for i in range(1, 5)], # 1 through 4
                        min=0,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_video_audios", optional=True,
                    template=io.Autogrow.TemplateNames(
                        input=io.Audio.Input(
                            "ref_video_audio",
                            tooltip="Soundtrack of the same-numbered reference video",
                        ),
                        names=[f"ref_video_audio_{i}" for i in range(1, 5)], # 1 through 4
                        min=0,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_audios", optional=True,
                    template=io.Autogrow.TemplateNames(
                        input=io.Audio.Input("ref_audio", tooltip="Standalone reference audio"),
                        names=[f"ref_audio_{i}" for i in range(1, 5)], # 1 through 4
                        min=0,
                    ),
                ),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()],
        )
    
    @classmethod
    def execute_local(
        cls, clip, vae, audio_vae, prompt, width, height, length,
        first_frame=None, last_frame=None,
        ref_image_size="match",
        ref_images=None, ref_videos=None, ref_video_audios=None, ref_audios=None,
    ) -> io.NodeOutput:
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

        payload = {}
        if keyframes:
            for kf in keyframes:
                kf["latent"] = vae.encode(kf.pop("image"))
            payload["minimax_keyframes"] = keyframes
        if ref_blocks:
            payload["minimax_refs"] = ref_blocks
        if payload:
            cond = node_helpers.conditioning_set_values(cond, payload)

        return io.NodeOutput(cond, latent)
    
    @classmethod
    def execute_remote(
        cls, clip_icern_info, vae_icern_info, audio_icern_info, server_url, timeout_seconds, prompt, width, height, length,
        first_frame=None, last_frame=None,
        ref_image_size="match",
        ref_images=None, ref_videos=None, ref_video_audios=None, ref_audios=None,
    ) -> io.NodeOutput:
        payload = {
            "clip": clip_icern_info,
            "vae": vae_icern_info,
            "audio_vae": audio_icern_info,
            "prompt": prompt,
            "width": width,
            "height": height,
            "length": length,
            "first_frame": first_frame,
            "last_frame": last_frame,
            "ref_image_size": ref_image_size,
            "ref_images": ref_images,
            "ref_videos": ref_videos,
            "ref_video_audios": ref_video_audios,
            "ref_audios": ref_audios,
        }

        ret = request(
            f"{server_url}/icern/minimaxh3__image_and_reference_to_video",
            payload,
            timeout_seconds=timeout_seconds
        )

        latent, frame_count = _empty_av_latent(width, height, length)
        cond = ret.pop("icern_cond")
        if ret:
            cond = node_helpers.conditioning_set_values(cond, ret)
        return io.NodeOutput(cond, latent)

    @classmethod
    def execute(
        cls, clip, vae, timeout_seconds, audio_vae, prompt, width, height, length,
        first_frame=None, last_frame=None,
        ref_image_size="match",
        ref_images=None, ref_videos=None, ref_video_audios=None, ref_audios=None,
    ) -> io.NodeOutput:
        clip_icern_info = getattr(clip, "icern_info", None)
        vae_icern_info = getattr(vae, "icern_info", None)
        audio_icern_info = getattr(audio_vae, "icern_info", None)
        if clip_icern_info is None and vae_icern_info is None and audio_icern_info is None:
            return cls.execute_local(clip, vae, audio_vae, prompt, width, height, length, first_frame, last_frame, ref_image_size, ref_images, ref_videos, ref_video_audios)
        if clip_icern_info is not None and vae_icern_info is not None and audio_icern_info is not None:
            server_url = clip_icern_info["url"]
            if server_url != vae_icern_info["url"] or server_url != audio_icern_info["url"]:
                raise ValueError(f"Require a matched remote url: clip({server_url}), vae({vae_icern_info['url']}), audio_vae({vae_icern_info['url']})")
            return cls.execute_remote(clip_icern_info, vae_icern_info, audio_icern_info, server_url, timeout_seconds, prompt, width, height, length, first_frame, last_frame, ref_image_size, ref_images, ref_videos, ref_video_audios)
        raise ValueError(f"Require clip/vae/audio_vae are all remote or all local")


from comfy_api.latest._input_impl.video_types import VideoFromComponents
from fractions import Fraction
class DecodedVideoComponent(VideoFromComponents):
    def __init__(self, data: bytes, width: int, height: int, frame_rate: Fraction):
        super().__init__(None)
        self.data = data
        self.width = width
        self.height = height
        self.frame_rate = frame_rate
    
    def get_dimensions(self) -> tuple[int, int]:
        return (self.width, self.height)
    
    def save_to(self, path: str, format: VideoContainer = VideoContainer.AUTO, codec: VideoCodec = VideoCodec.AUTO, metadata: dict | types.NoneType = None, bit_depth: int | types.NoneType = None, crf: float | types.NoneType = None, color_space: str | types.NoneType = None, preset: str | types.NoneType = None, chroma_subsampling: str | types.NoneType = None):
        if format != VideoContainer.AUTO or codec != VideoCodec.AUTO:
            logging.warning("[IceRemoteNode] format and codec are ignored when using remote vae decode video")
        with open(path, "wb") as f:
            f.write(self.data)

class RemoteVAEDecodeVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RemoteVAEDecodeVideo",
            search_aliases=["latent to video"],
            display_name="Remote VAE Decode Video",
            category="IceRemoteNode",
            inputs=[
                io.Latent.Input("samples"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.Float.Input("fps", default=30.0, min=1.0, max=120.0, step=1.0),
                io.Combo.Input(
                    "bit_depth",
                    options=["auto", 8, 10],
                    default="auto",
                    tooltip="Auto uses 8-bit for sRGB and 10-bit for HDR. Explicit 8-bit and 10-bit choices are independent of colorspace.",
                    optional=True,
                ),
                io.Int.Input("timeout_seconds", default=300),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            outputs=[
                io.Video.Output(),
            ],
        )

    @classmethod
    def execute(cls, samples, vae, audio_vae, fps, bit_depth, timeout_seconds) -> io.NodeOutput:
        vae_icern_info = getattr(vae, "icern_info", None)
        if vae_icern_info is None: raise ValueError(f"Require a remote vae")
        audio_icern_info = getattr(audio_vae, "icern_info", None)
        if audio_icern_info is None: raise ValueError(f"Require a remote audio vae")
        server_url = vae_icern_info["url"]
        if server_url != audio_icern_info["url"]:
            raise ValueError(f"Require a matched remote url: vae({vae_icern_info['url']}), audio_vae({vae_icern_info['url']})")


        saved_metadata = None
        from comfy.cli_args import args
        if not args.disable_metadata:
            metadata = {}
            if cls.hidden.extra_pnginfo is not None:
                metadata.update(cls.hidden.extra_pnginfo)
            if cls.hidden.prompt is not None:
                metadata["prompt"] = cls.hidden.prompt
            if len(metadata) > 0:
                saved_metadata = metadata
        # logging.info(f"GOT metadata for video: {saved_metadata}")

        payload = {
            "samples": samples,
            "vae": vae_icern_info,
            "audio_vae": audio_icern_info,
            "fps": fps,
            "bit_depth": bit_depth,
            "saved_metadata": saved_metadata,
        }
        ret = request(
            f"{server_url}/icern/minimaxh3__vae_decode_video",
            payload,
            timeout_seconds,
        )

        return io.NodeOutput(
            DecodedVideoComponent(
                ret["data"],
                ret["width"],
                ret["height"],
                Fraction(fps),
            )
        )

import json
import folder_paths
from PIL.PngImagePlugin import PngInfo
from PIL import Image
import numpy as np
class SaveImageToRemoteDisk():
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "server_url": ("STRING", {"default": "http://127.0.0.1:8189"}),
                "server_filename_prefix": ("STRING", {
                    "default": "%date:yyyy-MM-dd%/Comfy",
                    "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."
                }),
                "timeout_seconds": ("INT", {"default": 300, "min": 1, "max": 3600}),
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "IceRemoteNode"

    def save_images(self, images, server_url, server_filename_prefix="ComfyUI", timeout_seconds=300, prompt=None, extra_pnginfo=None):
        metadata = None
        from comfy.cli_args import args
        if not args.disable_metadata:
            metadata = PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))

        saved = []
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            buf = Bio.BytesIO()
            img.save(buf, format="png", pnginfo=None, compress_level=self.compress_level)
            saved.append(buf.getvalue())

        payload = {
            "images": saved,
            "filename_prefix": server_filename_prefix,
            "width0": images[0].shape[1],
            "height0": images[0].shape[0],
            "metadata": metadata,
            "compress_level": self.compress_level,
        }

        request(
            f"{server_url}/icern/save__image_to_disk",
            payload,
            timeout_seconds,
        )
        
        return (images,)

NODE_CLASS_MAPPINGS = {
    "RemoteCLIPLoader": RemoteCLIPLoader,
    "RemoteVAELoader": RemoteVAELoader,
    "RemoteMiniMaxH3ImageAndReferenceToVideo": RemoteMiniMaxH3ImageAndReferenceToVideo,
    "RemoteVAEDecodeVideo": RemoteVAEDecodeVideo,
    "SaveImageToRemoteDisk": SaveImageToRemoteDisk,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoteCLIPLoader": "Remote CLIP Loader",
    "RemoteVAELoader": "Remote VAE Loader",
    "RemoteMiniMaxH3ImageAndReferenceToVideo": "Remote/Local MiniMax H3 Image + Reference to Video",
    "RemoteVAEDecodeVideo": "Remote VAE Decode Video",
    "SaveImageToRemoteDisk": "Save Image To Remote Disk",
}
