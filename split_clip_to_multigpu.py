import re
import torch
import logging

class SplitClipToMultiGPU:
    """
    支持任意设备切分的 Transformer 层切分节点。

    device_map 格式示例：
        "cuda:0@0-40,cuda:1"
            → 0-40 层放 cuda:0，其余层放 cuda:1

        "cuda:0@0-20;40-50,cuda:1@21-39,cuda:0"
            → 0-20 与 40-50 放 cuda:0，21-39 放 cuda:1，其余层兜底回 cuda:0

        "cuda:0@0-30,cuda:1@31-50,cuda:2"
            → 三段切分

    说明：
        - 第一个条目决定“主设备”(primary)，最终模型输出会被搬回主设备。
        - 没有 @ 的条目是兜底设备，会接收所有未显式分配的层。
        - 同一设备多个范围用 ; 分隔；范围用 start-end 表示（含两端），单层可直接写数字。
        - 支持任意顺序、非连续的切分。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "device_map": ("STRING", {
                    "default": "cuda:0@0-40,cuda:1",
                    "multiline": False,
                }),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "apply_dual_gpu_split"
    CATEGORY = "IceRemoteNode"

    @staticmethod
    def _parse_device_map(s: str, total_layers: int):
        explicit = {}   # layer_idx -> torch.device
        fallback = None # 兜底设备
        primary = None  # 主设备

        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            if "@" in part:
                dev_str, ranges_str = part.split("@", 1)
                dev = torch.device(dev_str.strip())
                if primary is None:
                    primary = dev
                for r in ranges_str.split(";"):
                    r = r.strip()
                    if not r:
                        continue
                    if "-" in r:
                        a, b = r.split("-", 1)
                        for i in range(int(a), int(b) + 1):
                            explicit[i] = dev
                    else:
                        explicit[int(r)] = dev
            else:
                dev = torch.device(part)
                if primary is None:
                    primary = dev
                fallback = dev

        if primary is None:
            raise ValueError("device_map 不能为空")

        layer_device = {}
        for i in range(total_layers):
            if i in explicit:
                layer_device[i] = explicit[i]
            elif fallback is not None:
                layer_device[i] = fallback
            else:
                layer_device[i] = primary
        return layer_device, primary

    def apply_dual_gpu_split(self, clip, device_map):
        cond_stage = clip.cond_stage_model

        logging.info(f"[{self.CATEGORY}] Apply multi-gpu split: {device_map} =====================")

        # 0. 清理旧钩子，支持随时修改 device_map 重复运行
        def remove_hooks():
            for handle in getattr(cond_stage, "_iceremotenode_multigpu_hooks", []):
                handle.remove()
            cond_stage._iceremotenode_multigpu_hooks = []
        remove_hooks()
        hooks = cond_stage._iceremotenode_multigpu_hooks

        # 1. 定位 Transformer 层列表
        layers_prefix, layers_module = next(
            ((name, mod) for name, mod in cond_stage.named_modules()
             if isinstance(mod, torch.nn.ModuleList) and len(mod) > 4),
            (None, None)
        )
        if layers_module is None:
            raise RuntimeError(f"未找到 ModuleList 层列表, {cond_stage}")

        total_layers = len(layers_module)

        # 2. 解析设备映射
        layer_device, primary_device = self._parse_device_map(device_map, total_layers)

        layer_pattern = re.compile(rf"^{re.escape(layers_prefix)}\.(\d+)(?:\.|$)")

        def get_target_device(name: str) -> torch.device:
            match = layer_pattern.match(name)
            if match:
                return layer_device.get(int(match.group(1)), primary_device)
            return primary_device

        # 3. 按叶节点模块精确迁移权重
        for mod_name, mod in cond_stage.named_modules():
            if len(list(mod.children())) == 0:
                mod.to(get_target_device(mod_name))

        torch.cuda.empty_cache()

        # 统计显存
        devices_in_use = sorted({*layer_device.values(), primary_device},
                                key=lambda d: str(d))
        mem_info = []
        for d in devices_in_use:
            if d.type == "cuda":
                mem = torch.cuda.memory_allocated(d) / (1024 ** 3)
                mem_info.append(f"{d}: {mem:.2f} GB")
        logging.info(
            f"[MultiGPU] 切分 {layers_prefix} 完成 "
            f"({total_layers} 层, primary={primary_device}) -> "
            + " | ".join(mem_info)
        )

        # 4. 递归张量与 KV Cache 设备搬运
        def move_to_device(obj, device):
            if isinstance(obj, torch.Tensor):
                if obj.device != device:
                    return obj.to(device, non_blocking=True)
                return obj
            if isinstance(obj, tuple):
                return tuple(move_to_device(x, device) for x in obj)
            if isinstance(obj, list):
                return [move_to_device(x, device) for x in obj]
            if isinstance(obj, dict):
                return {k: move_to_device(v, device) for k, v in obj.items()}
            if hasattr(obj, "key_cache") and isinstance(obj.key_cache, list):
                for idx in range(len(obj.key_cache)):
                    if isinstance(obj.key_cache[idx], torch.Tensor):
                        obj.key_cache[idx] = move_to_device(obj.key_cache[idx], device)
            if hasattr(obj, "value_cache") and isinstance(obj.value_cache, list):
                for idx in range(len(obj.value_cache)):
                    if isinstance(obj.value_cache[idx], torch.Tensor):
                        obj.value_cache[idx] = move_to_device(obj.value_cache[idx], device)
            return obj

        def make_device_guard(target_device):
            caching: dict[str, tuple] = {}
            caching_keys = ("attention_mask", "freqs_cis")

            def guard_hook(module, args, kwargs):
                new_args = move_to_device(args, target_device)
                new_kwargs = {}
                for k, v in kwargs.items():
                    if k in caching_keys and False:
                        if k in caching and caching[k][0] is v:
                            new_kwargs[k] = caching[k][1]
                        else:
                            tv = move_to_device(v, target_device)
                            new_kwargs[k] = tv
                            caching[k] = (v, tv)
                    else:
                        new_kwargs[k] = move_to_device(v, target_device)
                return new_args, new_kwargs

            return guard_hook

        def post_hook_to_primary(module, args, output):
            return move_to_device(output, primary_device)

        # 5. 每层注册 pre-hook：把输入对齐到该层所在设备
        guard_by_device: dict[torch.device, object] = {}
        for i, mod in enumerate(layers_module):
            dev = layer_device.get(i, primary_device)
            if dev not in guard_by_device:
                guard_by_device[dev] = make_device_guard(dev)
            h = mod.register_forward_pre_hook(guard_by_device[dev], with_kwargs=True)
            hooks.append(h)

        # 6. 最后一层输出统一搬回主设备
        h = layers_module[-1].register_forward_hook(post_hook_to_primary)
        hooks.append(h)

        # 7. 接管 ModelPatcher，阻止 ComfyUI 重调度
        patcher = clip.patcher
        patcher.load_device = patcher.offload_device = patcher.current_device = primary_device
        patcher.model_size = lambda: 0
        patcher.patch_model = lambda *args, **kwargs: patcher.model
        patcher.partially_load = patcher.partially_unload = lambda *args, **kwargs: 0
        patcher.model_patches_to = lambda *args, **kwargs: None

        clip.icern_remove_hooks = remove_hooks
        return (clip,)


NODE_CLASS_MAPPINGS = {"SplitClipToMultiGPU": SplitClipToMultiGPU}
NODE_DISPLAY_NAME_MAPPINGS = {"SplitClipToMultiGPU": "Split Clip blocks to Multi-GPU"}
