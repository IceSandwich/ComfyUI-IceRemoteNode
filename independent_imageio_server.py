import asyncio
import io
import logging
import os
import re
import typing
from utils import format_template

from aiohttp import web
from PIL import Image, PngImagePlugin
import torch


logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s [%(levelname)s] %(message)s",
)

HOST = "127.0.0.1"
PORT = 8188
ROUTE_PATH = "/icern/save__image_to_disk"
OUTPUT_DIR = os.path.abspath("./output")


def _resolve_save_path(
	filename_prefix: str,
	output_dir: str,
) -> tuple[str, str, int]:
	"""解析输出子目录、基础文件名与起始自增编号。"""
	normalized = filename_prefix.replace("\\", "/").lstrip("/")
	parts = normalized.split("/")
	if len(parts) > 1:
		subfolder = os.path.join(*parts[:-1])
		filename = parts[-1]
	else:
		subfolder = ""
		filename = parts[0] or "image"

	subfolder = format_template(subfolder)
	full_output_folder = os.path.join(output_dir, subfolder)
	os.makedirs(full_output_folder, exist_ok=True)

	prefix_pattern = re.escape(filename.replace("%batch_num%", r"\d+"))
	file_re = re.compile(rf"^{prefix_pattern}_(\d+)_\.png$", re.IGNORECASE)

	max_counter = 0
	for entry in os.listdir(full_output_folder):
		match = file_re.match(entry)
		if match:
			max_counter = max(max_counter, int(match.group(1)))


	return full_output_folder, filename, max_counter + 1


def _build_pnginfo(metadata: typing.Any) -> typing.Optional[PngImagePlugin.PngInfo]:
	"""将传入的 metadata 转换为 PIL PngInfo（若适用）。"""
	if metadata is None:
		return None
	if isinstance(metadata, PngImagePlugin.PngInfo):
		return metadata
	pnginfo = PngImagePlugin.PngInfo()
	if isinstance(metadata, dict):
		for k, v in metadata.items():
			pnginfo.add_text(str(k), str(v))
		return pnginfo
	if isinstance(metadata, str):
		pnginfo.add_text("comment", metadata)
		return pnginfo
	return None


async def save_image_to_disk_handler(request: web.Request) -> web.StreamResponse:
	logging.info(f"[IceRemoteNode] POST {ROUTE_PATH}")
	try:
		raw_bytes = await request.read()
		payload = torch.load(
			io.BytesIO(raw_bytes),
			map_location="cpu",
			weights_only=False,
		)
	except Exception as e:
		logging.exception("[IceRemoteNode] 反序列化失败")
		return web.json_response({"error": str(e)}, status=400)

	if not isinstance(payload, dict):
		return web.json_response(
			{"error": "Payload must be a dict after torch.load"},
			status=400,
		)
	
	images: typing.List[bytes] = payload["images"]
	filename_prefix: str = payload.get("filename_prefix", "ComfyUI")
	metadata = payload.get("metadata")
	compress_level: int = int(payload.get("compress_level", 4))

	full_output_folder, filename, counter = _resolve_save_path(
		filename_prefix, OUTPUT_DIR
	)
	pnginfo = _build_pnginfo(metadata)

	for batch_number, image_bytes in enumerate(images):
		with Image.open(io.BytesIO(image_bytes)) as img:
			filename_with_batch_num = filename.replace(
				"%batch_num%", str(batch_number)
			)
			file_name = f"{filename_with_batch_num}_{counter:05}_.png"
			save_path = os.path.join(full_output_folder, file_name)
			save_kwargs: dict[str, typing.Any] = {"compress_level": compress_level}
			if pnginfo is not None:
				save_kwargs["pnginfo"] = pnginfo
			img.save(save_path, **save_kwargs)
			logging.info(f"[IceRemoteNode] Saved to {save_path}")
			counter += 1

	result = None
	buf_out = io.BytesIO()
	torch.save(result, buf_out)
	resp = web.Response(
		body=buf_out.getvalue(),
		content_type="application/octet-stream",
	)
	resp.enable_compression(web.ContentCoding.gzip)
	return resp


def create_app() -> web.Application:
	app = web.Application(client_max_size=10 * 1024 * 1024)
	app.router.add_post(ROUTE_PATH, save_image_to_disk_handler)
	return app


async def main() -> None:
	app = create_app()
	runner = web.AppRunner(app)
	await runner.setup()
	site = web.TCPSite(runner, HOST, PORT)
	await site.start()
	logging.info(f"Server listening on http://{HOST}:{PORT}")
	try:
		await asyncio.Event().wait()
	finally:
		await runner.cleanup()


if __name__ == "__main__":
	asyncio.run(main())