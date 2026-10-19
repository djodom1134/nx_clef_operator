"""EmbeddingGemma 2 as a small HTTP service next to Clef, for the PTZ hunter's cheap looks.

    HF_HOME=~/embed/hf ~/embed/.venv/bin/python scripts/embed_server.py --port 8300

POST /embed {"images": [base64 JPEG, ...]}             -> {"vectors": [[768 floats], ...], "seconds": s}
POST /embed {"texts": [str, ...], "prompt": "query"}    -> the same, for text (prompt: a model prompt name)
POST /embed {"images": [one JPEG], "crops": [[x, y, w, h], ...], "thumb": 320}
                                                        -> vectors for the image then each crop (fractions of the
                                                           image, cut here: no ffmpeg per tile), and "thumb", the image
                                                           as a base64 JPEG at most 320 px wide (the hunter's map)
GET  /health                                            -> {"status": "ready", "model": ..., "budget": ...}

Text and image encoders only (no audio), bf16, ``--budget`` soft tokens per image (70: about 50 ms an image and 0.35 s
for 9 tiles while Clef shares the GPU; the hunter's views ranked as well as at the default 280). Vectors are
normalized, so a dot product is the cosine similarity. One request at a time.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from PIL import Image
from sentence_transformers import SentenceTransformer


def load(model_id: str, budget: int) -> SentenceTransformer:
    model = SentenceTransformer(model_id, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16},
                                config_kwargs={"audio_config": None})
    processor = model[0].processor
    processor.image_processor.max_soft_tokens = processor.image_processor.image_seq_length = budget
    processor.image_seq_length = budget
    return model


def decode(data: str) -> Image.Image:
    image = Image.open(io.BytesIO(base64.b64decode(data)))
    image.draft("RGB", (896, 672))  # decode a 5 MP snapshot at a fraction of its size: the model sees far less
    return image.convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8300)
    parser.add_argument("--model", default="google/embeddinggemma-2")
    parser.add_argument("--budget", type=int, default=70)
    args = parser.parse_args()
    started = time.time()
    model = load(args.model, args.budget)
    model.encode(["warm up"], prompt_name="query")
    lock = threading.Lock()
    print(f"embedding server on {args.host}:{args.port}: {args.model}, budget {args.budget}, "
          f"loaded in {time.time() - started:.1f} s", flush=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):  # quiet: the hunter calls often
            pass

        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                return self._send(200, {"status": "ready", "model": args.model, "budget": args.budget})
            self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/embed":
                return self._send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                with lock:
                    began = time.time()
                    thumb = None
                    if body.get("images"):
                        inputs = [decode(data) for data in body["images"]]
                        for x, y, w, h in body.get("crops") or []:
                            W, H = inputs[0].size
                            inputs.append(inputs[0].crop((round(x * W), round(y * H), round((x + w) * W), round((y + h) * H))))
                        vectors = model.encode(inputs, batch_size=16, normalize_embeddings=True)
                        if body.get("thumb"):
                            small = inputs[0].copy()
                            small.thumbnail((int(body["thumb"]), int(body["thumb"])))
                            out = io.BytesIO()
                            small.save(out, "JPEG", quality=80)
                            thumb = base64.b64encode(out.getvalue()).decode()
                    else:
                        vectors = model.encode(list(body["texts"]), prompt_name=body.get("prompt") or None,
                                               batch_size=32, normalize_embeddings=True)
                    seconds = time.time() - began
                reply = {"vectors": [[round(float(x), 5) for x in v] for v in vectors], "seconds": round(seconds, 3)}
                if thumb:
                    reply["thumb"] = thumb
                self._send(200, reply)
            except Exception as error:  # noqa: BLE001 - report to the caller, keep serving
                self._send(400, {"error": repr(error)[:300]})

    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
