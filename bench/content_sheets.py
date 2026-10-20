"""Contact sheets of embedding-map entries (frames and tiles) for labelling by eye."""
import io, json, os, random
from PIL import Image, ImageDraw
p = os.path.expanduser("~/.clef_operator/embed_map_75c7d89c-e674-a201-e313-35a050612f7e.json")
entries = json.load(open(p))["entries"]
thumbs = p[:-5]
entries = [e for e in entries if os.path.exists(f"{thumbs}/{e['shot']}.jpg")]
def spread(items, n):
    items = sorted(items, key=lambda e: (round(e["pan"] / 10), e["tilt"], e["fov"]))
    step = max(1, len(items) / n)
    return [items[int(i * step)] for i in range(min(n, len(items)))]
chosen = spread([e for e in entries if e["kind"] == "view"], 40) + spread([e for e in entries if e["kind"] == "tile"], 200)
os.makedirs("/tmp/label_sheets", exist_ok=True)
index = []
W, H, COLS, ROWS = 200, 150, 5, 4
for n in range(0, len(chosen), COLS * ROWS):
    sheet = Image.new("RGB", (W * COLS, H * ROWS), "white")
    draw = ImageDraw.Draw(sheet)
    for k, e in enumerate(chosen[n:n + COLS * ROWS]):
        i = n + k
        im = Image.open(f"{thumbs}/{e['shot']}.jpg").convert("RGB")
        x, y, w, h = e["box"]
        im = im.crop((round(x * im.width), round(y * im.height), round((x + w) * im.width), round((y + h) * im.height)))
        im = im.resize((W - 4, H - 4))
        cx, cy = (k % COLS) * W, (k // COLS) * H
        sheet.paste(im, (cx + 2, cy + 2))
        draw.rectangle([cx + 2, cy + 2, cx + 34, cy + 18], fill="black")
        draw.text((cx + 5, cy + 4), str(i), fill="yellow")
        index.append({"id": i, "kind": e["kind"], "pan": e["pan"], "tilt": e["tilt"], "fov": e["fov"], "shot": e["shot"], "box": e["box"], "vec": e["vec"]})
    sheet.save(f"/tmp/label_sheets/sheet_{n // (COLS * ROWS):02d}.jpg", quality=85)
json.dump(index, open("/tmp/label_sheets/index.json", "w"))
print(len(index), "items on", (len(index) + COLS * ROWS - 1) // (COLS * ROWS), "sheets")
