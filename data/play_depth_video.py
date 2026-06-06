from pathlib import Path
from PIL import Image

folder = Path("debug_depth_frames")
frames = sorted(folder.glob("depth_env0_frame_*.png"))
frames = [p for p in frames if "_big" not in p.name and "preview" not in p.name]

imgs = []
for p in frames:
    img = Image.open(p).convert("L")
    img = img.resize((640, 480), Image.Resampling.NEAREST)
    imgs.append(img)

out = folder / "depth_env0_preview.gif"
imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=20, loop=0)
print(out)
