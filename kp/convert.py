import os
import struct
import ctypes
from PIL import Image

INPUT_DIR = "res"       # папка с .data
OUTPUT_DIR = "res_png"  # папка для .png

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Собираем список кадров: 0.data, 1.data, 2.data, ...
frames = [
    int(fname.split('.')[0])
    for fname in os.listdir(INPUT_DIR)
    if fname.endswith(".data") and fname.split('.')[0].isdigit()
]
frames.sort()

def from_data_to_png(data_path, png_path):
    with open(data_path, "rb") as fin:
        # Читаем два int: width, height (именно так пишет твоя программа)
        w, h = struct.unpack("ii", fin.read(8))

        buf = ctypes.create_string_buffer(4 * w * h)
        fin.readinto(buf)

    img = Image.new("RGBA", (w, h))
    pix = img.load()
    offset = 0
    for j in range(h):
        for i in range(w):
            # 4 байта = RGBA
            r, g, b, a = struct.unpack_from("BBBB", buf, offset)
            pix[i, j] = (r, g, b, a)
            offset += 4

    img.save(png_path)

for i in frames:
    data_path = os.path.join(INPUT_DIR, f"{i}.data")
    png_path = os.path.join(OUTPUT_DIR, f"{i}.png")
    print(f"Convert {data_path} -> {png_path}")
    from_data_to_png(data_path, png_path)
