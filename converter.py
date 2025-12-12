#!/usr/bin/env python3
"""
convert_img.py — конвертация между .data (w,h + RGBA) и .png

Формат .data:
- Первые 8 байт: два 32-битных целых (little-endian) — ширина (w), высота (h).
- Далее w*h*4 байта RGBA по пикселям в порядке строк (j от 0..h-1, i от 0..w-1).

Режимы:
  --to-png  : .data → .png
  --to-bin  : .png  → .data (альфа устанавливается в 255)

Примеры:
  1) PNG → DATA с явным выходом:
     python converter.py --to-bin 1.png -o 1.data

  3) DATA → PNG с явным выходом:
     python converter.py --to-png 1_out.data -o 1_out.png
"""

import argparse
import os
import struct
import ctypes
from  PIL import Image


def data_to_png(in_path: str, out_path: str | None):
    with open(in_path, 'rb') as f:
        header = f.read(8)
        if len(header) != 8:
            raise ValueError("Файл слишком короткий: нет заголовка 8 байт (w,h).")
        w, h = struct.unpack('ii', header)
        expected = 4 * w * h
        buff = ctypes.create_string_buffer(expected)
        n = f.readinto(buff)
        if n != expected:
            raise ValueError(f"Ожидалось {expected} байт пикселей, прочитано {n}.")

    img = Image.new('RGBA', (w, h))
    pix = img.load()

    offset = 0
    for j in range(h):
        for i in range(w):
            r, g, b, _a = struct.unpack_from('BBBB', buff, offset)
            pix[i, j] = (r, g, b, 255)
            offset += 4

    if not out_path:
        base, _ = os.path.splitext(in_path)
        out_path = base + '.png'
    img.save(out_path)
    print(f"OK: {in_path} → {out_path} ({w}x{h})")


def png_to_data(in_path: str, out_path: str | None):
    img = Image.open(in_path).convert('RGBA')
    w, h = img.size
    pix = img.load()

    buff = ctypes.create_string_buffer(4 * w * h)
    offset = 0
    for j in range(h):
        for i in range(w):
            r, g, b, _a = pix[i, j]
            a = 255
            struct.pack_into('BBBB', buff, offset, r, g, b, a)
            offset += 4

    if not out_path:
        base, _ = os.path.splitext(in_path)
        out_path = base + '.data'
    with open(out_path, 'wb') as out:
        out.write(struct.pack('ii', w, h))
        out.write(buff.raw)

    print(f"OK: {in_path} → {out_path} ({w}x{h}, {4*w*h} байт пикселей)")


def main():
    parser = argparse.ArgumentParser(
        description="Конвертация между .data (w,h + RGBA) и .png."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--to-png', action='store_true', help='Перевод .data → .png')
    mode.add_argument('--to-bin', action='store_true', help='Перевод .png → .data')
    parser.add_argument('input', help='Путь к входному файлу')
    parser.add_argument('-o', '--output', help='Путь к выходному файлу (необязательно)')

    args = parser.parse_args()

    if args.to_png:
        data_to_png(args.input, args.output)
    elif args.to_bin:
        png_to_data(args.input, args.output)


if __name__ == '__main__':
    main()
