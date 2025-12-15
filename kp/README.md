
# Ray Tracing на CUDA (Гексаэдр, Октаэдр, Икосаэдр)

Проект рендерит анимацию платоновых тел (гексаэдр, октаэдр, икосаэдр) на GPU/CPU, сохраняет кадры в бинарном формате `.data`, затем конвертирует их в PNG и собирает видео.

## Зависимости

- CUDA Toolkit (`nvcc`)
- C++ компилятор (идёт с CUDA)
- Python 3 + Pillow

---

## Запуск через `make` (рекомендуется)

Из каталога `kp`:

```bash
cd ~/pgp_ppt/kp
make
```

Что делает `make`:

1. Собирает бинарник `raytracer` из `main.cu`.
2. Запускает `./raytracer --default` и пишет конфиг в `config.txt`.
3. Пересоздаёт папку `res/` и рендерит туда кадры (`res/*.data`).
4. Пересоздаёт папку `res_png/` и конвертирует `.data → .png` через `convert.py`.
5. Собирает видео `video1080.mp4` из PNG через `ffmpeg`.

Очистка всего (бинарник, конфиг, кадры, PNG и видео):

```bash
make clean
```

---

## Ручной запуск (без `make`)

### 1. Сборка

```bash
nvcc -std=c++11 -O2 main.cu -o raytracer
```

### 2. Получить дефолтный конфиг

```bash
./raytracer --default > config.txt
```

Программа выводит в stdout входные параметры сцены, которые сохраняются в `config.txt`.

### 3. Рендер кадров в `res/`

```bash
rm -rf res
mkdir -p res
./raytracer < config.txt
```

Кадры сохраняются как `res/0.data`, `res/1.data`, …

### 4. Конвертация `.data → .png`

```bash
rm -rf res_png
mkdir -p res_png
python3 convert.py
```

Скрипт читает все `.data` из `res/` и пишет PNG в `res_png/0.png`, `res_png/1.png`, …

### 5. Сборка видео из PNG

```bash
ffmpeg -y -framerate 24 -i res_png/%d.png \
  -vf "scale=1920:1080:flags=lanczos" \
  -c:v libx264 -pix_fmt yuv420p -crf 18 video1080.mp4
```

На выходе — файл `video1080.mp4` в текущей директории.
