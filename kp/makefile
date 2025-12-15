NVCC      = nvcc
NVCCFLAGS = -std=c++11 -O2

PYTHON    = python3
FFMPEG    = ffmpeg

RAYTRACER = raytracer
CONFIG    = config.txt
RES_DIR   = res
PNG_DIR   = res_png
VIDEO     = video.mp4
FRAMERATE = 24

.PHONY: all render pngs clean

# Полный пайплайн: сборка бинарника -> рендер -> PNG -> видео
all: $(VIDEO)

# Сборка C++/CUDA
$(RAYTRACER): main.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Генерация конфига
$(CONFIG): $(RAYTRACER)
	./$(RAYTRACER) --default > $(CONFIG)

# Рендер кадров в res/*.data
render: $(RAYTRACER) $(CONFIG)
	rm -rf $(RES_DIR)
	mkdir -p $(RES_DIR)
	./$(RAYTRACER) < $(CONFIG)

# Конвертация .data -> .png
pngs: render convert.py
	rm -rf $(PNG_DIR)
	mkdir -p $(PNG_DIR)
	$(PYTHON) convert.py

# Сборка видео 1080p из PNG
$(VIDEO): pngs
	$(FFMPEG) -y -framerate $(FRAMERATE) -i $(PNG_DIR)/%d.png \
	  -vf "scale=1920:1080:flags=lanczos" \
	  -c:v libx264 -pix_fmt yuv420p -crf 18 $(VIDEO)

# Удалить всё
clean:
	rm -rf $(RAYTRACER) $(RES_DIR) $(PNG_DIR) $(CONFIG) $(VIDEO)
