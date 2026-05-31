"""Data pipeline: fetch/preprocess/labels/manifest. См. SRS §6, §11.1."""

import os

# WGS84↔UTM не требует сетевых PROJ-грид-файлов. Отключаем сетевые запросы PROJ
# (cdn.proj.org), иначе на нестабильной сети они виснут и/или дают деградированную
# трансформацию (наблюдались зависание preprocess и абсурдные размеры сетки).
os.environ.setdefault("PROJ_NETWORK", "OFF")

# Чтение удалённых COG через /vsicurl по нестабильной сети: повторять при обрыве
# тайла (TIFFReadEncodedTile failed) вместо падения. См. fetch._clip_remote_s1.
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "3")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "120")
os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD", "NO")
