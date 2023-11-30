#!/usr/bin/env sh
set -e
wget -O dataset.zip.001 https://mediastore.rz.uni-augsburg.de/get/iVIJmKlFEj/
wget -O dataset.zip.002 https://mediastore.rz.uni-augsburg.de/get/PcXjyew4KM/
wget -O dataset.zip.003 https://mediastore.rz.uni-augsburg.de/get/sZUNXyFxgo/
wget -O dataset.zip.004 https://mediastore.rz.uni-augsburg.de/get/BOzrHGUkrp/
wget -O dataset.zip.005 https://mediastore.rz.uni-augsburg.de/get/YzyW1sTsYE/
wget -O dataset.zip.006 https://mediastore.rz.uni-augsburg.de/get/i9NBHKJxH3/
wget -O dataset.zip.007 https://mediastore.rz.uni-augsburg.de/get/7YwHsAyroy/
wget -O dataset.zip.008 https://mediastore.rz.uni-augsburg.de/get/dxl_jHQYrZ/

cat dataset.zip.00* > dataset.zip
rm dataset.zip.00*