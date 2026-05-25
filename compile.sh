#!/usr/bin/env bash
set -euo pipefail

rm -rf build
cmake -S . -B bild -G Ninja -DCMake_EXPORT_COMPILE_COMMANDS=ON
cmake --build build

echo "\n PROGRAM BUILT"
