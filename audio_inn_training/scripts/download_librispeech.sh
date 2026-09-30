#!/usr/bin/env bash
set -euo pipefail
mkdir -p data
cd data
BASE="https://www.openslr.org/resources/12"
for F in train-clean-100.tar.gz dev-clean.tar.gz test-clean.tar.gz; do
  echo "Downloading $F"
  wget -c "$BASE/$F"
  tar -xzf "$F"
  rm -f "$F"
done
echo "LibriSpeech extracted under data/LibriSpeech"
