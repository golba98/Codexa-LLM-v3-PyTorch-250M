#!/usr/bin/env sh
set -eu
codexa tokenizer train --corpus examples/corpus --vocab-size 512 --output artifacts/tokenizer.json
codexa data prepare --corpus examples/corpus --tokenizer artifacts/tokenizer.json --output artifacts/data
codexa train --config configs/smoke.yaml --data artifacts/data --tokenizer artifacts/tokenizer.json --output runs/smoke --device cpu --until 50
codexa train --config configs/smoke.yaml --data artifacts/data --tokenizer artifacts/tokenizer.json --output runs/smoke --device cpu --resume runs/smoke/latest.pt
codexa generate --checkpoint runs/smoke/latest.pt --prompt 'Morning light' --max-tokens 20 --greedy
