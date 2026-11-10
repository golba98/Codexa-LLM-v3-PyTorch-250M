#!/usr/bin/env sh
set -eu
mkdir -p outputs/experiments
run_root=$(mktemp -d outputs/experiments/smoke.XXXXXX)
codexa tokenizer train --corpus examples/corpus --vocab-size 512 --output "$run_root/tokenizer.json"
codexa data prepare --corpus examples/corpus --tokenizer "$run_root/tokenizer.json" --output "$run_root/data"
codexa train --config configs/smoke.yaml --data "$run_root/data" --tokenizer "$run_root/tokenizer.json" --output "$run_root/run" --device "${CODEXA_DEVICE:-cpu}" --until 50
codexa train --config configs/smoke.yaml --data "$run_root/data" --tokenizer "$run_root/tokenizer.json" --output "$run_root/run" --device "${CODEXA_DEVICE:-cpu}" --resume "$run_root/run/latest.pt"
codexa generate --checkpoint "$run_root/run/latest.pt" --prompt 'Morning light' --max-tokens 20 --greedy
printf 'Preserved smoke outputs: %s\n' "$run_root"
