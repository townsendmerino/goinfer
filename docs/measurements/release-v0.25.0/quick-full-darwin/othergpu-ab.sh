#!/usr/bin/env bash
# Discriminator (coordinator, 2026-11-21): TestOtherGPUWindow_seesAnotherProcess fails at 36dbfbaf1 on the Mac (macOS 26.7.3
# since 11-19); it passed in the 11-04 quick -full at ccdfb3ad. Same test at v0.24.1 and 36dbfbaf1, A B A B, under the lock.
set -uo pipefail
command -v go >/dev/null 2>&1 || export PATH=/usr/local/go/bin:/opt/homebrew/bin:$PATH
D=$HOME/goinfer-logs/relgate-2026-11-21/othergpu-ab; mkdir -p "$D"
SRC=$HOME/tmcode/goinfer; NEW=$HOME/goinfer-bench/relgate-qf-wt; OLD=$HOME/goinfer-bench/othergpu-v0241
[ -d "$OLD" ] || git -C "$SRC" worktree add -q --detach "$OLD" v0.24.1 || exit 2
[ -e "$OLD/testdata/llama-tiny/model.safetensors" ] || { rm -rf "$OLD/testdata/llama-tiny.tmp"; cp -cR "$NEW/testdata/llama-tiny" "$OLD/testdata/llama-tiny.tmp" && rm -rf "$OLD/testdata/llama-tiny" && mv "$OLD/testdata/llama-tiny.tmp" "$OLD/testdata/llama-tiny"; }
(cd "$OLD" && bash scripts/make_work.sh >/dev/null 2>&1)
for x in OLD NEW; do w=${!x}; (cd "$w/backend/webgpu" && go test -c -tags gpu,goinfer_testhooks -o "$D/$x.test" .) || { echo "FATAL build $x"; exit 2; }; done
echo "built $(date '+%T'); old $(git -C "$OLD" rev-parse --short HEAD) new $(git -C "$NEW" rev-parse --short HEAD)"
cat > "$D/ab.sh" <<EOS
for x in OLD NEW OLD NEW; do
  case \$x in OLD) w="$OLD";; NEW) w="$NEW";; esac
  echo "== \$x \$(date '+%T')"; (cd "\$w/backend/webgpu" && "$D/\$x.test" -test.run '^TestOtherGPUWindow_seesAnotherProcess\$' -test.count=1 -test.v -test.short 2>&1 | grep -E 'gpuothers_test|^--- |^ok|^FAIL|panic')
done
EOS
python3 "$NEW/scripts/timing_lock.py" run --wait 3600 --label "othergpu A/B v0.24.1 vs 36dbfbaf1 (coordinator)" -- bash "$D/ab.sh"
echo "ALL DONE $(date '+%T')"
