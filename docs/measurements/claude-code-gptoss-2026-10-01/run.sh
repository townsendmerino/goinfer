#!/usr/bin/env bash
# A real Claude Code (2.1.286) run against goinfer-serve's /v1/messages with gpt-oss-20b. Isolated HOME, one tool, dummy key, loopback only.
D=$HOME/goinfer-bench/claude-code-gptoss-2026-10-01
CLAUDE=/home/francis/.vscode-server/extensions/anthropic.claude-code-2.1.286-linux-x64/resources/native-binary/claude
cd $D/work
date +%T > $D/started
env -i PATH="$PATH" HOME="$D/home" TERM=dumb \
  ANTHROPIC_BASE_URL=http://127.0.0.1:8100 ANTHROPIC_API_KEY=goinfer ANTHROPIC_MODEL=gptoss \
  ANTHROPIC_DEFAULT_HAIKU_MODEL=gptoss ANTHROPIC_DEFAULT_SONNET_MODEL=gptoss ANTHROPIC_DEFAULT_OPUS_MODEL=gptoss ANTHROPIC_SMALL_FAST_MODEL=gptoss \
  CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 DISABLE_TELEMETRY=1 DISABLE_AUTOUPDATER=1 \
  timeout 720 "$CLAUDE" -p "Use the Read tool to read notes.txt, then tell me the codeword it contains." \
    --bare --model gptoss --tools Read --allowedTools Read --max-turns 6 --no-session-persistence \
    --output-format stream-json --verbose > $D/claude.out 2> $D/claude.err
echo "claude exit $?" > $D/exit.txt; date +%T >> $D/exit.txt
