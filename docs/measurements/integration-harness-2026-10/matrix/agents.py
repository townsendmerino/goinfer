"""How each harness is run headless against goinfer, and how its output is read. One entry per harness; the table of I2
(docs/tasks/task-integration-harness-2026-10.md) says where each keeps its configuration, `connect` writes it."""
import glob, json, os, re

BIN = os.path.expanduser("~/harnesses/node_modules/.bin")
RO = [os.path.expanduser("~/harnesses")]   # read-only inside the sandbox, at the same path


def jsonl(text):
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                yield json.loads(line)
            except ValueError:
                pass


class Agent:
    name = ""
    connect_name = ""

    def config_path(self, home, work):
        raise NotImplementedError

    def env(self, home):
        return {}

    def argv(self, prompt):
        raise NotImplementedError

    def parse(self, stdout, stderr, home):
        """-> dict(answer, tool_calls, tool_errors, compactions, errors, ended_on_error, turns, events). events is the agent's
        own order of tool calls and compactions: ["tool", <its input as text>] or ["compact"]."""
        raise NotImplementedError

    def tell_window(self, path, window):
        """I1-v2's measurement configuration (never a user setting): rewrite the window `connect` wrote into the file at path
        to `window`, and return what the file says afterwards, read back, for the record and the runner's check."""
        raise NotImplementedError


class ClaudeCode(Agent):
    name = connect_name = "claude-code"

    def config_path(self, home, work):
        return os.path.join(home, ".claude", "settings.json")

    def argv(self, prompt):
        return [BIN + "/claude", "-p", prompt, "--output-format", "stream-json", "--verbose", "--dangerously-skip-permissions", "--max-turns", "40"]

    def parse(self, stdout, stderr, home):
        r = dict(answer="", tool_calls=0, tool_errors=0, compactions=0, errors=[], ended_on_error=False, turns=0, events=[])
        for j in jsonl(stdout):
            t = j.get("type")
            if t == "assistant":
                r["turns"] += 1
                for c in j.get("message", {}).get("content", []):
                    if c.get("type") == "tool_use":
                        r["tool_calls"] += 1
                        r["events"].append(["tool", json.dumps(c.get("input"))])
            elif t == "user":
                c = j.get("message", {}).get("content")
                if isinstance(c, list):
                    r["tool_errors"] += sum(1 for x in c if isinstance(x, dict) and x.get("type") == "tool_result" and x.get("is_error"))
            elif t == "system" and "compact" in str(j.get("subtype", "")):
                r["compactions"] += 1
                r["events"].append(["compact"])
            elif t == "result":
                r["answer"] = str(j.get("result", ""))
                if j.get("is_error") or str(j.get("subtype", "")).startswith("error"):
                    r["ended_on_error"] = True
                    r["errors"].append(str(j.get("subtype")) + ": " + str(j.get("result", ""))[:200])
        return r

    def tell_window(self, path, window):
        j = json.load(open(path))
        j["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(window)
        json.dump(j, open(path, "w"), indent=2)
        env = json.load(open(path))["env"]
        return dict(window=int(env.get("CLAUDE_CODE_MAX_CONTEXT_TOKENS", 0)), output_cap=int(env.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS", 0)))


class OpenCode(Agent):
    name = connect_name = "opencode"

    def config_path(self, home, work):
        return os.path.join(home, ".config", "opencode", "opencode.json")

    def argv(self, prompt, model="goinfer/MODEL"):
        return [BIN + "/opencode", "run", "-m", model, "--format", "json", prompt]

    def parse(self, stdout, stderr, home):
        r = dict(answer="", tool_calls=0, tool_errors=0, compactions=0, errors=[], ended_on_error=False, turns=0, events=[])
        texts = []
        for j in jsonl(stdout):
            t = j.get("type")
            p = j.get("part", {})
            if t == "step_start":
                r["turns"] += 1
            elif t == "text":
                texts.append(p.get("text", ""))
            elif t == "tool_use":
                r["tool_calls"] += 1
                r["events"].append(["tool", json.dumps(p.get("state", {}).get("input"))])
                if p.get("state", {}).get("status") == "error":
                    r["tool_errors"] += 1
            elif "compact" in str(t) or p.get("type") == "compaction":
                r["compactions"] += 1
                r["events"].append(["compact"])
            elif t == "error":
                r["errors"].append(json.dumps(j.get("error", j))[:200])
                r["ended_on_error"] = True
        r["answer"] = texts[-1] if texts else ""
        return r

    def tell_window(self, path, window):
        j = json.load(open(path))
        models = j["provider"]["goinfer"]["models"]
        for m in models.values():
            m["limit"]["context"] = window
        json.dump(j, open(path, "w"), indent=2)
        models = json.load(open(path))["provider"]["goinfer"]["models"]
        return dict(window=min(m["limit"]["context"] for m in models.values()), output_cap=min(m["limit"].get("output", 0) for m in models.values()))


class Codex(Agent):
    name = connect_name = "codex"

    def config_path(self, home, work):
        return os.path.join(home, ".codex", "config.toml")

    def env(self, home):
        return {"CODEX_HOME": os.path.join(home, ".codex"), "GOINFER_API_KEY": "goinfer"}

    def argv(self, prompt):
        return [BIN + "/codex", "exec", "--json", "--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox", prompt]

    def parse(self, stdout, stderr, home):
        r = dict(answer="", tool_calls=0, tool_errors=0, compactions=0, errors=[], ended_on_error=False, turns=0, events=[])
        msgs = []
        for j in jsonl(stdout):
            t = j.get("type")
            it = j.get("item", {})
            if t == "turn.started":
                r["turns"] += 1
            elif t == "item.completed":
                if it.get("type") == "agent_message":
                    msgs.append(it.get("text", ""))
                elif it.get("type") == "command_execution":
                    r["tool_calls"] += 1
                    if it.get("exit_code") not in (0, None):
                        r["tool_errors"] += 1
                elif it.get("type") == "error":
                    r["errors"].append(it.get("message", "")[:200])
            elif t in ("error", "turn.failed"):
                r["errors"].append(str(j.get("message") or j.get("error"))[:200])
                if t == "turn.failed":
                    r["ended_on_error"] = True
        r["answer"] = msgs[-1] if msgs else ""
        # Codex records its own compactions, and its tool calls in order, in the session log it keeps under CODEX_HOME;
        # the log also states the window Codex took from the configuration.
        rows = []
        for f in glob.glob(os.path.join(home, ".codex", "sessions", "**", "*.jsonl"), recursive=True):
            for line in open(f, errors="replace"):
                try:
                    j = json.loads(line)
                except ValueError:
                    continue
                rows.append(j)
                m = re.search(r'"model_context_window":\s*(\d+)', line)
                if m:
                    r["context_window_seen"] = int(m.group(1))
        rows.sort(key=lambda j: (j.get("timestamp", ""), j.get("ordinal", 0)))
        for j in rows:
            p = j.get("payload") if isinstance(j.get("payload"), dict) else {}
            if j.get("type") == "compacted":
                r["compactions"] += 1
                r["events"].append(["compact"])
            elif j.get("type") == "response_item" and p.get("type") == "function_call":
                r["events"].append(["tool", str(p.get("arguments"))])
        return r

    def tell_window(self, path, window):
        s = open(path).read()
        cw = int(re.search(r"^model_context_window\s*=\s*(\d+)", s, re.M).group(1))
        ac = int(re.search(r"^model_auto_compact_token_limit\s*=\s*(\d+)", s, re.M).group(1))
        out = cw - ac
        s = re.sub(r"^model_context_window\s*=\s*\d+", f"model_context_window = {window}", s, flags=re.M)
        s = re.sub(r"^model_auto_compact_token_limit\s*=\s*\d+", f"model_auto_compact_token_limit = {window - out}", s, flags=re.M)
        open(path, "w").write(s)
        s = open(path).read()
        cw = int(re.search(r"^model_context_window\s*=\s*(\d+)", s, re.M).group(1))
        ac = int(re.search(r"^model_auto_compact_token_limit\s*=\s*(\d+)", s, re.M).group(1))
        return dict(window=cw, auto_compact=ac, output_cap=cw - ac)


AGENTS = {a.name: a for a in (ClaudeCode(), OpenCode(), Codex())}
