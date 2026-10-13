"""How each harness is run headless against goinfer, and how its output is read. One entry per harness; the table of I2
(docs/tasks/task-integration-harness-2026-10.md) says where each keeps its configuration, `connect` writes it."""
import glob, json, os

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
        """-> dict(answer, tool_calls, tool_errors, compactions, errors, ended_on_error)"""
        raise NotImplementedError


class ClaudeCode(Agent):
    name = connect_name = "claude-code"

    def config_path(self, home, work):
        return os.path.join(home, ".claude", "settings.json")

    def argv(self, prompt):
        return [BIN + "/claude", "-p", prompt, "--output-format", "stream-json", "--verbose", "--dangerously-skip-permissions", "--max-turns", "40"]

    def parse(self, stdout, stderr, home):
        r = dict(answer="", tool_calls=0, tool_errors=0, compactions=0, errors=[], ended_on_error=False, turns=0)
        for j in jsonl(stdout):
            t = j.get("type")
            if t == "assistant":
                r["turns"] += 1
                for c in j.get("message", {}).get("content", []):
                    if c.get("type") == "tool_use":
                        r["tool_calls"] += 1
            elif t == "user":
                c = j.get("message", {}).get("content")
                if isinstance(c, list):
                    r["tool_errors"] += sum(1 for x in c if isinstance(x, dict) and x.get("type") == "tool_result" and x.get("is_error"))
            elif t == "system" and "compact" in str(j.get("subtype", "")):
                r["compactions"] += 1
            elif t == "result":
                r["answer"] = str(j.get("result", ""))
                if j.get("is_error") or str(j.get("subtype", "")).startswith("error"):
                    r["ended_on_error"] = True
                    r["errors"].append(str(j.get("subtype")) + ": " + str(j.get("result", ""))[:200])
        return r


class OpenCode(Agent):
    name = connect_name = "opencode"

    def config_path(self, home, work):
        return os.path.join(home, ".config", "opencode", "opencode.json")

    def argv(self, prompt, model="goinfer/MODEL"):
        return [BIN + "/opencode", "run", "-m", model, "--format", "json", prompt]

    def parse(self, stdout, stderr, home):
        r = dict(answer="", tool_calls=0, tool_errors=0, compactions=0, errors=[], ended_on_error=False, turns=0)
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
                if p.get("state", {}).get("status") == "error":
                    r["tool_errors"] += 1
            elif "compact" in str(t) or p.get("type") == "compaction":
                r["compactions"] += 1
            elif t == "error":
                r["errors"].append(json.dumps(j.get("error", j))[:200])
                r["ended_on_error"] = True
        r["answer"] = texts[-1] if texts else ""
        return r


class Codex(Agent):
    name = connect_name = "codex"

    def config_path(self, home, work):
        return os.path.join(home, ".codex", "config.toml")

    def env(self, home):
        return {"CODEX_HOME": os.path.join(home, ".codex"), "GOINFER_API_KEY": "goinfer"}

    def argv(self, prompt):
        return [BIN + "/codex", "exec", "--json", "--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox", prompt]

    def parse(self, stdout, stderr, home):
        r = dict(answer="", tool_calls=0, tool_errors=0, compactions=0, errors=[], ended_on_error=False, turns=0)
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
        # Codex records its own compactions in the session log it keeps under CODEX_HOME.
        for f in glob.glob(os.path.join(home, ".codex", "sessions", "**", "*.jsonl"), recursive=True):
            for line in open(f, errors="replace"):
                if '"type":"compacted"' in line.replace(" ", ""):
                    r["compactions"] += 1
        return r


AGENTS = {a.name: a for a in (ClaudeCode(), OpenCode(), Codex())}
