# Rebuilds a quick cell's go test command from its log's first line ("# go test ... -tags a b -timeout 45m -run RE PKG"), which quick
# prints without quoting, as a shell-quoted command with -tags joined by commas and -timeout replaced. Usage: cellcmd.py LOG TIMEOUT [RUN]  (RUN replaces the cell's -run/-skip)
import shlex, sys
toks = open(sys.argv[1]).readline().lstrip("# ").split()
assert toks[:2] == ["go", "test"], toks[:3]
out, i = [], 0
while i < len(toks):
    t = toks[i]
    if t == "-tags":
        j = i + 1; tags = []
        while j < len(toks) and not toks[j].startswith("-"):
            tags.append(toks[j]); j += 1
        out += ["-tags", ",".join(tags)]; i = j; continue
    if len(sys.argv) > 3 and t in ("-run", "-skip"):
        i += 2; continue
    if t == "-timeout":
        out += ["-timeout", sys.argv[2]]; i += 2; continue
    out.append(t); i += 1
if len(sys.argv) > 3:
    out[-1:-1] = ["-run", sys.argv[3]]
print(shlex.join(out))
