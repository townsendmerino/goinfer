#!/usr/bin/env python3
"""Pin internal/clef/testdata/render_cases.json: inputs and the output of Python 3's
json.dumps(json.loads(x), ensure_ascii=False, separators=(",", ":"), sort_keys=True), which is what Clef's
encode_record (joint_schema_model.py, render) puts in the prompt. The cases are the ones Go's encoding/json gets
wrong: float spelling, control-character escapes, key order over non-ASCII keys, a repeated key, integers past 2^53,
and characters Go would escape (<, >, &, U+2028). Run from the repo root: python3 scripts/pin_clef_render_cases.py
"""
import json

CASES = [
    '{"b":1,"a":[1,2.5,{"z":null,"y":true}]}',
    '{"é":"ü","日本":"語","a":"x"}',
    '"tab\\there\\u0001\\u001f\\u007f\\u2028 \\" \\\\ / \\/ <>&"',
    '[1e5,1E5,1.0,-0,-0.0,0.0,1e-7,0.00001,0.0001,123456789012345678901234567890,1e16,1e15,12345678901234567.0,1.5e300,5e-324,0.1,100,-1.25e-10,1e21,3.14159,2.5E+3]',
    '{"nested":{"b":{"d":1,"c":2},"a":[]},"empty":{},"e2":[]}',
    '{"a":1,"b":2,"a":3}',
    '"😀 \\ud83d\\ude00"',
    '[true,false,null,"",0]',
    '9007199254740993',
    '{"Z":1,"a":2,"_":3,"é":4,"\\u00e9":5,"10":6,"9":7}',
]
out = [{"in": c, "out": json.dumps(json.loads(c), ensure_ascii=False, separators=(",", ":"), sort_keys=True)} for c in CASES]
with open("internal/clef/testdata/render_cases.json", "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(len(out), "cases")
