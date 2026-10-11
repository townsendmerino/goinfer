"""Compare the Mac's two int4 arms (CPU, Metal) with nobara's float32 CPU pass on the same token ids: the cosine of the
last layer's residual per position, and argmax agreement with float32."""
import json, sys, math
mac = json.load(open(sys.argv[1])); nob = json.load(open(sys.argv[2]))
def cos(a, b):
    d = sum(x * y for x, y in zip(a, b)); na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return d / (na * nb + 1e-30)
tot = {'cpu': [0, 0], 'metal': [0, 0], 'amd': [0, 0]}
for d in mac:
    f = next(x for x in nob['f32'] if x['prompt'] == d['prompt']); a = next(x for x in nob['int4'] if x['prompt'] == d['prompt'])
    print(f"\n== prompt {d['prompt']} ({d['prompt_len']} prompt tokens, {len(d['ids'])} positions)")
    print("   pos token            cos(CPU-int4,f32) cos(Metal,f32) cos(amd64-int4,f32) cos(Metal,CPU)  argmax==f32: cpu metal")
    rows = []
    for i, tok in enumerate(d['tokens']):
        c1 = cos(d['cpu_resid'][i], f['resid'][i]); c2 = cos(d['metal_resid'][i], f['resid'][i]); c3 = cos(a['resid'][i], f['resid'][i]); c4 = cos(d['metal_resid'][i], d['cpu_resid'][i])
        ac = d['cpu_argmax'][i] == f['argmax'][i]; am = d['metal_argmax'][i] == f['argmax'][i]; aa = a['argmax'][i] == f['argmax'][i]
        for k, v in (('cpu', ac), ('metal', am), ('amd', aa)): tot[k][0] += v; tot[k][1] += 1
        rows.append((i, tok, c1, c2, c3, c4, ac, am))
        mark = ' <' if (c4 < 0.96 or i in (14, 18, 50)) else ''
        if i < d['prompt_len'] or mark or not (ac and am):
            print(f"   {i:3d} {tok!r:16.16s} {c1:10.5f} {c2:14.5f} {c3:16.5f} {c4:15.5f}        {str(ac):5s} {str(am):5s}{mark}")
    for name, sl in (('prompt positions', rows[:d['prompt_len']]), ('generated positions', rows[d['prompt_len']:])):
        n = len(sl)
        print(f"   {name}: mean cos to f32: CPU-int4 {sum(r[2] for r in sl)/n:.5f}, Metal {sum(r[3] for r in sl)/n:.5f}; Metal nearer f32 at {sum(1 for r in sl if r[3] > r[2])}/{n}; min CPU {min(r[2] for r in sl):.5f}, min Metal {min(r[3] for r in sl):.5f}")
print("\nargmax equal to float32:", {k: f"{v[0]}/{v[1]} = {100*v[0]/v[1]:.1f}%" for k, v in tot.items()})
