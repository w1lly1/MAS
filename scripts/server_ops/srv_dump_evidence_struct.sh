#!/bin/bash
# 看一个 run 的二次分析产物**到底长什么样**（顶层键 + 谁装了候选），一次问清，避免手写读法再数错。
set -u
cd /root/autodl-tmp/MAS || exit 1
venv/bin/python - <<'PY'
import json, glob, os
rel = open("reports/held_clean30_runs.txt", encoding="utf-8").read().split()[0]
files = sorted(glob.glob(os.path.join("reports/analysis", rel, "second_pass", "**", "*.json"), recursive=True))
print("run:", rel)
print("文件:", [os.path.basename(f) for f in files])
for f in files[:2]:
    d = json.load(open(f, encoding="utf-8"))
    print("\n--", os.path.basename(f), "--")
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, list):
                print("   %-24s list[%d]" % (k, len(v)))
            elif isinstance(v, dict):
                print("   %-24s dict{%s}" % (k, ",".join(list(v)[:6])))
            else:
                print("   %-24s %s" % (k, str(v)[:60]))
        ev = d.get("evidence")
        if isinstance(ev, dict):
            print("   └ evidence 里的键:")
            for k, v in ev.items():
                print("       %-22s %s" % (k, ("list[%d]" % len(v)) if isinstance(v, list) else type(v).__name__))
PY
echo "STRUCT_DONE"
