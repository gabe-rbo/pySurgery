with open("pysurgery/core/complexes.py", "r") as f:
    lines = f.readlines()

start_idx = -1
end_idx = -1
for i, line in enumerate(lines):
    if "def quick_mapper(" in line:
        start_idx = i
        break

if start_idx != -1:
    for i in range(start_idx + 1, len(lines)):
        if "class CWComplex" in lines[i]:
            end_idx = i
            break

method_lines = lines[start_idx:end_idx]
del lines[start_idx:end_idx]

target_idx = -1
for i, line in enumerate(lines):
    if "def _rank_mod_p" in line:
        target_idx = i
        break

lines = lines[:target_idx] + ["\n"] + method_lines + ["\n"] + lines[target_idx:]

with open("pysurgery/core/complexes.py", "w") as f:
    f.writelines(lines)
