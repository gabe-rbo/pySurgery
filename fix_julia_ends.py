with open("pysurgery/bridge/surgery_backend.jl", "r") as f:
    lines = f.readlines()

out = []
keep = True
for i, line in enumerate(lines):
    if "end # module" in line or (line.strip() == "end" and i > 170 and "s_vals =" in lines[i-5:i]):
        if "end" in line and "end # module" not in line:
           out.append(line)
        break
    out.append(line)

# Let's be safer: just find the end of abelianize_group, which should be the last function.
out = []
in_abelianize = False
found_end = False
for line in lines:
    if "function abelianize_group" in line:
        in_abelianize = True
    if in_abelianize and line.strip() == "end":
        out.append(line)
        out.append("\nend # module SurgeryBackend\n")
        break
    out.append(line)

with open("pysurgery/bridge/surgery_backend.jl", "w") as f:
    f.writelines(out)
