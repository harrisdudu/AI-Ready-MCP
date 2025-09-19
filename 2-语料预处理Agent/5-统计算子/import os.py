import os

fs = set()
for f in os.listdir(r"C:\域外第二批\word"):
    if os.path.splitext(f)[0] in fs: print(f)
    fs.add(os.path.splitext(f)[0])
print(len(fs))

