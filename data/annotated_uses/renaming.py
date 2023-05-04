import os
import shutil

for directory in os.listdir("dwug_sv/data"):
    if not directory.startswith("."):
        shutil.move(f"dwug_sv/data/{directory}/uses.csv", f"swedish/{directory.lower()}.csv")