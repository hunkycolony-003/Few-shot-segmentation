import glob

for f in glob.glob("pipeline/dataset_*.py"):
    with open(f, "r") as file:
        content = file.read()
    
    if "config.SAVE_MODEL = False" not in content:
        content = content.replace("config.FREEZE_BACKBONE = True", "config.FREEZE_BACKBONE = True\n    config.SAVE_MODEL = False")
        with open(f, "w") as file:
            file.write(content)
        print(f"Updated {f}")
