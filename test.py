import json
from pathlib import Path

NB_PATH = Path("main.ipynb")
bak = NB_PATH.with_suffix(".ipynb.bak")
NB_PATH.replace(bak)  # backup original
nb = json.loads(bak.read_text(encoding="utf-8"))

def mk_md(cell_text):
    return {"cell_type":"markdown","metadata":{"language":"markdown"},"source":[cell_text]}

new_cells = []
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell.get("source", []))
        if "HAM10000_metadata" in src or "HAM10000_metadata" in src:
            new_cells.append(mk_md("Purpose: Read HAM10000 metadata and create a binary `labels_binary.csv` mapping MEL vs NON-MEL for images in `CleanData/HAM10000`."))
        elif "ISIC_2019_Training_GroundTruth.csv" in src or "ISIC2019" in src and "MEL" in src:
            new_cells.append(mk_md("Purpose: Read ISIC2019 training ground-truth and create a binary `labels_binary.csv` mapping MEL vs NON-MEL for images in `CleanData/ISIC2019`."))
        elif "importlib" in src and "features.lbp" in src:
            new_cells.append(mk_md("Purpose: Reload the `features.lbp` module and run a quick LBP feature extraction on a sample image to verify outputs."))
        elif "from features.glcm import extract_glcm" in src:
            new_cells.append(mk_md("Purpose: Run the GLCM feature extractor on a sample image and print the resulting feature vector length and values."))
    new_cells.append(cell)

nb["cells"] = new_cells
NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Updated {NB_PATH} (backup saved as {bak})")