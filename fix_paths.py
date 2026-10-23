import json
from pathlib import Path

root = Path(__file__).parent
p = root / "simulation" / "client_indices.json"
data = json.load(open(p))

def fix(path):
    tail = path.replace("\\", "/").split("processed/", 1)[1]
    return str(root / "data" / "processed" / tail)

data = [[[fix(a), b] for a, b in client] for client in data]
json.dump(data, open(p, "w"))
print("fixed", len(data), "clients")