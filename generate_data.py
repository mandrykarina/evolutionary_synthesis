"""Повторить исходный синтетический экземпляр; запускается отдельно."""
from common import BASE, save_json
from model import generate

if __name__ == "__main__":
    save_json(BASE / "data" / "procurement.json", generate(seed=2650717))
    print("Data regenerated with seed 2650717")
