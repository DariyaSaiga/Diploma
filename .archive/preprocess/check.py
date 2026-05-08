import pickle
import numpy as np

pkl_path = "mosei_cleaned.pkl"

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print("Тип верхнего объекта:", type(data))

if isinstance(data, dict):
    print("Ключи верхнего уровня:", list(data.keys()))

    for split_name, split_data in data.items():
        print(f"\n=== {split_name} ===")
        print("Тип:", type(split_data))
        print("Длина:", len(split_data))

        if len(split_data) > 0:
            sample = split_data[0]
            print("Тип одного sample:", type(sample))

            if isinstance(sample, dict):
                print("Ключи sample:", list(sample.keys()))
                for k, v in sample.items():
                    if hasattr(v, "shape"):
                        print(f"  {k}: type={type(v)}, shape={v.shape}, dtype={getattr(v, 'dtype', None)}")
                    else:
                        print(f"  {k}: type={type(v)}, value={v}")