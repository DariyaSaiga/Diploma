import os
import pickle
import numpy as np

BASE = "/content/MELD.Features.Models/features"

FILES = [
    "text_emotion.pkl",
    "text_sentiment.pkl",
    "audio_emotion.pkl",
    "audio_sentiment.pkl",
    "bimodal_sentiment.pkl",
    "data_emotion.p",
    "data_sentiment.p"
]


def inspect(obj, name):
    print("\n====================")
    print("FILE/OBJ:", name)
    print("TYPE:", type(obj))

    if isinstance(obj, dict):
        print("DICT KEYS:", list(obj.keys())[:10])

        # try show sample label
        for k in list(obj.keys())[:3]:
            print(" SAMPLE KEY:", k)
            print(" SAMPLE VALUE TYPE:", type(obj[k]))
            print(" SAMPLE VALUE:", str(obj[k])[:200])

    elif isinstance(obj, (list, tuple)):
        print("LENGTH:", len(obj))

        for i, x in enumerate(obj[:3]):
            print(f" [{i}] TYPE:", type(x))
            print(" VALUE:", str(x)[:200])

    elif isinstance(obj, np.ndarray):
        print("SHAPE:", obj.shape)
        print("DTYPE:", obj.dtype)
        print("SAMPLE:", obj.flatten()[:10])

    else:
        print("VALUE:", obj)


for file in FILES:

    path = os.path.join(BASE, file)

    if not os.path.exists(path):
        continue

    try:
        with open(path, "rb") as f:
            data = pickle.load(f)

        inspect(data, file)

    except Exception as e:
        print("\nERROR in", file)
        print(e)