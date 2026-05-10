import pickle

with open("mosei_emotion_aligned_60.pkl", "rb") as f:
    data = pickle.load(f)

print("DATA TYPE:", type(data))
print("KEYS:", data.keys())

train = data["train"]
print(train.keys())
for k in train.keys():
    print(k, type(train[k]), len(train[k]))

print("\nTRAIN TYPE:", type(train))
print("TRAIN LENGTH:", len(train))

print("\nFIRST SAMPLE:")
print(train['text'][0])
print(train['audio'][0])
print(train['vision'][0])
print(train['labels'][0])