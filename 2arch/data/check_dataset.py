import pickle
import numpy as np

path_bert = "/Users/dariyaablanova/Desktop/unic_work/Diploma/Diploma_clone/Diploma/datasets/aligned_50.pkl"
path_old  = "/Users/dariyaablanova/Desktop/unic_work/Diploma/Diploma_clone/Diploma/datasets/mosei_emotion_aligned_60.pkl"

with open(path_bert, 'rb') as f:
    bert_data = pickle.load(f)

with open(path_old, 'rb') as f:
    old_data = pickle.load(f)

combined = {}
for split in ['train', 'valid', 'test']:
    combined[split] = {
        'input_ids':      np.array(bert_data[split]['text_bert'])[:, 0, :],  # (N, 50)
        'attention_mask': np.array(bert_data[split]['text_bert'])[:, 1, :],  # (N, 50)
        'audio':          np.array(old_data[split]['audio']),                 # (N, 60, 74)
        'vision':         np.array(old_data[split]['vision']),                # (N, 60, 35)
        'labels':         np.array(old_data[split]['labels']),                # (N, 6)
    }

print("=== ПРОВЕРКА ===")
for split in ['train', 'valid', 'test']:
    print(f"\n--- {split} ---")
    for key, val in combined[split].items():
        print(f"  {key}: {val.shape}, dtype={val.dtype}")

save_path = "/Users/dariyaablanova/Desktop/unic_work/Diploma/Diploma_clone/Diploma/datasets/mosei_finetune_bert.pkl"
with open(save_path, 'wb') as f:
    pickle.dump(combined, f)

print(f"\n✓ Сохранён: {save_path}")