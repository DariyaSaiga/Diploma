import h5py
import numpy as np

FILE_PATH = "/Users/Лейла/DesktopDiploma/mosei.hdf5"
# "/Users/dariyaablanova/Desktop/unic_work/Diploma/mosei.hdf5"

print("=" * 60)
print("ШАГ 1 — Верхнеуровневые группы (модальности)")
print("Это как папки на первом уровне внутри файла")
print("=" * 60)

with h5py.File(FILE_PATH, "r") as f:
    top_keys = list(f.keys())
    print(f"\nНайдено групп: {len(top_keys)}")
    for k in top_keys:
        print(f"  {k}")


print("\n" + "=" * 60)
print("ШАГ 2 — Для каждой группы: сколько сэмплов и какой shape")
print("Берём ОДИН пример из каждой группы")
print("=" * 60)

with h5py.File(FILE_PATH, "r") as f:
    for group_name in f.keys():
        group = f[group_name]

        sample_ids = list(group.keys())
        n_samples = len(sample_ids)
        first_id = sample_ids[0]
        first_sample = group[first_id]

        print(f"\n--- {group_name} ---")
        print(f"  Количество сэмплов: {n_samples}")
        print(f"  Пример ID: {first_id}")

        for key_inside in first_sample.keys():
            data = first_sample[key_inside]
            print(f"  Внутри каждого сэмпла есть: '{key_inside}'")
            print(f"    форма:  {data.shape}")
            print(f"    тип:    {data.dtype}")

            if data.dtype.kind in ('f', 'i', 'u'):
                sample_vals = np.array(data[:1]).flatten()
                print(f"    пример: {sample_vals[:10]}")
            elif data.dtype.kind in ('S', 'O', 'U'):
                sample_vals = np.array(data[:1]).flatten()
                print(f"    пример (текст): {sample_vals[:5]}")


print("\n" + "=" * 60)
print("ШАГ 3 — Расшифровка меток (All Labels)")
print("Смотрим что означают 7 чисел")
print("=" * 60)

EMOTION_NAMES = [
    "sentiment (настроение: -3 до +3)",
    "happy (счастье)",
    "sad (грусть)",
    "anger (злость)",
    "surprise (удивление)",
    "disgust (отвращение)",
    "fear (страх)",
]

with h5py.File(FILE_PATH, "r") as f:
    if "All Labels" in f:
        labels_group = f["All Labels"]
        sample_ids = list(labels_group.keys())

        print(f"\nВсего меток: {len(sample_ids)}")
        print("\nПример меток для первых 3 сэмплов:\n")

        for sid in sample_ids[:3]:
            vals = np.array(labels_group[sid]["features"]).flatten()
            print(f"  Сэмпл: {sid}")
            for i, (name, val) in enumerate(zip(EMOTION_NAMES, vals)):
                print(f"    [{i}] {name}: {val:.4f}")
            print()
    else:
        print("  'All Labels' не найдено в файле")


print("\n" + "=" * 60)
print("ШАГ 4 — Совпадают ли ID сэмплов между модальностями?")
print("Это важно: у каждой модальности должны быть одни и те же сэмплы")
print("=" * 60)

with h5py.File(FILE_PATH, "r") as f:
    top_keys = list(f.keys())
    id_sets = {}

    for group_name in top_keys:
        ids = set(f[group_name].keys())
        id_sets[group_name] = ids
        print(f"\n  {group_name}: {len(ids)} сэмплов")

    if len(id_sets) > 1:
        keys_list = list(id_sets.keys())
        base = id_sets[keys_list[0]]

        print(f"\nСравниваем все с '{keys_list[0]}':")
        for name in keys_list[1:]:
            common = len(base & id_sets[name])
            only_base = len(base - id_sets[name])
            only_other = len(id_sets[name] - base)
            print(f"  {name}:")
            print(f"    Общих ID:             {common}")
            print(f"    Только в {keys_list[0]}: {only_base}")
            print(f"    Только в {name}:  {only_other}")


print("\n" + "=" * 60)
print("ШАГ 5 — Проверка выбросов в числовых модальностях")
print("Берём по 100 сэмплов из каждой группы")
print("=" * 60)

with h5py.File(FILE_PATH, "r") as f:
    for group_name in f.keys():
        group = f[group_name]
        sample_ids = list(group.keys())[:100]

        all_vals = []
        for sid in sample_ids:
            for key_inside in group[sid].keys():
                data = group[sid][key_inside]
                if data.dtype.kind in ('f', 'i', 'u'):
                    vals = np.array(data).flatten().astype(np.float32)
                    vals_clean = vals[np.isfinite(vals)]
                    all_vals.extend(vals_clean.tolist())

        if all_vals:
            arr = np.array(all_vals)
            print(f"\n  {group_name}:")
            print(f"    Мин:     {arr.min():.4f}")
            print(f"    Макс:    {arr.max():.4f}")
            print(f"    Среднее: {arr.mean():.4f}")
            print(f"    Std:     {arr.std():.4f}")

            nan_count = int(np.sum(np.isnan(arr)))
            inf_count = int(np.sum(np.isinf(arr)))
            if nan_count > 0 or inf_count > 0:
                print(f"    ПРОБЛЕМА: NaN={nan_count}, Inf={inf_count}")
            else:
                print(f"    NaN/Inf: нет, всё чисто")
