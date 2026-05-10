# Diploma
Emotion recognition
в общем датасет выглядет так: (можно запустить show.py)
KEYS: dict_keys(['train', 'valid', 'test'])
dict_keys(['vision', 'audio', 'text', 'labels', 'id'])
vision: 16326
audio: 16326
text: 16326
labels: 16326

Сейчас реализован правильный BAF из статьи "Attention Bottlenecks for Multimodal Fusion" (NeurIPS 2021). Работает в два шага: сначала небольшое число learnable bottleneck-токенов собирают информацию от всех трёх модальностей через cross-attention, затем каждая модальность обновляет себя через эти сжатые токены. Это заставляет модель реально "договариваться" между модальностями, а не просто конкатенировать.


