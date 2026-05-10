import pickle
import torch
from torch.utils.data import Dataset, DataLoader


class BottleneckDataset(Dataset):
    """Multimodal dataset for emotion recognition with dynamic padding."""

    def __init__(self, samples, tokenizer, max_text_len=128, max_audio_len=100, max_visual_len=100):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.max_audio_len = max_audio_len
        self.max_visual_len = max_visual_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Text
        tokens = self.tokenizer(
            sample['text'],
            padding='max_length',
            truncation=True,
            max_length=self.max_text_len,
            return_tensors='pt',
        )
        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)

        # Audio: (T, audio_dim) → pad to max_audio_len
        audio = torch.tensor(sample['audio'], dtype=torch.float32)
        audio_len = min(audio.shape[0], self.max_audio_len)
        audio_padded = torch.zeros(self.max_audio_len, audio.shape[1])
        audio_padded[:audio_len] = audio[:audio_len]
        audio_mask = torch.zeros(self.max_audio_len, dtype=torch.bool)
        audio_mask[:audio_len] = True

        # Visual: (T, visual_dim) → pad to max_visual_len
        visual = torch.tensor(sample['visual'], dtype=torch.float32)
        visual_len = min(visual.shape[0], self.max_visual_len)
        visual_padded = torch.zeros(self.max_visual_len, visual.shape[1])
        visual_padded[:visual_len] = visual[:visual_len]
        visual_mask = torch.zeros(self.max_visual_len, dtype=torch.bool)
        visual_mask[:visual_len] = True

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'audio': audio_padded,
            'audio_mask': audio_mask,
            'visual': visual_padded,
            'visual_mask': visual_mask,
            'labels': torch.tensor(sample['label'], dtype=torch.long),
        }


def collate_fn(batch):
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'audio': torch.stack([item['audio'] for item in batch]),
        'audio_mask': torch.stack([item['audio_mask'] for item in batch]),
        'visual': torch.stack([item['visual'] for item in batch]),
        'visual_mask': torch.stack([item['visual_mask'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
    }


def make_bottleneck_loaders(data_path, batch_size=32, max_text_len=128,
                            max_audio_len=100, max_visual_len=100, num_workers=2):
    """Load pickle data and create train/val/test DataLoaders."""
    from transformers import BertTokenizer

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_dataset = BottleneckDataset(
        data['train'], tokenizer, max_text_len, max_audio_len, max_visual_len
    )
    val_dataset = BottleneckDataset(
        data['val'], tokenizer, max_text_len, max_audio_len, max_visual_len
    )
    test_dataset = BottleneckDataset(
        data['test'], tokenizer, max_text_len, max_audio_len, max_visual_len
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
