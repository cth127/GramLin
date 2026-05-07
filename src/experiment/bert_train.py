import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from tqdm import tqdm

from src.utils import write_json


MODEL_DICT = {
    'base':  ('bert-base-uncased',  32)
}

DATASET_DICT = {
    'sst2': {
        'task_type': 'seq_cls',
        'path': 'glue', 'name': 'sst2',
        'text_cols': ['sentence'], 'label_col': 'label',
        'num_labels': 2, 'eval_split': 'validation',
    },
    'mnli': {
        'task_type': 'seq_cls',
        'path': 'glue', 'name': 'mnli',
        'text_cols': ['premise', 'hypothesis'], 'label_col': 'label',
        'num_labels': 3, 'eval_split': 'validation_matched',
    },
    'pos': {
        'task_type': 'token_cls',
        'path': 'eriktks/conll2003', 'revision': 'convert/parquet',
        'label_col': 'pos_tags',
        'num_labels': 47, 'eval_split': 'validation',
    },
    'ner': {
        'task_type': 'token_cls',
        'path': 'eriktks/conll2003', 'revision': 'convert/parquet',
        'label_col': 'ner_tags',
        'num_labels': 9, 'eval_split': 'validation',
    },
}

EPOCHS       = 3
MAX_LEN      = 128
LR           = 2e-5
WARMUP_RATIO = 0.1
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT         = Path(__file__).parents[2]


def tokenize_token_cls(dataset, tokenizer, label_col):
    def _tokenize(batch):
        enc = tokenizer(
            batch['tokens'],
            is_split_into_words=True,
            truncation=True, padding='max_length', max_length=MAX_LEN,
        )
        all_labels = []
        for i, labels in enumerate(batch[label_col]):
            word_ids = enc.word_ids(batch_index=i)
            aligned, prev_wid = [], None
            for wid in word_ids:
                if wid is None:
                    aligned.append(-100)
                elif wid != prev_wid:
                    aligned.append(labels[wid])
                else:
                    aligned.append(-100)
                prev_wid = wid
            all_labels.append(aligned)
        enc['labels'] = all_labels
        return enc

    keep = ['input_ids', 'attention_mask', 'token_type_ids', 'labels']
    tokenized = dataset.map(_tokenize, batched=True, remove_columns=dataset.column_names)
    tokenized.set_format('torch', columns=[c for c in keep if c in tokenized.column_names])
    return tokenized


def tokenize(dataset, tokenizer, text_cols, label_col):
    def _tokenize(batch):
        if len(text_cols) == 1:
            enc = tokenizer(
                batch[text_cols[0]],
                truncation=True, padding='max_length', max_length=MAX_LEN,
            )
        else:
            enc = tokenizer(
                batch[text_cols[0]], batch[text_cols[1]],
                truncation=True, padding='max_length', max_length=MAX_LEN,
            )
        enc['labels'] = batch[label_col]
        return enc

    keep = ['input_ids', 'attention_mask', 'token_type_ids', 'labels']
    tokenized = dataset.map(_tokenize, batched=True, remove_columns=dataset.column_names)
    tokenized.set_format('torch', columns=[c for c in keep if c in tokenized.column_names])
    return tokenized


def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in tqdm(loader, desc='  train', leave=False):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        n = batch['labels'].size(0)
        total_loss += outputs.loss.item() * n
        correct    += (outputs.logits.argmax(-1) == batch['labels']).sum().item()
        total      += n
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for batch in tqdm(loader, desc='  eval ', leave=False):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        n = batch['labels'].size(0)
        total_loss += outputs.loss.item() * n
        correct    += (outputs.logits.argmax(-1) == batch['labels']).sum().item()
        total      += n
    return total_loss / total, correct / total


def train_epoch_token(model, loader, optimizer, scheduler):
    model.train()
    total_loss, correct, total, total_seqs = 0.0, 0, 0, 0
    for batch in tqdm(loader, desc='  train', leave=False):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        labels = batch['labels']
        mask = labels != -100
        bs = labels.size(0)
        total_seqs += bs
        total_loss += outputs.loss.item() * bs
        correct    += (outputs.logits.argmax(-1)[mask] == labels[mask]).sum().item()
        total      += mask.sum().item()
    return total_loss / total_seqs, correct / total


@torch.no_grad()
def evaluate_token(model, loader):
    model.eval()
    total_loss, correct, total, total_seqs = 0.0, 0, 0, 0
    for batch in tqdm(loader, desc='  eval ', leave=False):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        labels = batch['labels']
        mask = labels != -100
        bs = labels.size(0)
        total_seqs += bs
        total_loss += outputs.loss.item() * bs
        correct    += (outputs.logits.argmax(-1)[mask] == labels[mask]).sum().item()
        total      += mask.sum().item()
    return total_loss / total_seqs, correct / total


def run_token(model_size, dataset_name):
    model_id, batch_size = MODEL_DICT[model_size]
    ds_cfg = DATASET_DICT[dataset_name]

    ckpt_dir   = ROOT / 'params'  / 'bert' / model_size / dataset_name
    result_dir = ROOT / 'result'  / 'bert' / model_size / dataset_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'  model={model_id}  dataset={dataset_name}')
    print(f'{"="*60}')

    raw = load_dataset(ds_cfg['path'], revision=ds_cfg['revision'])
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    train_tok = tokenize_token_cls(raw['train'],               tokenizer, ds_cfg['label_col'])
    eval_tok  = tokenize_token_cls(raw[ds_cfg['eval_split']], tokenizer, ds_cfg['label_col'])

    train_loader = DataLoader(train_tok, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    eval_loader  = DataLoader(eval_tok,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = AutoModelForTokenClassification.from_pretrained(
        model_id, num_labels=ds_cfg['num_labels']
    ).to(DEVICE)

    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    optimizer    = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    history   = []
    best_acc  = 0.0
    best_epoch = 1

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch_token(model, train_loader, optimizer, scheduler)
        eval_loss,  eval_acc  = evaluate_token(model, eval_loader)

        print(
            f'Epoch {epoch}/{EPOCHS} | '
            f'train loss {train_loss:.4f} acc {train_acc:.4f} | '
            f'eval  loss {eval_loss:.4f} acc {eval_acc:.4f}'
        )

        history.append({
            'epoch': epoch,
            'train_loss': round(train_loss, 6), 'train_acc': round(train_acc, 6),
            'eval_loss':  round(eval_loss,  6), 'eval_acc':  round(eval_acc,  6),
        })

        torch.save(model.state_dict(), ckpt_dir / f'epoch{epoch:03d}.pt')

        if eval_acc > best_acc:
            best_acc   = eval_acc
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_dir / 'best.pt')

    (ckpt_dir / 'best.pt').rename(ckpt_dir / f'best_epoch{best_epoch:03d}.pt')
    torch.save(model.state_dict(), ckpt_dir / 'final.pt')
    write_json(history, result_dir / 'metrics.json')

    print(f'Done. Best eval acc: {best_acc:.4f} (epoch {best_epoch})')


def run(model_size, dataset_name):
    model_id, batch_size = MODEL_DICT[model_size]
    ds_cfg = DATASET_DICT[dataset_name]

    ckpt_dir   = ROOT / 'params'  / 'bert' / model_size / dataset_name
    result_dir = ROOT / 'result'  / 'bert' / model_size / dataset_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'  model={model_id}  dataset={dataset_name}')
    print(f'{"="*60}')

    # ── Data ──────────────────────────────────────────────────────
    raw = load_dataset(ds_cfg['path'], ds_cfg['name'])
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    train_tok = tokenize(raw['train'],            tokenizer, ds_cfg['text_cols'], ds_cfg['label_col'])
    eval_tok  = tokenize(raw[ds_cfg['eval_split']], tokenizer, ds_cfg['text_cols'], ds_cfg['label_col'])

    train_loader = DataLoader(train_tok, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    eval_loader  = DataLoader(eval_tok,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=ds_cfg['num_labels']
    ).to(DEVICE)

    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    optimizer    = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Training loop ─────────────────────────────────────────────
    history   = []
    best_acc  = 0.0
    best_epoch = 1

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler)
        eval_loss,  eval_acc  = evaluate(model, eval_loader)

        print(
            f'Epoch {epoch}/{EPOCHS} | '
            f'train loss {train_loss:.4f} acc {train_acc:.4f} | '
            f'eval  loss {eval_loss:.4f} acc {eval_acc:.4f}'
        )

        history.append({
            'epoch': epoch,
            'train_loss': round(train_loss, 6), 'train_acc': round(train_acc, 6),
            'eval_loss':  round(eval_loss,  6), 'eval_acc':  round(eval_acc,  6),
        })

        torch.save(model.state_dict(), ckpt_dir / f'epoch{epoch:03d}.pt')

        if eval_acc > best_acc:
            best_acc   = eval_acc
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_dir / 'best.pt')

    # rename best checkpoint to include epoch number
    (ckpt_dir / 'best.pt').rename(ckpt_dir / f'best_epoch{best_epoch:03d}.pt')
    torch.save(model.state_dict(), ckpt_dir / 'final.pt')
    write_json(history, result_dir / 'metrics.json')

    print(f'Done. Best eval acc: {best_acc:.4f} (epoch {best_epoch})')


def main():
    sizes    = sys.argv[1].split(',') if len(sys.argv) > 1 else list(MODEL_DICT.keys())
    datasets = sys.argv[2].split(',') if len(sys.argv) > 2 else list(DATASET_DICT.keys())

    for size in sizes:
        for ds in datasets:
            if DATASET_DICT[ds]['task_type'] == 'seq_cls':
                run(size, ds)
            else:
                run_token(size, ds)


if __name__ == '__main__':
    main()
