import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoConfig,
)
from datasets import load_dataset
from tqdm import tqdm

from src.metric import target_linearity
from src.utils import write_json


# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DICT = {
    'base':  'bert-base-uncased'
}

DATASET_DICT = {
    'sst2': {
        'task_type': 'seq_cls',
        'path': 'glue', 'name': 'sst2',
        'text_cols': ['sentence'], 'label_col': 'label',
        'num_labels': 2,
    },
    'mnli': {
        'task_type': 'seq_cls',
        'path': 'glue', 'name': 'mnli',
        'text_cols': ['premise', 'hypothesis'], 'label_col': 'label',
        'num_labels': 3,
    },
    'pos': {
        'task_type': 'token_cls',
        'path': 'eriktks/conll2003', 'revision': 'convert/parquet',
        'label_col': 'pos_tags',
        'num_labels': 47,
    },
    'ner': {
        'task_type': 'token_cls',
        'path': 'eriktks/conll2003', 'revision': 'convert/parquet',
        'label_col': 'ner_tags',
        'num_labels': 9,
    },
}

MODEL_SIZE   = 'base' # sys.argv[1]                                   # 'base' | 'large'
DATASET_NAME = 'sst2' # sys.argv[2]                                   # 'sst2' | 'mnli' | 'pos' | 'ner'
N_SAMPLING   = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
PCA_DIM      = int(sys.argv[4]) if len(sys.argv) > 4 else 256

BATCH_SIZE = 64
MAX_LEN    = 128
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT       = Path(__file__).parents[2]
# ─────────────────────────────────────────────────────────────────────────────


def tokenize_seq_cls(raw_split, tokenizer, text_cols, label_col):
    def _tok(batch):
        if len(text_cols) == 1:
            enc = tokenizer(batch[text_cols[0]], truncation=True, padding='max_length', max_length=MAX_LEN)
        else:
            enc = tokenizer(batch[text_cols[0]], batch[text_cols[1]], truncation=True, padding='max_length', max_length=MAX_LEN)
        enc['labels'] = batch[label_col]
        return enc

    keep = ['input_ids', 'attention_mask', 'token_type_ids', 'labels']
    tok = raw_split.map(_tok, batched=True, remove_columns=raw_split.column_names)
    tok.set_format('torch', columns=[c for c in keep if c in tok.column_names])
    return tok


def tokenize_token_cls(raw_split, tokenizer, label_col):
    def _tok(batch):
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
    tok = raw_split.map(_tok, batched=True, remove_columns=raw_split.column_names)
    tok.set_format('torch', columns=[c for c in keep if c in tok.column_names])
    return tok


def make_loader(dataset, n_sampling=None):
    if n_sampling is not None and n_sampling < len(dataset):
        idx = np.random.choice(len(dataset), size=n_sampling, replace=False)
        dataset = Subset(dataset, idx.tolist())
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)


def load_model(model_id, num_labels, task_type, ckpt_path=None, random_init=False):
    ModelCls = (AutoModelForSequenceClassification if task_type == 'seq_cls'
                else AutoModelForTokenClassification)
    if random_init:
        config = AutoConfig.from_pretrained(model_id, num_labels=num_labels)
        model  = ModelCls.from_config(config)
    else:
        model  = ModelCls.from_pretrained(
            model_id, num_labels=num_labels, ignore_mismatched_sizes=True,
        )
        if ckpt_path is not None:
            model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    return model.to(DEVICE)


def layer_names(n_encoder_layers):
    return ['embed'] + [f'layer_{i:02d}' for i in range(n_encoder_layers)]


@torch.no_grad()
def extract_cls_hidden(model, loader):
    """
    Collects the [CLS] token hidden state at every layer for sequence classification.

    Returns:
        hidden: list of [N, hidden_size] tensors  (length = n_encoder_layers + 1)
        labels: [N] int tensor
    """
    model.eval()
    all_hidden = None
    all_labels = []

    for batch in tqdm(loader, desc='  extract', leave=False):
        labels = batch.pop('labels')
        batch  = {k: v.to(DEVICE) for k, v in batch.items()}

        out = model(**batch, output_hidden_states=True)
        cls = [h[:, 0, :].cpu() for h in out.hidden_states]

        if all_hidden is None:
            all_hidden = [[] for _ in cls]
        for i, c in enumerate(cls):
            all_hidden[i].append(c)
        all_labels.append(labels.cpu())

    return [torch.cat(h) for h in all_hidden], torch.cat(all_labels)


@torch.no_grad()
def extract_token_hidden(model, loader):
    """
    Collects hidden states at first-subword positions (label != -100) for
    token classification. N_SAMPLING sentences yield ~N_SAMPLING * avg_words tokens.

    Returns:
        hidden: list of [N_tokens, hidden_size] tensors  (length = n_encoder_layers + 1)
        labels: [N_tokens] int tensor
    """
    model.eval()
    all_hidden = None
    all_labels = []

    for batch in tqdm(loader, desc='  extract', leave=False):
        labels = batch.pop('labels')      # [B, seq_len], CPU
        mask   = labels != -100           # [B, seq_len], CPU bool
        batch  = {k: v.to(DEVICE) for k, v in batch.items()}

        out      = model(**batch, output_hidden_states=True)
        mask_dev = mask.to(DEVICE)
        tok_h    = [h[mask_dev].cpu() for h in out.hidden_states]  # each [N_tokens, hidden]

        if all_hidden is None:
            all_hidden = [[] for _ in tok_h]
        for i, t in enumerate(tok_h):
            all_hidden[i].append(t)
        all_labels.append(labels[mask].cpu())

    return [torch.cat(h) for h in all_hidden], torch.cat(all_labels)


def compute_metrics(hidden_states, labels, num_labels, pca_dim):
    Y = F.one_hot(labels, num_classes=num_labels).float().to(DEVICE)
    Y = Y - Y.mean(dim=0)

    tl_list = []
    for H in hidden_states:
        H = H.to(DEVICE)
        if H.shape[1] > pca_dim:
            _, _, V = torch.pca_lowrank(H, q=pca_dim)
            H = torch.matmul(H, V)
        tl_list.append(round(target_linearity(Y, H), 6))

    return tl_list


def main():
    model_id   = MODEL_DICT[MODEL_SIZE]
    ds_cfg     = DATASET_DICT[DATASET_NAME]
    num_labels = ds_cfg['num_labels']
    task_type  = ds_cfg['task_type']

    ckpt_dir   = ROOT / 'params' / 'bert' / MODEL_SIZE / DATASET_NAME
    result_dir = ROOT / 'result' / 'bert' / MODEL_SIZE / DATASET_NAME
    result_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(0)
    torch.manual_seed(0)

    # ── Data ──────────────────────────────────────────────────────────────────
    print('Loading and tokenizing dataset...')
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if task_type == 'seq_cls':
        raw       = load_dataset(ds_cfg['path'], ds_cfg['name'])
        tok_train = tokenize_seq_cls(raw['train'], tokenizer, ds_cfg['text_cols'], ds_cfg['label_col'])
    else:
        raw       = load_dataset(ds_cfg['path'], revision=ds_cfg['revision'])
        tok_train = tokenize_token_cls(raw['train'], tokenizer, ds_cfg['label_col'])

    loader = make_loader(tok_train, n_sampling=N_SAMPLING)

    # ── Layer names from config (no weight download) ───────────────────────────
    config = AutoConfig.from_pretrained(model_id)
    names  = layer_names(config.num_hidden_layers)  # ['embed', 'layer_00', ...]

    # ── Stages: init → pretrained → fine-tuned epochs ─────────────────────────
    epoch_ckpts = sorted(ckpt_dir.glob('epoch*.pt'))
    if not epoch_ckpts:
        print(f'Warning: no epoch checkpoints found in {ckpt_dir}')

    stages = (
        [('init',       None, True),
         ('pretrained', None, False)]
        + [(p.stem, p, False) for p in epoch_ckpts]
    )

    ret = {
        'checkpoint':       [],
        'layers':           names,
        'target_linearity': [],
    }

    for stage_name, ckpt_path, random_init in stages:
        print(f'\n{"─"*55}')
        print(f'  stage={stage_name}  model={MODEL_SIZE}  dataset={DATASET_NAME}')
        print(f'{"─"*55}')

        model = load_model(model_id, num_labels, task_type, ckpt_path=ckpt_path, random_init=random_init)

        if task_type == 'seq_cls':
            hidden, labels = extract_cls_hidden(model, loader)
        else:
            hidden, labels = extract_token_hidden(model, loader)

        tl = compute_metrics(hidden, labels, num_labels, PCA_DIM)

        for name, t in zip(names, tl):
            print(f'  {name}: tl={t:.4f}')

        ret['checkpoint'].append(stage_name)
        ret['target_linearity'].append(tl)

        del model
        torch.cuda.empty_cache()

    save_path = result_dir / 'target_linearity.json'
    write_json(ret, save_path)
    print(f'\nSaved to {save_path}')


if __name__ == '__main__':
    main()
