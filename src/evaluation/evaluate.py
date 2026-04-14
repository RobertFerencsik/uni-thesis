import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from src.lstm.dataset import SpamHamDataset
from src.lstm.model import BiLSTMSpamClassifier
from src.lstm.tokenizer import SentencePieceTokenizer


DEFAULT_THRESHOLD = 0.5


def load_checkpoint(checkpoint_path: Path, tokenizer: SentencePieceTokenizer):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    info = checkpoint['model_info']

    model = BiLSTMSpamClassifier(
        **{k: info[k] for k in ['vocab_size', 'embedding_dim', 'hidden_size',
                                 'num_layers', 'dropout_rate', 'dense_hidden']},
        padding_idx=tokenizer.pad_id
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def compute_metrics(preds, labels) -> Dict[str, float]:
    preds = torch.tensor(preds, dtype=torch.int64)
    labels = torch.tensor(labels, dtype=torch.int64)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def evaluate(
    checkpoint_path: Path,
    tokenizer_path: Path,
    eval_csv_path: Path,
    batch_size: int = 64,
    device: str = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, float]:
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = SentencePieceTokenizer(str(tokenizer_path))
    model = load_checkpoint(checkpoint_path, tokenizer)
    model.to(device)

    eval_dataset = SpamHamDataset(str(eval_csv_path), tokenizer)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for token_ids, attention_mask, labels in eval_loader:
            token_ids = token_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(token_ids, attention_mask)
            probs = torch.sigmoid(logits.squeeze(1))
            batch_preds = (probs > threshold).long()

            all_preds.extend(batch_preds.cpu().tolist())
            all_labels.extend(labels.long().cpu().tolist())

    metrics = compute_metrics(all_preds, all_labels)
    metrics['num_samples'] = len(all_labels)
    metrics['threshold'] = threshold
    return metrics


def _aggregate_metric_dicts(results: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not results:
        return {}

    aggregated = {}
    keys = results[0].keys()
    for key in keys:
        values = [r[key] for r in results]
        if key in {'tp', 'tn', 'fp', 'fn', 'num_samples'}:
            aggregated[f'{key}_sum'] = int(sum(values))
        else:
            aggregated[f'{key}_mean'] = sum(values) / len(values)
            if len(values) > 1:
                variance = sum((x - aggregated[f'{key}_mean']) ** 2 for x in values) / len(values)
                aggregated[f'{key}_std'] = variance ** 0.5
    aggregated['runs'] = len(results)
    return aggregated


def evaluate_learning_curve(
    runs: Sequence[Dict[str, object]],
    tokenizer_path: Path,
    eval_csv_path: Path,
    batch_size: int = 64,
    device: Optional[str] = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, object]:
    """
    Evaluate multiple checkpoints for learning-curve experiments.

    Each run config requires:
      - corpus_size: int (or any sortable size indicator)
      - checkpoint_path: path-like
    Optional:
      - run_name: str (for repeated runs per corpus size)
    """
    grouped: Dict[object, List[Dict[str, float]]] = {}
    detailed_results: List[Dict[str, object]] = []

    for run in runs:
        corpus_size = run['corpus_size']
        checkpoint_path = Path(str(run['checkpoint_path']))
        run_name = str(run.get('run_name', checkpoint_path.stem))

        metrics = evaluate(
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
            eval_csv_path=eval_csv_path,
            batch_size=batch_size,
            device=device,
            threshold=threshold,
        )

        detailed_results.append(
            {
                'corpus_size': corpus_size,
                'run_name': run_name,
                'checkpoint_path': str(checkpoint_path),
                **metrics,
            }
        )
        grouped.setdefault(corpus_size, []).append(metrics)

    learning_curve = []
    for corpus_size in sorted(grouped):
        summary = _aggregate_metric_dicts(grouped[corpus_size])
        learning_curve.append({'corpus_size': corpus_size, **summary})

    return {
        'detailed_runs': detailed_results,
        'curve': learning_curve,
    }


def save_learning_curve_results(results: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
