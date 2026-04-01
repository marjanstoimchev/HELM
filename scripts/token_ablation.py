"""
Token Design Ablation — Generate complete LaTeX table.
Loads HELM model per dataset, masks tokens, evaluates leaf-level AUPRC.
Exports: stat_test_figures/ablation/token_ablation_table.tex

Usage:
    python scripts/token_ablation.py --gpu 0
    python scripts/token_ablation.py --gpu 0 --datasets ucm aid
"""

import os, sys, glob, yaml, types, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pandas as pd
import lightning as L
from sklearn.metrics import average_precision_score

from data.dataset_pipeline import DatasetPipeline
from data.hierarchy import create_edge_index, extract_paths
from datamodules.base_datamodule import BaseDataModule
from models.model import h_deit_base_embedding
from augmentations import Preprocess
from utils.utils import Dotdict, predict
from trainers.ssl_graph_byol import SemiSupervisedGraphBYOLModel


# ── Config ──
ALL_DATASETS = ['ucm', 'aid', 'dfc_15', 'mlrsnet', 'mured', 'hpa', 'nihchestxray', 'padchest']
DS_NAMES = {
    'ucm': 'UCM', 'aid': 'AID', 'dfc_15': 'DFC-15',
    'mlrsnet': 'MLRSNet', 'mured': 'MuReD', 'hpa': 'HPA',
    'nihchestxray': 'ChestX-14', 'padchest': 'PadChest',
}
FRACTION = 'fraction_100'
SEED = 42
MODEL_DIRS = ['saved_models', 'last_saved_models']
METHOD = 'hmlc-sl-graph-byol'

CONFIG_TEMPLATE = {
    'training': {
        'lr': 1e-4, 'head_lr': 1e-4, 'max_lr': 3e-4,
        'apply_scheduler': True, 'epochs': 100, 'min_epochs': 5,
        'patience': 5, 'lr_schedule_patience': 5,
        'accumulate_grad_batches': 5, 'deterministic': True,
        'log_every_n_steps': 1,
    },
    'dataset': {},
}

VARIANTS = {
    'HELM': 'full',
    'HELM_g': 'graph',
    'HELM_b': 'cls',
}

TOKEN_CONFIGS = ['All', 'Leaf', 'Inter']


def find_checkpoint(dataset):
    for model_dir in MODEL_DIRS:
        ckpt_dir = os.path.join(model_dir, dataset, METHOD, FRACTION, f'seed_{SEED}')
        ckpts = glob.glob(os.path.join(ckpt_dir, '*.ckpt'))
        if ckpts:
            return ckpts[0]
    return None


def make_predict_step(mode='full'):
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch['x'], batch['h_one_hot']
        x_cls, _ = self.backbone(x)
        embeddings = x_cls.mean(dim=-1)
        logits_cls = self.fc(embeddings)
        _, logits_gcn = self.sage(x_cls, self.edge_index.to(x_cls.device))
        if mode == 'full':
            logits = logits_cls + logits_gcn
        elif mode == 'cls':
            logits = logits_cls
        elif mode == 'graph':
            logits = logits_gcn
        else:
            logits = logits_cls + logits_gcn
        return {
            'logits': logits[:, :self.num_leaves],
            'labels': y[:, :self.num_leaves],
        }
    return predict_step


def get_levels(hierarchy, names):
    levels = {}
    def _walk(node, depth=0):
        for k, v in node.items():
            if k in set(names):
                levels[k] = depth
            if isinstance(v, dict) and v:
                _walk(v, depth + 1)
    _walk(hierarchy)
    return [levels.get(n, 0) for n in names]


def run_dataset(dataset, gpu):
    ckpt = find_checkpoint(dataset)
    if ckpt is None:
        print(f'  {dataset}: no checkpoint found, skipping')
        return None

    print(f'\n{"="*50}\n  {dataset.upper()}\n{"="*50}')
    print(f'  Checkpoint: {ckpt}')

    yaml_path = f'configs/dataset/{dataset}.yaml'
    with open(yaml_path) as f:
        ds_config = yaml.safe_load(f)

    lp, ip = extract_paths(ds_config['hierarchy'])
    ordered = {k: lp[k] for k in ds_config['leaf_labels']}
    final = {**ordered, **ip}
    all_names = list(final.keys())
    M = len(final)
    n_leaf = ds_config['num_classes']

    pipeline = DatasetPipeline(yaml_path, seed=SEED)
    outputs = pipeline.run_pipeline(fraction_labeled=None)
    dm = BaseDataModule(outputs, batch_size=32, num_workers=0, transforms=Preprocess())
    edge_index = create_edge_index(hierarchy=pipeline.label_to_predecessors)

    config = Dotdict({
        'training': CONFIG_TEMPLATE['training'],
        'dataset': ds_config,
    })

    bb = h_deit_base_embedding(num_classes=M, pretrained=False)
    model = SemiSupervisedGraphBYOLModel.load_from_checkpoint(
        ckpt, config=config, backbone=bb, num_leaves=M,
        learning_task='hmlc', edge_index=edge_index, map_location='cpu')
    model.eval()

    token_levels = np.array(get_levels(ds_config['hierarchy'], all_names))
    is_leaf = np.array([i < n_leaf for i in range(M)])
    intermediate_idx = [i for i in range(M) if not is_leaf[i]]
    leaf_idx = [i for i in range(M) if is_leaf[i]]

    orig_cls = model.backbone.cls_tokens.data.clone()
    orig_predict_step = model.predict_step
    trainer = L.Trainer(accelerator='gpu', devices=[gpu],
                        logger=False, enable_progress_bar=False)

    results = {}
    for variant, mode in VARIANTS.items():
        for token_cfg in TOKEN_CONFIGS:
            # Set inference path
            model.predict_step = types.MethodType(make_predict_step(mode), model)
            model.backbone.cls_tokens.data = orig_cls.clone()

            # Mask tokens
            if token_cfg == 'Leaf':
                for t in intermediate_idx:
                    model.backbone.cls_tokens.data[0, t, :] = 0.0
            elif token_cfg == 'Inter':
                for t in leaf_idx:
                    model.backbone.cls_tokens.data[0, t, :] = 0.0

            Y = predict(trainer, model, dm)
            auprc = average_precision_score(Y['y_true'], Y['y_scores'], average='macro')
            results[(variant, token_cfg)] = auprc
            print(f'  {variant:8s} {token_cfg:5s}  AUPRC={auprc:.4f}')

            # Restore
            model.backbone.cls_tokens.data = orig_cls.clone()
            model.predict_step = orig_predict_step

    del model, trainer, bb
    torch.cuda.empty_cache()
    return results


def generate_latex(all_results):
    """Generate the LaTeX table."""
    datasets_with_data = [ds for ds in ALL_DATASETS if ds in all_results]

    lines = []
    lines.append(r'\begin{table*}[ht!]')
    lines.append(r'\centering')
    lines.append(r'\caption{Inference-time token design ablation. All values report leaf-level '
                 r'AU$\overline{\textrm{PRC}}$. For each dataset, we evaluate three inference paths '
                 r'(HELM, HELM$_g$, HELM$_b$) with all, leaf-only, or intermediate-only tokens active. '
                 r'Best per dataset in \textbf{bold}.}')
    lines.append(r'\label{tab:token_ablation}')
    lines.append(r'\scriptsize')
    lines.append(r'\setlength{\tabcolsep}{4pt}')
    lines.append(r'\begin{NiceTabular}{ll cccc cccc}')
    lines.append(r'\toprule')
    lines.append(r'& & \multicolumn{4}{c}{\textbf{Remote Sensing}} & '
                 r'\multicolumn{4}{c}{\textbf{Medical Imaging}} \\')
    lines.append(r'\cmidrule(lr){3-6} \cmidrule(lr){7-10}')

    ds_headers = ' & '.join([r'\textbf{' + DS_NAMES[ds] + '}' for ds in ALL_DATASETS])
    lines.append(r'\textbf{Method} & \textbf{Tokens} & ' + ds_headers + r' \\')
    lines.append(r'\midrule')

    # Find best "All tokens" per dataset
    best_per_ds = {}
    for ds in ALL_DATASETS:
        if ds not in all_results:
            continue
        best = max(all_results[ds].get((v, 'All'), 0) for v in VARIANTS)
        best_per_ds[ds] = best

    for vi, variant in enumerate(VARIANTS):
        if vi > 0:
            lines.append(r'\arrayrulecolor{gray!30}\midrule\arrayrulecolor{black}')

        tex_name = variant.replace('HELM_g', r'HELM$_g$').replace('HELM_b', r'HELM$_b$')

        for ti, token_cfg in enumerate(TOKEN_CONFIGS):
            row_parts = []

            # Multirow method name on first row
            if ti == 0:
                row_parts.append(r'\multirow{3}{*}{' + tex_name + '}')
            else:
                row_parts.append('')

            row_parts.append(token_cfg)

            for ds in ALL_DATASETS:
                if ds not in all_results:
                    row_parts.append('---')
                else:
                    val = all_results[ds].get((variant, token_cfg))
                    if val is None:
                        row_parts.append('---')
                    else:
                        is_best = (token_cfg == 'All' and
                                   abs(val - best_per_ds.get(ds, -1)) < 1e-4)
                        if is_best:
                            row_parts.append(r'\textbf{' + f'{val:.3f}' + '}')
                        else:
                            row_parts.append(f'{val:.3f}')

            row = ' & '.join(row_parts) + r' \\'

            # Highlight HELM rows
            if variant == 'HELM':
                row = r'\rowcolor{blue!10} ' + row

            lines.append(row)

    lines.append(r'\bottomrule')
    lines.append(r'\end{NiceTabular}')
    lines.append(r'\end{table*}')

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Specific datasets to run (default: all)')
    args = parser.parse_args()

    datasets = args.datasets if args.datasets else ALL_DATASETS
    all_results = {}

    for ds in datasets:
        result = run_dataset(ds, args.gpu)
        if result is not None:
            all_results[ds] = result

    # Generate LaTeX
    latex = generate_latex(all_results)

    os.makedirs('stat_test_figures/ablation', exist_ok=True)
    out_path = 'stat_test_figures/ablation/token_ablation_table.tex'
    with open(out_path, 'w') as f:
        f.write(latex)
    print(f'\n{"="*50}')
    print(f'LaTeX table saved to: {out_path}')
    print(f'{"="*50}\n')
    print(latex)

    # Also save CSV for reference
    rows = []
    for ds, results in all_results.items():
        for (variant, token_cfg), auprc in results.items():
            rows.append({
                'dataset': ds, 'variant': variant,
                'tokens': token_cfg, 'auprc': round(auprc, 4),
            })
    df = pd.DataFrame(rows)
    csv_path = 'stat_test_figures/ablation/token_ablation.csv'
    df.to_csv(csv_path, index=False)
    print(f'CSV saved to: {csv_path}')


if __name__ == '__main__':
    main()
