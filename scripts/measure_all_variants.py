"""
Measure params, throughput, latency, memory for ALL variants × ALL datasets.
Outputs a complete LaTeX table.

Usage:
    python scripts/measure_all_variants.py --gpu 0
    python scripts/measure_all_variants.py --gpu 0 --datasets ucm aid
"""

import os, sys, glob, yaml, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pandas as pd

from data.dataset_pipeline import DatasetPipeline
from data.hierarchy import create_edge_index, extract_paths
from models.model import h_deit_base_embedding
from utils.utils import Dotdict
from trainers.supervised import SupervisedModel
from trainers.ssl_graph import GraphBasedModel
from trainers.ssl_byol import SemiSupervisedBYOLModel
from trainers.ssl_graph_byol import SemiSupervisedGraphBYOLModel

ALL_DATASETS = ['ucm', 'aid', 'dfc_15', 'mlrsnet',
                'mured', 'hpa', 'nihchestxray', 'padchest']
DS_NAMES = {'ucm': 'UCM', 'aid': 'AID', 'dfc_15': 'DFC-15',
            'mlrsnet': 'MLRSNet', 'mured': 'MuReD', 'hpa': 'HPA',
            'nihchestxray': 'ChestX-14', 'padchest': 'PadChest'}

VARIANTS = {
    'MLC':      {'trainer': SupervisedModel,              'graph': False, 'byol': False, 'use_leaf_only': True},
    'HMLC':     {'trainer': SupervisedModel,              'graph': False, 'byol': False, 'use_leaf_only': False},
    'HELM_g':   {'trainer': GraphBasedModel,              'graph': True,  'byol': False, 'use_leaf_only': False},
    'HELM_b':   {'trainer': SemiSupervisedBYOLModel,      'graph': False, 'byol': True,  'use_leaf_only': False},
    'HELM':     {'trainer': SemiSupervisedGraphBYOLModel,  'graph': True,  'byol': True,  'use_leaf_only': False},
}

TEX_NAMES = {
    'MLC': r'MLC$^\dagger$', 'HMLC': 'HMLC',
    'HELM_g': r'HELM$_g$', 'HELM_b': r'HELM$_b$', 'HELM': 'HELM',
}

CONFIG = {
    'training': {
        'lr': 1e-4, 'head_lr': 1e-4, 'max_lr': 3e-4,
        'apply_scheduler': True, 'epochs': 100, 'min_epochs': 5,
        'patience': 5, 'lr_schedule_patience': 5,
        'accumulate_grad_batches': 5, 'deterministic': True,
        'log_every_n_steps': 1,
    },
}

IMG_SIZE = (3, 224, 224)
BATCH_SIZE = 16
N_WARMUP = 5
N_RUNS = 20


def measure_variant(variant_name, variant_cfg, M, n_leaf, edge_index, ds_config, gpu):
    """Build model, measure params + throughput + memory."""
    device = torch.device(f'cuda:{gpu}')

    num_cls = n_leaf if variant_cfg['use_leaf_only'] else M
    bb = h_deit_base_embedding(num_classes=num_cls, pretrained=False)
    ei = edge_index if variant_cfg['graph'] else None

    config = Dotdict({
        'training': CONFIG['training'],
        'dataset': ds_config,
    })

    model = variant_cfg['trainer'](
        config=config, backbone=bb, num_leaves=num_cls,
        learning_task='hmlc' if not variant_cfg['use_leaf_only'] else 'mlc',
        edge_index=ei)

    # Params
    total_params = sum(p.numel() for p in model.parameters())

    # Throughput + latency
    model = model.to(device).eval()
    dummy = torch.randn(BATCH_SIZE, *IMG_SIZE, device=device)

    with torch.no_grad():
        for _ in range(N_WARMUP):
            model.backbone(dummy)
    torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(N_RUNS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.backbone(dummy)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

    avg_time = np.mean(times)
    throughput = BATCH_SIZE / avg_time
    latency_ms = avg_time / BATCH_SIZE * 1000

    # Memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(gpu)
    with torch.no_grad():
        model.backbone(dummy)
    peak_mb = torch.cuda.max_memory_allocated(gpu) / 1e6

    model = model.cpu()
    del model, bb, dummy
    torch.cuda.empty_cache()

    return {
        'params_m': total_params / 1e6,
        'throughput': throughput,
        'latency_ms': latency_ms,
        'peak_mb': peak_mb,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--datasets', nargs='+', default=None)
    args = parser.parse_args()

    datasets = args.datasets if args.datasets else ALL_DATASETS
    all_rows = []

    for ds in datasets:
        yaml_path = f'configs/dataset/{ds}.yaml'
        if not os.path.exists(yaml_path):
            print(f'{ds}: config missing, skipping')
            continue

        with open(yaml_path) as f:
            ds_config = yaml.safe_load(f)

        lp, ip = extract_paths(ds_config['hierarchy'])
        ordered = {k: lp[k] for k in ds_config['leaf_labels']}
        final = {**ordered, **ip}
        M = len(final)
        n_leaf = ds_config['num_classes']

        pipeline = DatasetPipeline(yaml_path, seed=42)
        pipeline.run_pipeline(fraction_labeled=None)
        edge_index = create_edge_index(
            hierarchy=pipeline.label_to_predecessors)

        print(f'\n{"="*50}\n  {DS_NAMES[ds]} (M={M}, leaf={n_leaf})\n{"="*50}')

        for v_name, v_cfg in VARIANTS.items():
            try:
                result = measure_variant(
                    v_name, v_cfg, M, n_leaf, edge_index,
                    ds_config, args.gpu)
                row = {
                    'dataset': ds,
                    'Dataset': DS_NAMES[ds],
                    'M': M,
                    'n_leaf': n_leaf,
                    'Variant': v_name,
                    **result,
                }
                all_rows.append(row)
                print(f'  {v_name:8s}  {result["params_m"]:7.1f}M  '
                      f'{result["throughput"]:6.0f} img/s  '
                      f'{result["latency_ms"]:5.1f} ms  '
                      f'{result["peak_mb"]:6.0f} MB')
            except Exception as e:
                print(f'  {v_name}: ERROR - {e}')

    df = pd.DataFrame(all_rows)

    # Save CSV
    os.makedirs('stat_test_figures/computational', exist_ok=True)
    csv_path = 'stat_test_figures/computational/all_variants.csv'
    df.to_csv(csv_path, index=False)
    print(f'\nCSV saved to {csv_path}')

    # Generate LaTeX
    print('\n' + '='*60)
    print('LaTeX table:')
    print('='*60 + '\n')

    # Pivot for each metric
    params_piv = df.pivot(index='dataset', columns='Variant', values='params_m')
    tp_piv = df.pivot(index='dataset', columns='Variant', values='throughput')
    lat_piv = df.pivot(index='dataset', columns='Variant', values='latency_ms')
    mem_piv = df.pivot(index='dataset', columns='Variant', values='peak_mb')

    v_order = ['MLC', 'HMLC', 'HELM_g', 'HELM_b', 'HELM']

    lines = []
    lines.append(r'\begin{table*}[ht!]')
    lines.append(r'\centering')
    lines.append(r'\caption{Comprehensive computational analysis. '
                 r'Params: total parameters (millions). '
                 r'Throughput (img/s), latency (ms/img), and peak GPU memory (MB) '
                 r'measured on a single A100 GPU (batch 16). '
                 r'$^\dagger$MLC uses leaf tokens only ($M_\text{leaf}$); '
                 r'all others use all $M$ hierarchy tokens. '
                 r'19.3 GFLOPs at inference for all variants '
                 r'(BYOL inactive).}')
    lines.append(r'\label{tab:computational}')
    lines.append(r'\scriptsize')
    lines.append(r'\setlength{\tabcolsep}{2.5pt}')
    lines.append(r'\begin{NiceTabular}{l r '
                 r'rrrrr '   # params
                 r'rrrrr '   # img/s
                 r'rrrrr}')  # MB
    lines.append(r'\toprule')
    lines.append(r'& & \multicolumn{5}{c}{\textbf{Params (M)}} '
                 r'& \multicolumn{5}{c}{\textbf{Throughput (img/s)}} '
                 r'& \multicolumn{5}{c}{\textbf{Peak Memory (MB)}} \\')
    lines.append(r'\cmidrule(lr){3-7} \cmidrule(lr){8-12} \cmidrule(lr){13-17}')

    # Column headers
    hdr = r'\textbf{Dataset} & $M$'
    for _ in range(3):
        for v in v_order:
            tex = TEX_NAMES[v]
            hdr += f' & \\textbf{{{tex}}}'
    hdr += r' \\'
    lines.append(hdr)
    lines.append(r'\midrule')

    rs_ds = [d for d in datasets if d in ['ucm', 'aid', 'dfc_15', 'mlrsnet']]
    med_ds = [d for d in datasets if d in ['mured', 'hpa', 'nihchestxray', 'padchest']]

    for label, ds_list in [('Remote Sensing', rs_ds),
                           ('Medical Imaging', med_ds)]:
        if not ds_list:
            continue
        lines.append(r'\multicolumn{17}{l}{\textit{' + label + r'}} \\')
        lines.append(r'\midrule')

        for ds in ds_list:
            sub = df[df['dataset'] == ds]
            if sub.empty:
                continue
            M = sub.iloc[0]['M']
            row = f'{DS_NAMES[ds]} & {M}'

            # Params
            for v in v_order:
                s = sub[sub['Variant'] == v]
                if not s.empty:
                    row += f' & {s.iloc[0]["params_m"]:.1f}'
                else:
                    row += ' & ---'
            # Throughput
            for v in v_order:
                s = sub[sub['Variant'] == v]
                if not s.empty:
                    row += f' & {s.iloc[0]["throughput"]:.0f}'
                else:
                    row += ' & ---'
            # Memory
            for v in v_order:
                s = sub[sub['Variant'] == v]
                if not s.empty:
                    row += f' & {s.iloc[0]["peak_mb"]:.0f}'
                else:
                    row += ' & ---'

            row += r' \\'

            # Highlight HELM row
            if True:  # all rows same, no highlight needed
                lines.append(row)

        if label == 'Remote Sensing':
            lines.append(r'\midrule')

    lines.append(r'\bottomrule')
    lines.append(r'\end{NiceTabular}')
    lines.append(r'\end{table*}')

    latex = '\n'.join(lines)
    tex_path = 'stat_test_figures/computational/computational_table.tex'
    with open(tex_path, 'w') as f:
        f.write(latex)
    print(latex)
    print(f'\nLaTeX saved to {tex_path}')


if __name__ == '__main__':
    main()
