"""Compute per-observation SINR location embeddings for PS-SPLIT from a SINR checkpoint.

You do NOT need this script to reproduce the paper: the embeddings it produces are
released as ``PSSPLITS/sinr/sinr_obs_embeddings.npz`` (use scripts/fanout_sinr_features.py).
It documents how that file was made and lets you build embeddings from another SINR
model.

Model: ``ResidualFCNet`` from SINR (Cole et al., ICML 2023, github.com/elijahcole/sinr,
MIT licence), re-implemented below so the SINR code base is not needed. The checkpoint
must be a SINR-style dict ``{'state_dict': ..., 'params': {...}}`` with
``params['model'] == 'ResidualFCNet'`` and ``params['input_enc'] == 'sin_cos'``.
The embedding of a location is the 256-d output of ``model.feats`` (the input to the
species classifier), i.e. SINR's ``return_feats=True`` output, computed in eval mode.

Location input: the ``location`` column of the PS-SPLIT metadata CSVs, a string
``"(latitude, longitude)"`` in decimal degrees (WGS84, as exported by iNaturalist).
It is rescaled to lon/180, lat/90 in [-1, 1] and encoded by one of two encodings,
chosen from ``params['input_dim']``:

* 14 (the model used in the paper): multi-scale sinusoidal encoding
      [lon, lat,
       sin(pi lon),  cos(pi lon),  sin(pi lat),
       sin(2 pi lon), cos(2 pi lon), sin(2 pi lat),
       sin(4 pi lon), cos(4 pi lon), sin(4 pi lat),
       sin(8 pi lon), cos(8 pi lon), sin(8 pi lat)]
* 4 (public SINR ``sin_cos`` models): [sin(pi lon), sin(pi lat), cos(pi lon), cos(pi lat)]

Any other input encoding (e.g. ``env``, which needs a bioclim raster) is rejected.

Usage (run from submission/):
    python scripts/compute_sinr_embeddings.py --checkpoint <sinr_model.pt> \
        --output <dir>/sinr_obs_embeddings.npz [--dataset_root <kaggle>/PSSPLITS] \
        [--reference_npz <kaggle>/PSSPLITS/sinr/sinr_obs_embeddings.npz]
"""
import argparse
import math
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sinr_common import AVES_DATASETS, enumerate_aves_samples, override_aves_paths  # noqa: E402


# ---------------------------------------------------------------------------
# SINR ResidualFCNet (layout and parameter names as in elijahcole/sinr models.py, MIT)
# ---------------------------------------------------------------------------

class ResLayer(nn.Module):
    def __init__(self, linear_size):
        super().__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        return x + y


class ResidualFCNet(nn.Module):
    def __init__(self, num_inputs, num_classes, num_filts, depth=4):
        super().__init__()
        self.inc_bias = False
        self.class_emb = nn.Linear(num_filts, num_classes, bias=self.inc_bias)
        layers = [nn.Linear(num_inputs, num_filts), nn.ReLU(inplace=True)]
        for _ in range(depth):
            layers.append(ResLayer(num_filts))
        self.feats = nn.Sequential(*layers)

    def forward(self, x):
        return self.feats(x)  # location embedding (SINR's return_feats=True)


def load_sinr(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if not isinstance(ckpt, dict) or 'state_dict' not in ckpt or 'params' not in ckpt:
        raise ValueError(f'{checkpoint_path}: expected a dict with keys state_dict, params')
    p = ckpt['params']
    if p.get('model') != 'ResidualFCNet':
        raise ValueError(f'Only ResidualFCNet checkpoints are supported, got {p.get("model")!r}')
    if p.get('input_enc') != 'sin_cos':
        raise ValueError(f'Only input_enc="sin_cos" is supported, got {p.get("input_enc")!r}')
    model = ResidualFCNet(num_inputs=p['input_dim'], num_classes=p['num_classes'],
                          num_filts=p['num_filts'], depth=p['depth'])
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.to(device).eval()
    summary = {k: p.get(k) for k in ('experiment_name', 'input_enc', 'input_dim', 'num_filts',
                                     'depth', 'loss', 'num_classes', 'num_epochs',
                                     'hard_cap_num_per_class', 'obs_file')}
    print(f'Loaded SINR checkpoint {checkpoint_path}: {summary}')
    return model, int(p['input_dim'])


# ---------------------------------------------------------------------------
# Location parsing and encoding
# ---------------------------------------------------------------------------

_LOC_RE = re.compile(r'^\(?\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\)?$')


def parse_lat_lon(location_str):
    """'(lat, lon)' -> (lat, lon) as floats; raises on anything else."""
    m = _LOC_RE.match(str(location_str).strip())
    if m is None:
        raise ValueError(f'Cannot parse location {location_str!r}; expected "(lat, lon)"')
    lat, lon = float(m.group(1)), float(m.group(2))
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        raise ValueError(f'Location {location_str!r} out of range for (lat, lon)')
    return lat, lon


def encode_locations(lonlat_scaled, input_dim):
    """lonlat_scaled: float32 tensor [N, 2] = (lon/180, lat/90)."""
    if input_dim == 14:
        pi = math.pi
        lon = lonlat_scaled[:, 0].unsqueeze(1)
        lat = lonlat_scaled[:, 1].unsqueeze(1)
        return torch.cat((lon, lat, torch.sin(lon*pi), torch.cos(lon*pi), torch.sin(lat*pi),
                          torch.sin(2*lon*pi), torch.cos(2*lon*pi), torch.sin(2*lat*pi),
                          torch.sin(4*lon*pi), torch.cos(4*lon*pi), torch.sin(4*lat*pi),
                          torch.sin(8*lon*pi), torch.cos(8*lon*pi), torch.sin(8*lat*pi)), 1)
    if input_dim == 4:
        return torch.cat((torch.sin(math.pi * lonlat_scaled),
                          torch.cos(math.pi * lonlat_scaled)), 1)
    raise ValueError(f'Unsupported SINR input_dim {input_dim} (expected 14 or 4)')


# ---------------------------------------------------------------------------
# Observation locations for the released samples
# ---------------------------------------------------------------------------

def collect_locations(datasets):
    """Return (observation_ids, location strings) for every sample of the datasets."""
    import data.aves_small as aves_small
    import data.aves_full as aves_full
    meta_dirs = {'aves_small': aves_small.aves_small_metadata_dir,
                 'aves_full': aves_full.aves_full_metadata_dir}
    loc_of = {}
    for ds in datasets:
        folds = enumerate_aves_samples(ds)
        needed = {}
        for records in folds.values():
            for _, _, taxon_id, obs_id in records:
                needed.setdefault(taxon_id, set()).add(obs_id)
        for taxon_id, obs_ids in needed.items():
            meta_path = os.path.join(meta_dirs[ds], f'metadata_{taxon_id}_updated.csv')
            df = pd.read_csv(meta_path, usecols=['observation_id', 'location'])
            df = df[df['observation_id'].isin(obs_ids)]
            for obs_id, loc in zip(df['observation_id'].tolist(), df['location'].tolist()):
                if not isinstance(loc, str) or not loc.strip():
                    raise ValueError(f'{meta_path}: empty location for observation {obs_id}')
                if obs_id in loc_of and loc_of[obs_id] != loc:
                    raise ValueError(f'observation {obs_id} has two locations: '
                                     f'{loc_of[obs_id]!r} and {loc!r}')
                loc_of[obs_id] = loc
            missing = obs_ids - set(df['observation_id'].tolist())
            if missing:
                raise KeyError(f'{meta_path}: {len(missing)} observations without a metadata row')
    obs_ids = np.array(sorted(loc_of), dtype=np.int64)
    return obs_ids, [loc_of[int(o)] for o in obs_ids]


@torch.no_grad()
def compute_embeddings(model, input_dim, locations, batch_size, device):
    lonlat = []
    for loc in locations:
        lat, lon = parse_lat_lon(loc)
        lonlat.append([lon, lat])  # SINR expects (lon, lat)
    lonlat = np.float32(np.array(lonlat))
    scaled = lonlat / np.array([180.0, 90.0], dtype=np.float32)
    out = []
    for i in range(0, len(scaled), batch_size):
        x = encode_locations(torch.from_numpy(scaled[i:i + batch_size]).to(device), input_dim)
        out.append(model(x).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def compare_to_reference(obs_ids, emb, reference_npz, atol, rtol):
    z = np.load(reference_npz, allow_pickle=False)
    ref = {int(k): v for k, v in zip(z['observation_id'].tolist(), z['emb'])}
    missing = [int(o) for o in obs_ids if int(o) not in ref]
    if missing:
        print(f'Reference: {len(missing)} observations missing (first {missing[0]})')
        return False
    ref_emb = np.stack([ref[int(o)] for o in obs_ids]).astype(np.float32)
    diff = np.abs(emb - ref_emb)
    unit = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    ref_unit = ref_emb / np.linalg.norm(ref_emb, axis=1, keepdims=True)
    cos = np.sum(unit * ref_unit, axis=1)
    ok = bool(np.allclose(emb, ref_emb, atol=atol, rtol=rtol))
    print(f'Reference {reference_npz}: {len(obs_ids)} common observations '
          f'(reference has {len(ref)}); raw max |diff| = {diff.max():.3e}, mean |diff| = '
          f'{diff.mean():.3e}; L2-normalised max |diff| = {np.abs(unit - ref_unit).max():.3e}; '
          f'min cosine = {cos.min():.8f}; allclose(atol={atol}, rtol={rtol}) = {ok}')
    return ok


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True, help='SINR checkpoint (.pt)')
    parser.add_argument('--output', required=True, help='Output .npz (keys observation_id, emb)')
    parser.add_argument('--dataset_name', choices=list(AVES_DATASETS) + ['both'], default='both',
                        help='Observations to embed (default: union of both datasets)')
    parser.add_argument('--dataset_root', default=None,
                        help='Kaggle PSSPLITS/ folder; overrides the aves_* paths in config.py')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--reference_npz', default=None,
                        help='Optional npz to compare against (e.g. the released one)')
    parser.add_argument('--atol', type=float, default=1e-4)
    parser.add_argument('--rtol', type=float, default=1e-4)
    args = parser.parse_args()

    if os.path.exists(args.output):
        raise FileExistsError(f'{args.output} exists; choose another --output')
    if args.dataset_root is not None:
        override_aves_paths(args.dataset_root)
    datasets = list(AVES_DATASETS) if args.dataset_name == 'both' else [args.dataset_name]

    model, input_dim = load_sinr(args.checkpoint, args.device)
    obs_ids, locations = collect_locations(datasets)
    print(f'Embedding {len(obs_ids)} unique observations on {args.device} ...')
    emb = compute_embeddings(model, input_dim, locations, args.batch_size, args.device)
    if not np.isfinite(emb).all():
        raise ValueError('Non-finite SINR embeddings')

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(args.output, observation_id=obs_ids, emb=emb)
    print(f'Saved {emb.shape} embeddings to {args.output}')

    if args.reference_npz is not None:
        if not compare_to_reference(obs_ids, emb, args.reference_npz, args.atol, args.rtol):
            sys.exit(1)


if __name__ == '__main__':
    main()
