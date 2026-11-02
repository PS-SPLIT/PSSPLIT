"""
download_aves_metadata_photos_local_vlm.py

Step 3 of the birds-only iNat dataset pipeline — local VLM variant.

Identical to download_aves_metadata_photos.py except the Claude API is replaced
by a locally-run Qwen2.5-VL model (default: Qwen/Qwen2.5-VL-7B-Instruct).
Model weights are downloaded on first run and cached to --hf_cache_dir.

Processes each split in aves_splits.json as a unit:
  1. Generated species (top-2 by observation count) are processed first via an
     iterative download → MegaDetector → VLM loop until target_cap images are
     accepted in the majority pose, or all available metadata is exhausted.
  2. The majority pose is determined across both generated species together after
     the first VLM batch.
  3. The control species is then processed with the same pose constraint and a
     target equal to the minimum accepted count across the generated pair.

Photos are never deleted — originals go to --photo_dir, MegaDetector crops to
--crops_dir.  All pass/fail results are written as new columns in each species'
metadata CSV so the pipeline can be resumed from any point.

New metadata CSV columns added:
  md_pass, md_conf, md_bbox_px, md_reject_reason, md_crop_path,
  vlm_pass, vlm_pose, vlm_reject_reason, dataset_pass

Usage:
    python \\
        download_aves_metadata_photos_local_vlm.py \\
        --aves_splits aves_splits.json \\
        --aves_controls aves_controls.json \\
        --model_id Qwen/Qwen2.5-VL-7B-Instruct \\
        --device cuda:2 \\
        --hf_cache_dir ./hf_cache

For the most powerful available model (AWQ-quantised 72B, fits in ~40 GB):
        --model_id Qwen/Qwen2.5-VL-72B-Instruct-AWQ
"""

import os
import sys
import json
import csv
import argparse
import shutil
import time
import traceback
import random
from collections import Counter, deque
from math import ceil

import torch
import requests
from PIL import Image
from megadetector.detection.run_detector_batch import load_and_run_detector_batch

# ── MegaDetector defaults (match test_megadetector.py) ───────────────────────
MD_CONF_THRESHOLD = 0.75
MD_MIN_BBOX_PX    = 190
MD_TARGET_SIZE    = 256
MD_BBOX_PADDING   = 0.071

# ── VLM defaults ──────────────────────────────────────────────────────────────
DEFAULT_MODEL_ID = 'Qwen/Qwen2.5-VL-7B-Instruct'
DEFAULT_DEVICE   = 'cuda:0'
DEFAULT_HF_CACHE = 'hf_cache'

VLM_PROMPT = """\
You are reviewing a bird photo for dataset inclusion. Answer these 4 yes/no questions:
1. Is the animal clearly an adult bird (not a chick or obvious juvenile)?
2. Is most of the bird's body clearly visible and unobscured? (answer no if the bird is a close-up of just the head/wing/beak, OR if the body is mostly hidden by foliage, branches, or other obstructions)
3. Are there visible watermarks, logos, or overlaid text on the image?
4. Is this an unusual capture (photo of a screen, through telescope, heavily processed)?
Also classify the pose: "perched", "flying", "on-ground", "in-water", "other"
Return JSON only (no markdown): {"q1":bool,"q2":bool,"q3":bool,"q4":bool,"pose":"..."}
Accept if: q1=true, q2=true, q3=false, q4=false"""

VALID_POSES = {'perched', 'flying', 'on-ground', 'in-water', 'other'}

# ── Metadata CSV schema ───────────────────────────────────────────────────────
BASE_COLS = ['observation_id', 'iconic_taxon_id', 'taxon_id', 'subspecies_id',
             'user_id', 'research_grade', 'date_of_observation', 'location', 'photo_url']
FILTER_COLS = ['md_pass', 'md_conf', 'md_bbox_px', 'md_reject_reason', 'md_crop_path',
               'vlm_pass', 'vlm_pose', 'vlm_reject_reason', 'dataset_pass']
ALL_COLS = BASE_COLS + FILTER_COLS

MIN_VIABLE = 10  # fall back to accepting all poses if species drops below this


# =============================================================================
# Metadata CSV helpers
# =============================================================================

def meta_path(taxon_id, metadata_dir):
    return os.path.join(metadata_dir, f'metadata_{taxon_id}_updated.csv')


def load_metadata(taxon_id, metadata_dir):
    """Load metadata CSV as list of row dicts, adding any missing filter columns."""
    path = meta_path(taxon_id, metadata_dir)
    if not os.path.exists(path):
        return []
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for col in FILTER_COLS:
            if col not in row:
                row[col] = ''
    return rows


def save_metadata(rows, taxon_id, metadata_dir):
    """Write rows back to metadata CSV, always using ALL_COLS column order."""
    if not rows:
        return
    path = meta_path(taxon_id, metadata_dir)
    os.makedirs(metadata_dir, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# Row-level selectors and counters
# =============================================================================

def unprocessed_rows(rows):
    """Rows where MegaDetector has not yet been run (md_pass is empty)."""
    return [r for r in rows if r.get('md_pass', '') == '']


def md_passing_unvlmd(rows):
    """Rows that passed MD, have a crop, but VLM hasn't been run yet."""
    return [r for r in rows
            if r.get('md_pass') in ('True', True)
            and r.get('md_crop_path', '')
            and r.get('vlm_pass', '') == '']


def count_dataset_passing(rows, majority_pose):
    """Count VLM-accepted rows in the majority pose."""
    return sum(1 for r in rows
               if r.get('vlm_pass') in ('True', True)
               and r.get('vlm_pose') == majority_pose)


def compute_majority_pose(rows_list):
    """Most common VLM-accepted pose across multiple species' row lists."""
    counts = Counter()
    for rows in rows_list:
        for r in rows:
            if r.get('vlm_pass') in ('True', True) and r.get('vlm_pose'):
                counts[r['vlm_pose']] += 1
    return counts.most_common(1)[0][0] if counts else None


def mark_dataset_pass(rows, majority_pose):
    """Set dataset_pass for all VLM-evaluated rows based on majority pose."""
    for r in rows:
        if r.get('vlm_pass', '') == '':
            continue  # not yet evaluated
        if r.get('vlm_pass') in ('True', True):
            r['dataset_pass'] = str(r.get('vlm_pose') == majority_pose)
        else:
            r['dataset_pass'] = 'False'


# =============================================================================
# Image helpers
# =============================================================================

def crop_padded_square(image, bbox_norm, padding, target_size):
    """Crop a square region centred on the MegaDetector bbox and resize."""
    img_w, img_h = image.size
    x_n, y_n, w_n, h_n = bbox_norm
    x, y = x_n * img_w, y_n * img_h
    w, h = w_n * img_w, h_n * img_h
    pad_px = padding * max(w, h)
    x0, y0 = x - pad_px, y - pad_px
    x1, y1 = x + w + pad_px, y + h + pad_px
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) / 2
    left  = max(0, cx - half)
    top   = max(0, cy - half)
    right = min(img_w, cx + half)
    bot   = min(img_h, cy + half)
    aw, ah = right - left, bot - top
    if aw < ah:
        right = min(img_w, left + ah)
    elif ah < aw:
        bot = min(img_h, top + aw)
    return image.crop((int(left), int(top), int(right), int(bot))).resize(
        (target_size, target_size), Image.LANCZOS)


# =============================================================================
# Photo download
# =============================================================================

def ensure_original_url(url):
    for res in ('square', 'medium', 'small', 'large'):
        if f'/{res}.' in url:
            return url.replace(f'/{res}.', '/original.')
    return url


def download_photo(url, dest_path, retries=3):
    """Download a single photo. Returns True on success."""
    if os.path.exists(dest_path):
        return True
    url = ensure_original_url(url)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=30, stream=True)
            resp.raise_for_status()
            from io import BytesIO
            img = Image.open(BytesIO(resp.content)).convert('RGB')
            img.save(dest_path, 'JPEG', quality=90)
            return True
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
            else:
                print(f'      Download failed ({url[:60]}…): {e}')
    return False


# =============================================================================
# MegaDetector
# =============================================================================

def run_md_on_paths(image_paths, conf_thresh, min_bbox_px, target_size,
                    bbox_padding, crops_dir):
    """
    Run MegaDetector on a list of image paths.
    Always saves a crop (pass or fail on bbox size) to crops_dir.
    Returns dict: image_path -> result dict.
    """
    if not image_paths:
        return {}

    md_output = load_and_run_detector_batch('MDV5A', image_paths)
    results = {}

    for item in md_output:
        img_path = item.get('file')
        detections = item.get('detections', [])
        animal_dets = [d for d in detections if d.get('category') == '1']

        if not animal_dets:
            results[img_path] = {
                'md_pass': False, 'md_conf': 0.0, 'md_bbox_px': 0,
                'md_reject_reason': 'no_detection', 'md_crop_path': '',
            }
            continue

        best = max(animal_dets, key=lambda d: d.get('conf', 0))
        conf = best.get('conf', 0)

        try:
            img = Image.open(img_path).convert('RGB')
            img_w, img_h = img.size
            bbox_px = max(best['bbox'][2] * img_w, best['bbox'][3] * img_h)
        except Exception as e:
            results[img_path] = {
                'md_pass': False, 'md_conf': round(conf, 4), 'md_bbox_px': 0,
                'md_reject_reason': 'image_read_error', 'md_crop_path': '',
            }
            continue

        # Save crop regardless of pass/fail so VLM can inspect borderline cases
        fname = os.path.basename(img_path)
        crop_path = os.path.join(crops_dir, fname)
        os.makedirs(crops_dir, exist_ok=True)
        try:
            crop = crop_padded_square(img, best['bbox'], bbox_padding, target_size)
            crop.save(crop_path, 'JPEG', quality=90)
        except Exception:
            crop_path = ''

        if conf < conf_thresh:
            passed, reason = False, 'low_conf'
        elif bbox_px < min_bbox_px:
            passed, reason = False, 'small_bbox'
        else:
            passed, reason = True, ''

        results[img_path] = {
            'md_pass': passed,
            'md_conf': round(conf, 4),
            'md_bbox_px': int(bbox_px),
            'md_reject_reason': reason,
            'md_crop_path': crop_path if passed else '',
        }

    return results


# =============================================================================
# Local VLM (Qwen2.5-VL)
# =============================================================================

_vlm_model = None
_vlm_processor = None


def _load_vlm(model_id, device, hf_cache_dir):
    """Load model and processor once; subsequent calls are no-ops."""
    global _vlm_model, _vlm_processor
    if _vlm_model is not None:
        return
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    os.makedirs(hf_cache_dir, exist_ok=True)
    print(f'Loading VLM {model_id} on {device}  (cache: {hf_cache_dir})')
    print('First run will download weights — this may take a while…')
    _vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
        cache_dir=hf_cache_dir,
    )
    _vlm_model.eval()
    _vlm_processor = AutoProcessor.from_pretrained(model_id, cache_dir=hf_cache_dir)
    print('VLM ready.')


def call_vlm_one_local(crop_path):
    """Run the local VLM on a single crop. Returns result dict."""
    try:
        image = Image.open(crop_path).convert('RGB')
        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'image': image},
            {'type': 'text', 'text': VLM_PROMPT},
        ]}]
        text = _vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = _vlm_processor(
            text=[text], images=[image], return_tensors='pt'
        ).to(_vlm_model.device)
        with torch.no_grad():
            generated_ids = _vlm_model.generate(**inputs, max_new_tokens=200)
        raw = _vlm_processor.decode(
            generated_ids[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()

        if raw.startswith('```'):
            raw = raw.split('```')[1]
            if raw.startswith('json'):
                raw = raw[4:]
            raw = raw.strip()
        r = json.loads(raw)
        pose = r.get('pose', 'other')
        if pose not in VALID_POSES:
            pose = 'other'
        accepted = (bool(r.get('q1')) and bool(r.get('q2'))
                    and not bool(r.get('q3')) and not bool(r.get('q4')))
        if not accepted:
            if not r.get('q1'):   reason = 'not_adult'
            elif not r.get('q2'): reason = 'partial_body'
            elif r.get('q3'):     reason = 'watermark'
            elif r.get('q4'):     reason = 'unusual_capture'
            else:                 reason = 'unknown'
        else:
            reason = ''
        return {'vlm_pass': accepted, 'vlm_pose': pose, 'vlm_reject_reason': reason}
    except json.JSONDecodeError:
        return {'vlm_pass': False, 'vlm_pose': '', 'vlm_reject_reason': 'parse_error'}
    except Exception as e:
        return {'vlm_pass': False, 'vlm_pose': '',
                'vlm_reject_reason': f'vlm_error:{str(e)[:60]}'}


def run_vlm_batch(crop_paths, args):
    """Run local VLM sequentially on a list of crop paths. Returns list of result dicts."""
    _load_vlm(args.model_id, args.device, args.hf_cache_dir)
    results = []
    for i, path in enumerate(crop_paths):
        res = call_vlm_one_local(path)
        results.append(res)
        status = 'PASS' if res['vlm_pass'] else f'FAIL({res["vlm_reject_reason"]})'
        print(f'      VLM [{i+1}/{len(crop_paths)}] {os.path.basename(path)}: {status}')
    return results


# =============================================================================
# Metadata fetch (self-contained iNat API client)
# =============================================================================

def fetch_metadata_from_inat(taxon_id, metadata_dir, max_obs, observer_limit=10):
    """
    Fetch up to max_obs research-grade licensed observations from iNat for taxon_id.
    Caps any single observer at observer_limit photos (rotating out over-represented
    users as better-distributed observations are found).
    Writes the metadata CSV and returns the rows.
    """
    save_path = meta_path(taxon_id, metadata_dir)
    taxon_id = int(taxon_id)
    base_url = 'https://api.inaturalist.org/v1/observations'
    params = {
        'photos': 'true', 'per_page': '200', 'taxon_id': taxon_id,
        'taxon_is_active': 'true', 'quality_grade': 'research',
        'licensed': 'true', 'photo_licensed': 'true',
        'order': 'desc', 'order_by': 'random',
    }

    try:
        resp = requests.get(base_url, params={**params, 'page': 1}, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f'    iNat fetch failed for {taxon_id}: {e}')
        return []

    info = resp.json()
    total_pages = min(ceil(info.get('total_results', 0) / 200), 50)
    if total_pages == 0:
        print(f'    No observations found for taxon {taxon_id}')
        return []
    print(f'    taxon {taxon_id}: {total_pages} pages available, requesting {max_obs} obs…')

    observers = {}
    all_obs = deque()

    for page in range(1, total_pages + 1):
        r = None
        for attempt in range(3):
            try:
                r = requests.get(base_url, params={**params, 'page': page}, timeout=100)
                r.raise_for_status()
                break
            except Exception as e:
                r = None
                if attempt == 2:
                    print(f'    Page {page} failed: {e}')
                else:
                    time.sleep(1.5 * (attempt + 1))
        if r is None:
            break

        results = r.json().get('results', [])
        if not results:
            break

        for obs in results:
            photos = obs.get('observation_photos') or []
            loc = obs.get('location')
            usr = obs.get('user', {}).get('id')
            if not (photos and loc and usr):
                continue

            user_count = observers.get(usr, 0)
            can_add = len(all_obs) < max_obs

            if not can_add:
                if user_count >= observer_limit:
                    continue
                over_user = next((u for u, c in observers.items() if c > observer_limit), None)
                if over_user is None:
                    continue
                for idx, rec in enumerate(all_obs):
                    if rec['user_id'] == over_user:
                        all_obs.rotate(-idx)
                        all_obs.popleft()
                        all_obs.rotate(idx)
                        observers[over_user] -= 1
                        can_add = True
                        break

            if not can_add:
                continue

            photo_url = photos[0]['photo']['url']
            for res in ('square', 'medium', 'small', 'large', 'thumb'):
                photo_url = photo_url.replace(f'/{res}.', '/original.')

            rec = {
                'observation_id': obs['id'],
                'iconic_taxon_id': obs['taxon']['iconic_taxon_id'],
                'taxon_id': taxon_id,
                'subspecies_id': obs['taxon']['id'],
                'user_id': usr,
                'research_grade': obs['quality_grade'],
                'date_of_observation': obs['observed_on'],
                'location': tuple(map(float, loc.split(','))),
                'photo_url': photo_url,
            }
            for col in FILTER_COLS:
                rec[col] = ''
            all_obs.append(rec)
            observers[usr] = observers.get(usr, 0) + 1

        print(f'    Page {page}/{total_pages}: {len(all_obs)} obs collected')
        time.sleep(1.1 + random.uniform(0, 0.4))

        if len(all_obs) >= max_obs and not any(c > observer_limit for c in observers.values()):
            break

    rows = list(all_obs)
    os.makedirs(metadata_dir, exist_ok=True)
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f'    Saved {len(rows)} observations for taxon {taxon_id}')
    return rows


def append_metadata_from_inat(taxon_id, metadata_dir, rows, additional, observer_limit=10):
    """
    Fetch up to `additional` new observations from iNat not already in rows.
    Extends rows in-place, saves to disk, and returns the count of new rows added.
    Returns 0 if all available observations have already been fetched.
    """
    existing_ids = {str(r['observation_id']) for r in rows}
    int_taxon_id = int(taxon_id)
    base_url = 'https://api.inaturalist.org/v1/observations'
    params = {
        'photos': 'true', 'per_page': '200', 'taxon_id': int_taxon_id,
        'taxon_is_active': 'true', 'quality_grade': 'research',
        'licensed': 'true', 'photo_licensed': 'true',
        'order': 'desc', 'order_by': 'random',
    }
    try:
        resp = requests.get(base_url, params={**params, 'page': 1}, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f'    iNat fetch failed for {int_taxon_id}: {e}')
        return 0

    total_results = resp.json().get('total_results', 0)
    if len(existing_ids) >= total_results:
        print(f'    taxon {int_taxon_id}: all {total_results} observations already fetched')
        return 0

    total_pages = min(ceil(total_results / 200), 50)
    print(f'    taxon {int_taxon_id}: appending more obs '
          f'(have {len(existing_ids)}/{total_results}, want {additional} new)…')

    # Count per-observer usage already in rows (user_id may be string from CSV)
    observers = {}
    for r in rows:
        try:
            usr = int(r['user_id'])
            observers[usr] = observers.get(usr, 0) + 1
        except (ValueError, TypeError, KeyError):
            pass

    new_rows = []
    for page in range(1, total_pages + 1):
        if len(new_rows) >= additional:
            break
        r = None
        for attempt in range(3):
            try:
                r = requests.get(base_url, params={**params, 'page': page}, timeout=100)
                r.raise_for_status()
                break
            except Exception as e:
                r = None
                if attempt == 2:
                    print(f'    Page {page} failed: {e}')
                else:
                    time.sleep(1.5 * (attempt + 1))
        if r is None:
            break

        results = r.json().get('results', [])
        if not results:
            break

        for obs in results:
            if len(new_rows) >= additional:
                break
            obs_id = str(obs['id'])
            if obs_id in existing_ids:
                continue
            photos = obs.get('observation_photos') or []
            loc = obs.get('location')
            usr = obs.get('user', {}).get('id')
            if not (photos and loc and usr):
                continue
            if observers.get(usr, 0) >= observer_limit:
                continue
            photo_url = photos[0]['photo']['url']
            for res in ('square', 'medium', 'small', 'large', 'thumb'):
                photo_url = photo_url.replace(f'/{res}.', '/original.')
            rec = {
                'observation_id': obs['id'],
                'iconic_taxon_id': obs['taxon']['iconic_taxon_id'],
                'taxon_id': int_taxon_id,
                'subspecies_id': obs['taxon']['id'],
                'user_id': usr,
                'research_grade': obs['quality_grade'],
                'date_of_observation': obs['observed_on'],
                'location': tuple(map(float, loc.split(','))),
                'photo_url': photo_url,
            }
            for col in FILTER_COLS:
                rec[col] = ''
            new_rows.append(rec)
            existing_ids.add(obs_id)
            observers[usr] = observers.get(usr, 0) + 1

        time.sleep(1.1 + random.uniform(0, 0.4))

    if new_rows:
        rows.extend(new_rows)
        save_metadata(rows, taxon_id, metadata_dir)
        print(f'    Appended {len(new_rows)} new observations for taxon {int_taxon_id}')
    else:
        print(f'    taxon {int_taxon_id}: no new observations found in {total_pages} pages')
    return len(new_rows)


def ensure_metadata(taxon_id, metadata_dir, max_obs, existing_meta_dir):
    """
    Ensure metadata CSV exists. Copies from existing subspecies data first,
    then fetches from iNat API. Returns loaded rows.
    If the file already exists (even with fewer rows than max_obs), returns
    it as-is — never overwrites a file that may already have MD/VLM columns.
    """
    rows = load_metadata(taxon_id, metadata_dir)
    if rows:
        return rows

    ex_path = os.path.join(existing_meta_dir, f'metadata_{taxon_id}_updated.csv')
    if os.path.exists(ex_path):
        os.makedirs(metadata_dir, exist_ok=True)
        shutil.copy2(ex_path, meta_path(taxon_id, metadata_dir))
        rows = load_metadata(taxon_id, metadata_dir)
        if rows:
            return rows

    print(f'    Fetching metadata for {taxon_id} (requesting up to {max_obs} obs)…')
    return fetch_metadata_from_inat(taxon_id, metadata_dir, int(max_obs))


# =============================================================================
# Existing species reuse (symlink photos, copy metadata)
# =============================================================================

def setup_existing_species(taxon_id, existing_meta_dir, existing_photo_dir,
                            metadata_dir, photo_base):
    """Copy metadata and symlink photos from the subspecies pipeline if available."""
    if existing_meta_dir is None or existing_photo_dir is None:
        return False
    ex_meta   = os.path.join(existing_meta_dir, f'metadata_{taxon_id}_updated.csv')
    ex_photos = os.path.join(existing_photo_dir, str(taxon_id))
    if not os.path.exists(ex_meta) or not os.path.isdir(ex_photos):
        return False

    os.makedirs(metadata_dir, exist_ok=True)
    tgt_meta = meta_path(taxon_id, metadata_dir)
    if not os.path.exists(tgt_meta):
        shutil.copy2(ex_meta, tgt_meta)

    os.makedirs(photo_base, exist_ok=True)
    tgt_photos = os.path.join(photo_base, str(taxon_id))
    if not os.path.exists(tgt_photos):
        os.symlink(os.path.abspath(ex_photos), tgt_photos)

    return True


# =============================================================================
# Core per-species iteration
# =============================================================================

def download_and_detect_batch(batch_rows, taxon_id, photo_base, crops_base, args):
    """
    Download photos for batch_rows (if not already on disk), run MegaDetector,
    save crops, and update each row in-place with md_* columns.
    Returns n_new_md_passing.
    """
    pdir = os.path.join(photo_base, str(taxon_id))
    cdir = os.path.join(crops_base, str(taxon_id))
    os.makedirs(pdir, exist_ok=True)
    os.makedirs(cdir, exist_ok=True)

    to_detect = []
    for row in batch_rows:
        obs_id = row.get('observation_id', '')
        dest   = os.path.join(pdir, f'{obs_id}.jpg')
        if os.path.exists(dest):
            to_detect.append((row, dest))
        elif row.get('photo_url', ''):
            ok = download_photo(row['photo_url'], dest)
            time.sleep(1.1 + random.uniform(0, 0.4))
            if ok:
                to_detect.append((row, dest))
            else:
                row['md_pass'] = 'False'
                row['md_reject_reason'] = 'download_failed'

    if not to_detect:
        return 0

    md_results = run_md_on_paths(
        [p for _, p in to_detect],
        conf_thresh=args.conf_threshold,
        min_bbox_px=args.min_bbox_px,
        target_size=args.target_size,
        bbox_padding=args.bbox_padding,
        crops_dir=cdir,
    )

    n_passing = 0
    for row, dest in to_detect:
        r = md_results.get(dest, {})
        row['md_pass']         = str(r.get('md_pass', False))
        row['md_conf']         = str(r.get('md_conf', ''))
        row['md_bbox_px']      = str(r.get('md_bbox_px', ''))
        row['md_reject_reason'] = r.get('md_reject_reason', '')
        row['md_crop_path']    = r.get('md_crop_path', '')
        if r.get('md_pass'):
            n_passing += 1

    return n_passing


def fill_md_queue(rows, taxon_id, photo_base, crops_base, args, queue_target):
    """
    Download + detect unprocessed rows until we have `queue_target` new
    MD-passing-unVLM'd rows, or we run out of unprocessed rows.
    Modifies rows in-place. Returns number of new MD-passing rows added.
    """
    pending = unprocessed_rows(rows)
    n_added = 0
    while len(md_passing_unvlmd(rows)) < queue_target and pending:
        batch  = pending[:args.batch_size]
        pending = pending[args.batch_size:]
        n_added += download_and_detect_batch(batch, taxon_id, photo_base, crops_base, args)
    return n_added


def run_vlm_step(rows, args):
    """Run VLM on all MD-passing unVLM'd rows. Updates rows in-place."""
    to_vlm = md_passing_unvlmd(rows)
    if not to_vlm:
        return
    crop_paths = [r['md_crop_path'] for r in to_vlm]
    print(f'      VLM: sending {len(crop_paths)} crops…')
    results = run_vlm_batch(crop_paths, args)
    for row, res in zip(to_vlm, results):
        row['vlm_pass']          = str(res.get('vlm_pass', False))
        row['vlm_pose']          = res.get('vlm_pose', '')
        row['vlm_reject_reason'] = res.get('vlm_reject_reason', '')


# =============================================================================
# Split selection helper
# =============================================================================

def get_top2_generated(split):
    """Return (taxon_id, taxon_id) for the 2 generated species with most observations."""
    gen = split.get('generated_species', [])
    gen_sorted = sorted(gen, key=lambda s: int(s.get('observations', 0)), reverse=True)
    return [int(s['taxon_id']) for s in gen_sorted[:2]]


# =============================================================================
# Per-split orchestration
# =============================================================================

def process_split(split_id, split, controls, args):
    gen_tids = get_top2_generated(split)
    if len(gen_tids) < 2:
        print(f'  [skip] fewer than 2 generated species')
        return

    ctrl_info = controls.get(str(split_id)) or controls.get(split_id)
    ctrl_tid  = int(ctrl_info['taxon_id']) if ctrl_info and ctrl_info.get('taxon_id') else None

    print(f'  generated={gen_tids}  control={ctrl_tid}')

    # ── Resume check ────────────────────────────────────────────────────────
    gen_rows = {tid: load_metadata(tid, args.metadata_dir) for tid in gen_tids}
    existing_pose = compute_majority_pose(list(gen_rows.values()))

    if existing_pose:
        final_count_check = min(count_dataset_passing(r, existing_pose)
                                for r in gen_rows.values())
        # If previously abandoned (below MIN_VIABLE and metadata exhausted), skip
        if final_count_check < MIN_VIABLE:
            exhausted = all(len(unprocessed_rows(r)) == 0 for r in gen_rows.values())
            if exhausted:
                print(f'  [skip] previously abandoned '
                      f'(final_count={final_count_check} < MIN_VIABLE={MIN_VIABLE})')
                return

        gen_done = all(count_dataset_passing(r, existing_pose) >= args.target_cap
                       for r in gen_rows.values())
        ctrl_rows_check = load_metadata(ctrl_tid, args.metadata_dir) if ctrl_tid else []
        if ctrl_tid:
            ctrl_target = sum(
                min(count_dataset_passing(r, existing_pose), args.target_cap)
                for r in gen_rows.values()
            )
            ctrl_done = sum(1 for r in ctrl_rows_check if r.get('dataset_pass') == 'True') >= ctrl_target
        else:
            ctrl_done = True
        if gen_done and ctrl_done:
            print(f'  [skip] already complete (pose={existing_pose})')
            return

    # ── Phase A: Metadata + initial MD+VLM for generated species ────────────
    for tid in gen_tids:
        print(f'\n    [gen] taxon {tid}')
        setup_existing_species(tid, args.existing_metadata_dir, args.existing_photo_dir,
                               args.metadata_dir, args.photo_dir)
        rows = ensure_metadata(tid, args.metadata_dir,
                               int(args.target_cap * args.fetch_multiplier),
                               args.existing_metadata_dir)
        fill_md_queue(rows, tid, args.photo_dir, args.crops_dir, args, args.vlm_batch_size)
        save_metadata(rows, tid, args.metadata_dir)
        run_vlm_step(rows, args)
        save_metadata(rows, tid, args.metadata_dir)
        gen_rows[tid] = rows

    # ── Phase B: Majority pose ───────────────────────────────────────────────
    majority_pose = compute_majority_pose(list(gen_rows.values()))
    if not majority_pose:
        print('  [warn] No VLM results — skipping split')
        return
    print(f'\n    Majority pose: {majority_pose}')

    for tid, rows in gen_rows.items():
        mark_dataset_pass(rows, majority_pose)
        save_metadata(rows, tid, args.metadata_dir)

    # ── Phase C: Continue until target_cap for each generated species ────────
    for tid in gen_tids:
        rows = gen_rows[tid]
        while count_dataset_passing(rows, majority_pose) < args.target_cap:
            if not unprocessed_rows(rows):
                n_new = append_metadata_from_inat(
                    tid, args.metadata_dir, rows,
                    int(args.target_cap * args.fetch_multiplier))
                if n_new == 0:
                    n = count_dataset_passing(rows, majority_pose)
                    print(f'      taxon {tid}: all observations exhausted at '
                          f'{n}/{args.target_cap} (pose={majority_pose})')
                    break
            fill_md_queue(rows, tid, args.photo_dir, args.crops_dir, args, args.vlm_batch_size)
            save_metadata(rows, tid, args.metadata_dir)
            run_vlm_step(rows, args)
            mark_dataset_pass(rows, majority_pose)
            save_metadata(rows, tid, args.metadata_dir)
        n = count_dataset_passing(rows, majority_pose)
        # Hard-cap: if overshoot past target_cap, trim the excess back to False
        if n > args.target_cap:
            accepted_indices = [i for i, r in enumerate(rows) if r.get('dataset_pass') == 'True']
            for i in accepted_indices[args.target_cap:]:
                rows[i]['dataset_pass'] = 'False'
            save_metadata(rows, tid, args.metadata_dir)
            n = args.target_cap
        print(f'    taxon {tid}: {n} accepted (pose={majority_pose})')

    # ── Phase D: Final count (sum of capped gen counts → control target) ─────
    final_count = sum(
        min(count_dataset_passing(gen_rows[tid], majority_pose), args.target_cap)
        for tid in gen_tids
    )
    print(f'\n    Final count for split {split_id}: {final_count}  (pose={majority_pose})')

    if final_count < MIN_VIABLE:
        print(f'  [abandon] final_count={final_count} < MIN_VIABLE={MIN_VIABLE} '
              f'— not enough data for this split, skipping control')
        return

    if ctrl_tid is None:
        return

    # ── Phase E: Control species (own pose, independent of generated species) ──
    print(f'\n    [ctrl] taxon {ctrl_tid}  (target={final_count})')
    setup_existing_species(ctrl_tid, args.existing_metadata_dir, args.existing_photo_dir,
                           args.metadata_dir, args.photo_dir)
    ctrl_rows = ensure_metadata(ctrl_tid, args.metadata_dir,
                                int(final_count * args.fetch_multiplier),
                                args.existing_metadata_dir)

    # Initial MD+VLM batch to establish ctrl_pose
    fill_md_queue(ctrl_rows, ctrl_tid, args.photo_dir, args.crops_dir, args, args.vlm_batch_size)
    save_metadata(ctrl_rows, ctrl_tid, args.metadata_dir)
    run_vlm_step(ctrl_rows, args)
    save_metadata(ctrl_rows, ctrl_tid, args.metadata_dir)

    # Control picks its own majority pose independently of generated species
    ctrl_pose = compute_majority_pose([ctrl_rows]) or 'other'
    mark_dataset_pass(ctrl_rows, ctrl_pose)
    save_metadata(ctrl_rows, ctrl_tid, args.metadata_dir)

    # Continue until final_count accepted in ctrl_pose, fetching more from iNat if needed
    while count_dataset_passing(ctrl_rows, ctrl_pose) < final_count:
        if not unprocessed_rows(ctrl_rows):
            n_new = append_metadata_from_inat(
                ctrl_tid, args.metadata_dir, ctrl_rows,
                int(final_count * args.fetch_multiplier))
            if n_new == 0:
                break
        fill_md_queue(ctrl_rows, ctrl_tid, args.photo_dir, args.crops_dir, args, args.vlm_batch_size)
        save_metadata(ctrl_rows, ctrl_tid, args.metadata_dir)
        run_vlm_step(ctrl_rows, args)
        mark_dataset_pass(ctrl_rows, ctrl_pose)
        save_metadata(ctrl_rows, ctrl_tid, args.metadata_dir)

    n_ctrl = count_dataset_passing(ctrl_rows, ctrl_pose)

    # Hard-cap at final_count
    if n_ctrl > final_count:
        accepted_indices = [i for i, r in enumerate(ctrl_rows) if r.get('dataset_pass') == 'True']
        for i in accepted_indices[final_count:]:
            ctrl_rows[i]['dataset_pass'] = 'False'
        save_metadata(ctrl_rows, ctrl_tid, args.metadata_dir)
        n_ctrl = final_count

    # Fall back to any VLM-passing pose if too few in majority pose
    if n_ctrl < MIN_VIABLE:
        n_any = sum(1 for r in ctrl_rows if r.get('vlm_pass') in ('True', True))
        if n_any >= MIN_VIABLE:
            print(f'    [fallback] ctrl only {n_ctrl} in pose={ctrl_pose}; '
                  f'accepting all {n_any} VLM-passing poses')
            for r in ctrl_rows:
                if r.get('vlm_pass') in ('True', True):
                    r['dataset_pass'] = 'True'
            save_metadata(ctrl_rows, ctrl_tid, args.metadata_dir)
            n_ctrl = n_any

    print(f'    ctrl taxon {ctrl_tid}: {n_ctrl}/{final_count} accepted  (pose={ctrl_pose})')


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Iterative download + MegaDetector + local VLM filter for Aves splits.')
    parser.add_argument('--aves_splits',   default='aves_splits.json')
    parser.add_argument('--aves_controls', default='aves_controls.json')
    parser.add_argument('--metadata_dir',  default='aves_metadata')
    parser.add_argument('--photo_dir',     default='aves_photos')
    parser.add_argument('--crops_dir',     default='aves_photos_cropped')
    # Optional: reuse metadata/photos already downloaded by an earlier pipeline run.
    parser.add_argument('--existing_metadata_dir', default=None)
    parser.add_argument('--existing_photo_dir', default=None)
    # Filter thresholds — defaults match test_megadetector.py
    parser.add_argument('--conf_threshold', type=float, default=MD_CONF_THRESHOLD)
    parser.add_argument('--min_bbox_px',    type=int,   default=MD_MIN_BBOX_PX)
    parser.add_argument('--target_size',    type=int,   default=MD_TARGET_SIZE)
    parser.add_argument('--bbox_padding',   type=float, default=MD_BBOX_PADDING)
    # Download / VLM sizing
    parser.add_argument('--target_cap',      type=int,   default=100,
                        help='Target accepted images per species after VLM+pose filter')
    parser.add_argument('--fetch_multiplier', type=float, default=4.0,
                        help='Metadata rows to fetch = target_cap × fetch_multiplier')
    parser.add_argument('--batch_size',      type=int,   default=20,
                        help='Photos per MegaDetector batch')
    parser.add_argument('--vlm_batch_size',  type=int,   default=10,
                        help='MD-passing photos to accumulate before calling VLM')
    # Local VLM settings
    parser.add_argument('--model_id',     default=DEFAULT_MODEL_ID,
                        help='HuggingFace model ID (e.g. Qwen/Qwen2.5-VL-72B-Instruct-AWQ)')
    parser.add_argument('--device',       default=DEFAULT_DEVICE,
                        help='Torch device for VLM inference (e.g. cuda:2, cuda:0, cpu)')
    parser.add_argument('--hf_cache_dir', default=DEFAULT_HF_CACHE,
                        help='Directory to cache downloaded HuggingFace model weights')
    # Selective processing
    parser.add_argument('--split_ids', nargs='*', default=None,
                        help='Process only these split IDs (default: all)')
    args = parser.parse_args()

    with open(args.aves_splits) as f:
        aves_splits = json.load(f)
    with open(args.aves_controls) as f:
        aves_controls = json.load(f)

    os.makedirs(args.metadata_dir, exist_ok=True)
    os.makedirs(args.photo_dir,    exist_ok=True)
    os.makedirs(args.crops_dir,    exist_ok=True)

    split_ids = args.split_ids or list(aves_splits.keys())
    print(f'Processing {len(split_ids)} splits  (target_cap={args.target_cap}, '
          f'fetch_multiplier={args.fetch_multiplier}, model={args.model_id})')

    for i, split_id in enumerate(split_ids, 1):
        split = aves_splits.get(str(split_id)) or aves_splits.get(split_id)
        if split is None:
            print(f'[{i}/{len(split_ids)}] split {split_id}: not found, skipping')
            continue
        print(f'\n[{i}/{len(split_ids)}] split {split_id}')
        try:
            process_split(str(split_id), split, aves_controls, args)
        except Exception:
            print(f'  ERROR in split {split_id}:')
            traceback.print_exc()

    print('\nDone.')


if __name__ == '__main__':
    main()
