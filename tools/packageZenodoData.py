"""
packageZenodoData.py

Generates a manifest and a self-contained server-side packaging script.

Sources (all local, no network access):
  - ../simplexAnimalEphy.config       : dates, rec types, trial numbers
  - zenodo_recordings_cache.csv       : maps dates -> session folder names

Outputs:
  - zenodo_manifest.txt               : human-readable list of included trial folders
  - zenodo_package.py                 : self-contained script to run on the server
                                        (converts video to MJPG, creates per-animal tar.gz)

Usage:
  python packageZenodoData.py [--dest /path/to/output] [--dry-run]
"""

import yaml
import csv
import os
import argparse

TARGET_ANIMALS = [
    '220211_f38', '220214_f43', '220205_f57', '220205_f61',
    '220507_m81', '220507_m90', '220525_m19', '220525_m27',
    '220525_m28', '220716_f65', '220716_f67',
]

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'simplexAnimalEphy.config')
CSV_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'zenodo_recordings_cache.csv')
BASE_DATA_PATH = '/media/invivodata2/altair_data/dataMichael'
TRIAL_PREFIX = 'locomotionEphys2Motor60sec'


def parse_trials(trials_str):
    before_comment = str(trials_str).split('#')[0]
    return [int(p.strip()) for p in before_comment.split(',')
            if p.strip().isdigit()]


def load_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    animal_data = {}
    for entry in config.values():
        mouse = entry['mouse']
        if mouse not in TARGET_ANIMALS:
            continue
        animal_data[mouse] = {}
        for date_int, day_data in entry['days'].items():
            date_str = str(date_int)
            all_trials = set()
            found_ephys = False
            for rec_type in ('recsMLI', 'recsPC'):
                if rec_type not in day_data:
                    continue
                found_ephys = True
                for rec_data in day_data[rec_type].values():
                    if isinstance(rec_data, dict):
                        trials_raw = rec_data['trials']
                    else:
                        trials_raw = str(rec_data).split('trials:')[-1]
                    all_trials.update(parse_trials(trials_raw))
            if found_ephys:
                animal_data[mouse][date_str] = sorted(all_trials)
    return animal_data


def load_folder_map(csv_path):
    with open(csv_path) as f:
        rows = list(csv.reader(f))
    all_ids = {r[0].strip() for r in rows if r and r[0].strip()}
    animal_starts = {}
    for i, row in enumerate(rows):
        if row and row[0].strip() in TARGET_ANIMALS:
            animal_starts[row[0].strip()] = i
    folder_map = {}
    for animal in TARGET_ANIMALS:
        if animal not in animal_starts:
            continue
        folder_map[animal] = {}
        start = animal_starts[animal]
        for i in range(start, min(start + 80, len(rows))):
            row = rows[i]
            if i > start and row and row[0].strip() and row[0].strip() in all_ids \
                    and row[0].strip() != animal:
                break
            date = row[5].strip() if len(row) > 5 else ''
            folder = row[11].strip() if len(row) > 11 else ''
            if date and folder:
                folder_map[animal][date] = folder
    return folder_map


def build_manifest(animal_data, folder_map):
    manifest = {a: [] for a in TARGET_ANIMALS}
    warnings = []
    for animal in TARGET_ANIMALS:
        if animal not in animal_data:
            warnings.append(f'WARNING: {animal} not found in config')
            continue
        for date_str, trials in sorted(animal_data[animal].items()):
            folder = folder_map.get(animal, {}).get(date_str)
            if not folder:
                warnings.append(f'WARNING: no session folder for {animal} on {date_str}')
                continue
            for trial in trials:
                trial_folder = f'{TRIAL_PREFIX}_{trial:03d}'
                src = os.path.join(BASE_DATA_PATH, folder, trial_folder)
                manifest[animal].append((folder, trial_folder, src))
    return manifest, warnings


def check_paths(manifest):
    return [src for entries in manifest.values()
            for _, _, src in entries if not os.path.isdir(src)]


def write_manifest_file(manifest, warnings, out_path):
    with open(out_path, 'w') as f:
        f.write('# Zenodo raw data manifest\n')
        f.write(f'# Base path: {BASE_DATA_PATH}\n\n')
        total = sum(len(v) for v in manifest.values())
        f.write(f'# Total trial folders: {total}\n\n')
        for animal in TARGET_ANIMALS:
            entries = manifest[animal]
            if not entries:
                continue
            f.write(f'\n## {animal}  ({len(entries)} trial folders)\n')
            for folder, trial_folder, _ in entries:
                f.write(f'  {folder}/{trial_folder}\n')
        if warnings:
            f.write('\n## WARNINGS\n')
            for w in warnings:
                f.write(f'  {w}\n')
    print(f'Manifest written to {out_path}')


def write_package_script(manifest, dest, out_path):
    # Serialise manifest as a Python literal
    manifest_lines = ['MANIFEST = {']
    for animal in TARGET_ANIMALS:
        entries = manifest[animal]
        if not entries:
            continue
        manifest_lines.append(f"    '{animal}': [")
        for folder, trial_folder, _ in entries:
            manifest_lines.append(f"        ('{folder}', '{trial_folder}'),")
        manifest_lines.append('    ],')
    manifest_lines.append('}')
    manifest_literal = '\n'.join(manifest_lines)

    script = f'''#!/usr/bin/env python3
"""
zenodo_package.py  --  auto-generated by packageZenodoData.py
Converts video_000.ma -> video_000.avi (MJPG) and packages data
into one tar.gz per animal for Zenodo upload.

Requirements: h5py, opencv-python, numpy
Usage:
  python zenodo_package.py [--dest /path/to/output] [--quality 85] [--animal 220211_f38]
"""

import h5py
import cv2
import numpy as np
import os
import shutil
import tarfile
import argparse
import sys
from datetime import datetime

BASE_DATA    = '{BASE_DATA_PATH}'
DEFAULT_DEST = '{dest}'
VIDEO_FILE   = 'video_000.ma'
VIDEO_OUT    = 'video_000.avi'

{manifest_literal}


def log(msg):
    print(f'[{{datetime.now().strftime("%H:%M:%S")}}] {{msg}}', flush=True)


def convert_video(src_ma, dst_avi, quality):
    """Convert an ACQ4 video .ma (HDF5) to MJPG .avi frame-by-frame."""
    with h5py.File(src_ma, 'r') as f:
        frames = f['data']
        times  = f['info/0/values'][()]
        n      = frames.shape[0]
        h, w   = frames.shape[1], frames.shape[2]
        is_color = (frames.ndim == 4)
        fps    = n / (times[-1] - times[0]) if times[-1] > times[0] else 25.0

        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        writer = cv2.VideoWriter(dst_avi, fourcc, fps, (w, h))
        try:
            writer.set(cv2.VIDEOWRITER_PROP_QUALITY, quality)
        except AttributeError:
            pass  # older OpenCV versions

        for i in range(n):
            frame = frames[i]
            if frame.dtype != np.uint8:
                frame = (frame >> 8).astype(np.uint8)
            if not is_color:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            writer.write(frame)
        writer.release()


def process_trial(src_trial, dst_trial, quality):
    """Copy all files from src trial folder; convert video .ma to MJPG."""
    os.makedirs(dst_trial, exist_ok=True)
    for item in os.listdir(src_trial):
        src_item = os.path.join(src_trial, item)
        dst_item = os.path.join(dst_trial, item)
        if os.path.isdir(src_item):
            # Camera subfolder (CameraGigEBehavior)
            os.makedirs(dst_item, exist_ok=True)
            for fname in os.listdir(src_item):
                src_f = os.path.join(src_item, fname)
                if fname == VIDEO_FILE:
                    dst_f = os.path.join(dst_item, VIDEO_OUT)
                    log(f'    converting {{fname}} -> {{VIDEO_OUT}}')
                    convert_video(src_f, dst_f, quality)
                else:
                    shutil.copy2(src_f, os.path.join(dst_item, fname))
        else:
            shutil.copy2(src_item, dst_item)


def package_animal(animal, entries, dest, quality):
    archive = os.path.join(dest, f'{{animal}}.tar.gz')
    if os.path.exists(archive):
        log(f'SKIP {{animal}} ({{archive}} already exists)')
        return

    staging = os.path.join(dest, '_staging', animal)
    os.makedirs(staging, exist_ok=True)
    log(f'=== {{animal}}: {{len(entries)}} trial folders ===')

    try:
        for idx, (session_folder, trial_folder) in enumerate(entries, 1):
            src_trial = os.path.join(BASE_DATA, session_folder, trial_folder)
            dst_trial = os.path.join(staging, session_folder, trial_folder)
            log(f'  [{{idx}}/{{len(entries)}}] {{session_folder}}/{{trial_folder}}')
            process_trial(src_trial, dst_trial, quality)

        log(f'  Creating {{os.path.basename(archive)}} ...')
        with tarfile.open(archive, 'w:gz') as tar:
            tar.add(staging, arcname=animal)
        log(f'  Done: {{archive}}')
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dest',    default=DEFAULT_DEST,
                        help='Output directory for tar.gz archives')
    parser.add_argument('--quality', type=int, default=85,
                        help='MJPG quality 0-100 (default 85)')
    parser.add_argument('--animal',  default=None,
                        help='Process only this animal (e.g. 220211_f38)')
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)

    animals = [args.animal] if args.animal else list(MANIFEST.keys())
    for animal in animals:
        if animal not in MANIFEST:
            print(f'Unknown animal: {{animal}}')
            sys.exit(1)
        package_animal(animal, MANIFEST[animal], args.dest, args.quality)

    # Copy README alongside the archives (only when running full packaging)
    if not args.animal:
        readme_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'README_zenodo.md')
        readme_dst = os.path.join(args.dest, 'README.md')
        if os.path.exists(readme_src):
            shutil.copy2(readme_src, readme_dst)
            log(f'README copied to {{readme_dst}}')

    log('All done.')


if __name__ == '__main__':
    main()
'''

    with open(out_path, 'w') as f:
        f.write(script)
    os.chmod(out_path, 0o755)
    print(f'Packaging script written to {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Generate Zenodo packaging script.')
    parser.add_argument('--dest', default='/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/packagedData',
                        help='Destination directory for tar.gz archives (written into the script)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Check paths and write manifest only')
    args = parser.parse_args()

    print(f'Loading config from {CONFIG_PATH}')
    animal_data = load_config(CONFIG_PATH)

    print(f'Loading folder map from {CSV_CACHE_PATH}')
    folder_map = load_folder_map(CSV_CACHE_PATH)

    manifest, warnings = build_manifest(animal_data, folder_map)
    for w in warnings:
        print(w)

    total = sum(len(v) for v in manifest.values())
    print(f'\nManifest: {total} trial folders across {len(TARGET_ANIMALS)} animals')

    missing = check_paths(manifest)
    if missing:
        print(f'\n{len(missing)} source paths NOT FOUND:')
        for p in missing:
            print(f'  MISSING: {p}')
    else:
        print('All source paths verified on server.')

    out_dir = os.path.dirname(os.path.abspath(__file__))
    write_manifest_file(manifest, warnings,
                        os.path.join(out_dir, 'zenodo_manifest.txt'))

    if not args.dry_run:
        write_package_script(manifest, args.dest,
                             os.path.join(out_dir, 'zenodo_package.py'))
    else:
        print('Dry-run mode: packaging script not written.')


if __name__ == '__main__':
    main()
