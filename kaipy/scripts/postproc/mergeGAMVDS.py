"""
mergeGAMVDS.py

Utility to build an HDF5 Virtual Dataset (VDS) from output files in subdirectories.
- Step# groups: each appears as a separate group in the VDS.
- timeAttributeCache: variables are concatenated in step order, VDS used.
- Other variables: copied from the first file only.

Usage (example):
    python mergeGAMVDS.py --input-dir <input_dir> --output <output_file.h5>

"""
import os
import re
import argparse
import h5py
from glob import glob
from collections import defaultdict

STEP_GROUP_PATTERN = re.compile(r"^Step#(\d+)$")
def create_command_line_parser():
    parser = argparse.ArgumentParser(description="Build HDF5 VDS files from output files in subdirectories.")
    parser.add_argument('--input-dir', required=True, help='Input directory with HDF5 files')
    parser.add_argument('--output-dir', required=True, help='Output directory for VDS HDF5 files')
    return parser

def find_hdf5_files(input_dir):
    """Recursively find all HDF5 files in input_dir."""
    h5_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.endswith('.h5') or f.endswith('.hdf5'):
                h5_files.append(os.path.join(root, f))
    return sorted(h5_files)

def index_step_groups(h5_files):
    """Map step number to (file, group name)."""
    step_map = dict()
    for f in h5_files:
        with h5py.File(f, 'r') as h5:
            for g in h5:
                m = STEP_GROUP_PATTERN.match(g)
                if m:
                    step = int(m.group(1))
                    if step in step_map:
                        raise ValueError(f"Duplicate Step#{step} in {f} and {step_map[step][0]}")
                    step_map[step] = (f, g)
    if not step_map:
        raise ValueError("No Step# groups found in input files.")
    return dict(sorted(step_map.items()))

def collect_time_attribute_cache(h5_files, step_map):
    """Collect timeAttributeCache datasets and their shapes/dtypes in step order."""
    cache_data = defaultdict(list)
    cache_attrs = dict()
    for step, (f, _) in step_map.items():
        with h5py.File(f, 'r') as h5:
            if 'timeAttributeCache' in h5:
                group = h5['timeAttributeCache']
                for dset in group:
                    data = group[dset]
                    cache_data[dset].append({'file': f, 'shape': data.shape, 'dtype': data.dtype, 'name': dset})
                    if dset not in cache_attrs:
                        cache_attrs[dset] = dict(data.attrs)
    return cache_data, cache_attrs

def copy_other_variables(first_file, out_h5):
    """Copy all root-level datasets/groups except Step#, timeAttributeCache from first_file to out_h5."""
    with h5py.File(first_file, 'r') as h5:
        for k in h5:
            if STEP_GROUP_PATTERN.match(k) or k == 'timeAttributeCache':
                continue
            h5.copy(k, out_h5)
        # Copy root attributes
        for k, v in h5.attrs.items():
            out_h5.attrs[k] = v

def create_vds_for_steps(step_map, out_h5):
    """Create VDS groups for each Step# in step_map."""
    for step, (f, gname) in step_map.items():
        vds_group = out_h5.create_group(f'Step#{step}')
        with h5py.File(f, 'r') as h5:
            src_group = h5[gname]
            for dset in src_group:
                src = h5py.VirtualSource(f, f'{gname}/{dset}', shape=src_group[dset].shape)
                layout = h5py.VirtualLayout(shape=src_group[dset].shape, dtype=src_group[dset].dtype)
                layout[...] = src
                vds_group.create_virtual_dataset(dset, layout, fillvalue=None)
                # Copy attributes
                for k, v in src_group[dset].attrs.items():
                    vds_group[dset].attrs[k] = v
            # Copy group attributes
            for k, v in src_group.attrs.items():
                vds_group.attrs[k] = v

def create_vds_for_time_attribute_cache(cache_data, cache_attrs, out_h5):
    """Create VDS for each variable in timeAttributeCache, concatenated in step order."""
    if not cache_data:
        return
    # Sort files by minimum step number for correct order
    def min_step_in_file(f):
        with h5py.File(f, 'r') as h5:
            steps = [int(g.split('#')[1]) for g in h5 if g.startswith('Step#')]
            return min(steps) if steps else float('inf')
    # Get all unique files from cache_data
    all_files = set()
    for parts in cache_data.values():
        for p in parts:
            all_files.add(p['file'])
    sorted_files = sorted(all_files, key=min_step_in_file)
    # Collect all datasets and their slices in correct order
    timecache_datasets = {}
    timecache_attrs = {}
    for f in sorted_files:
        with h5py.File(f, 'r') as h5:
            if 'timeAttributeCache' in h5:
                for dset in h5['timeAttributeCache']:
                    arr = h5['timeAttributeCache'][dset]
                    if dset not in timecache_datasets:
                        timecache_datasets[dset] = []
                        timecache_attrs[dset] = dict(arr.attrs)
                    timecache_datasets[dset].append((f, f'timeAttributeCache/{dset}', arr.shape, arr.dtype))
    if timecache_datasets:
        if 'timeAttributeCache' not in out_h5:
            group = out_h5.create_group('timeAttributeCache')
        else:
            group = out_h5['timeAttributeCache']
        for dset, entries in timecache_datasets.items():
            total_len = sum(e[2][0] for e in entries)
            shape0 = entries[0][2]
            dtype = entries[0][3]
            out_shape = (total_len,) + shape0[1:]
            layout = h5py.VirtualLayout(shape=out_shape, dtype=dtype)
            offset = 0
            for f, dpath, shape, _ in entries:
                vsource = h5py.VirtualSource(f, dpath, shape)
                layout[offset:offset+shape[0]] = vsource
                offset += shape[0]
            d = group.create_virtual_dataset(dset, layout)
            for ak, av in timecache_attrs[dset].items():
                d.attrs[ak] = av


def main():
    parser = create_command_line_parser()
    args = parser.parse_args()

    h5_files = find_hdf5_files(args.input_dir)
    if not h5_files:
        raise RuntimeError("No HDF5 files found in input directory.")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Group input files by their basename (excluding directory)
    from collections import defaultdict
    files_by_name = defaultdict(list)
    for f in h5_files:
        files_by_name[os.path.basename(f)].append(f)

    for fname, files in files_by_name.items():
        # Build step map and timeAttributeCache info for this group of files
        step_map = index_step_groups(files)
        cache_data, cache_attrs = collect_time_attribute_cache(files, step_map)

        # Use the first file as the reference for copying other variables
        ref_file = files[0]
        out_file = os.path.join(args.output_dir, fname)
        with h5py.File(out_file, 'w', libver='latest') as out_h5:
            copy_other_variables(ref_file, out_h5)
            # Include all Step# groups from all files in this group
            create_vds_for_steps(step_map, out_h5)
            # Include all timeAttributeCache slices from all files in this group
            create_vds_for_time_attribute_cache(cache_data, cache_attrs, out_h5)
        print(f"VDS file created: {out_file}")

if __name__ == "__main__":
    main()
