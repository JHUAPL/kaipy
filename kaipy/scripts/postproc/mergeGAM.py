# Robust Gamera HDF5 Merger Utility (Template)

import os
import glob
import argparse
from collections import defaultdict, OrderedDict

#Third-party modules
import numpy as np
import h5py

def create_command_line_parser():
	import argparse
	parser = argparse.ArgumentParser(description="Merge Gamera HDF5 MPI segment files into merged rank files in an output directory.")
	parser.add_argument('-i', '--input', dest='root_dir', help='Root directory containing segment subdirectories')
	parser.add_argument('-o', '--output', dest='output_dir', help='Output directory for merged files')
	parser.add_argument('--basename', default='mage', help='Basename pattern for files to merge (default: mage)')
	return parser

def find_segment_files(root_dir, pattern='mage*.gam.h5'):
    """Recursively find all mage*.gam.h5 files in segment subdirectories."""
    segment_files = []
    for subdir, dirs, files in os.walk(root_dir):
        h5s = sorted(glob.glob(os.path.join(subdir, pattern)))
        print(f"Found {len(h5s)} files in {subdir}")
        print(h5s)
        for h5file in h5s:
            segment_files.append(h5file)
    return segment_files

def index_timesteps_and_ranks(files):
    """For each timestep, collect all files (ranks) that contribute to that step."""
    timestep_map = defaultdict(list)  # {step: [file1, file2, ...]}
    for f in files:
        with h5py.File(f, 'r') as h5:
            for g in h5:
                if g.startswith('Step#'):
                    step = int(g.split('#')[1])
                    timestep_map[step].append(f)
    all_steps = sorted(timestep_map.keys())
    if not all_steps:
        raise ValueError("No timesteps found in any input files.")
    expected = list(range(all_steps[0], all_steps[-1]+1))
    missing = set(expected) - set(all_steps)
    if missing:
        raise ValueError(f"Missing timesteps: {sorted(missing)}")
    return OrderedDict((step, timestep_map[step]) for step in expected)

def collect_metadata(reference_file):
	"""Extract global and dataset metadata from a reference file."""
	meta = {}
	with h5py.File(reference_file, 'r') as h5:
		meta['attrs'] = dict(h5.attrs)
		meta['structure'] = {}
		for k in h5:
			meta['structure'][k] = {}
			if hasattr(h5[k], 'attrs'):
				meta['structure'][k]['attrs'] = dict(h5[k].attrs)
			if isinstance(h5[k], h5py.Dataset):
				meta['structure'][k]['shape'] = h5[k].shape
				meta['structure'][k]['dtype'] = h5[k].dtype
	return meta

def create_output_file(output_path, metadata, shape_info):
	"""Create output HDF5 file, pre-allocating only Step# groups and copying root attributes."""
	with h5py.File(output_path, 'w') as h5:
		for k, v in metadata['attrs'].items():
			h5.attrs[k] = v
		for g, info in metadata['structure'].items():
			if g.startswith('Step#'):
				if 'shape' in info:
					h5.create_group(g)
			else:
				# Do not pre-create root-level datasets or groups; let h5py.copy handle them
				pass
			if 'attrs' in info and g in h5:
				for ak, av in info['attrs'].items():
					h5[g].attrs[ak] = av



def merge_all_timesteps_per_rank(rank_files, merged_path):
	"""
	For a given rank, merge all timesteps from all files into a single merged file.
	Each file contains unique timesteps for this rank.
	"""
	# Map: timestep -> file
	timestep_files = {}
	for f in rank_files:
		with h5py.File(f, 'r') as h5:
			for g in h5:
				if g.startswith('Step#'):
					step = int(g.split('#')[1])
					if step in timestep_files:
						raise ValueError(f"Duplicate timestep {step} for rank in {f} and {timestep_files[step]}")
					timestep_files[step] = f
	all_steps = sorted(timestep_files.keys())
	if not all_steps:
		raise ValueError("No timesteps found for this rank.")
	# Use first file as reference for metadata
	ref_file = rank_files[0]
	metadata = collect_metadata(ref_file)
	# Create output file and copy global attrs
	create_output_file(merged_path, metadata, {k: v['shape'] for k, v in metadata['structure'].items() if 'shape' in v})
	with h5py.File(merged_path, 'a') as h5out:
		for step in all_steps:
			src_file = timestep_files[step]
			gname = f'Step#{step}'
			print(f"Merging timestep {step} from {src_file}")
			with h5py.File(src_file, 'r') as h5in:
				if gname not in h5in:
					raise ValueError(f"Timestep {step} not found in {src_file}")
				# Copy the entire group (datasets and attributes)
				if gname in h5out:
					del h5out[gname]
				h5in.copy(gname, h5out)
		# Copy non-step datasets and groups from reference file, except timeAttributeCache
		with h5py.File(ref_file, 'r') as h5ref:
			for k in h5ref:
				if k == 'timeAttributeCache':
					continue
				if not k.startswith('Step#') and k not in h5out:
					h5ref.copy(k, h5out)
		# Special handling for timeAttributeCache: concatenate datasets across all files in file order (not step order)
		timecache_data = {}
		timecache_attrs = {}
		print(f"Collecting timeAttributeCache data from {len(rank_files)} files for rank")
		# Sort rank_files by the minimum timestep in each file to ensure correct order
		def min_step_in_file(f):
			with h5py.File(f, 'r') as h5:
				steps = [int(g.split('#')[1]) for g in h5 if g.startswith('Step#')]
				return min(steps) if steps else float('inf')
		sorted_rank_files = sorted(rank_files, key=min_step_in_file)
		for f in sorted_rank_files:
			with h5py.File(f, 'r') as h5in:
				if 'timeAttributeCache' in h5in:
					for dset in h5in['timeAttributeCache']:
						arr = h5in['timeAttributeCache'][dset][...]
						if dset not in timecache_data:
							timecache_data[dset] = []
							timecache_attrs[dset] = dict(h5in['timeAttributeCache'][dset].attrs)
						timecache_data[dset].append(arr)
		# Write concatenated datasets to merged file
		if timecache_data:
			if 'timeAttributeCache' not in h5out:
				h5out.create_group('timeAttributeCache')
			for dset, arrs in timecache_data.items():
				concat = arrs[0]
				if len(arrs) > 1:
					concat = np.concatenate(arrs, axis=0)
				d = h5out['timeAttributeCache'].create_dataset(dset, data=concat)
				# Copy attributes
				for ak, av in timecache_attrs[dset].items():
					d.attrs[ak] = av

def find_serial_files(root_dir, pattern):
	serial_files = []
	for subdir, dirs, files in os.walk(root_dir):
		h5s = sorted(glob.glob(os.path.join(subdir, pattern)))
		for h5file in h5s:
			serial_files.append(h5file)
	return serial_files

def merge_serial_files(files, merged_path):
	if not files:
		return
	# Map: timestep -> file
	timestep_files = {}
	for f in files:
		with h5py.File(f, 'r') as h5:
			for g in h5:
				if g.startswith('Step#'):
					step = int(g.split('#')[1])
					if step in timestep_files:
						raise ValueError(f"Duplicate timestep {step} in {f} and {timestep_files[step]}")
					timestep_files[step] = f
	all_steps = sorted(timestep_files.keys())
	if not all_steps:
		raise ValueError("No timesteps found for this file type.")
	ref_file = files[0]
	metadata = collect_metadata(ref_file)
	create_output_file(merged_path, metadata, {k: v['shape'] for k, v in metadata['structure'].items() if 'shape' in v})
	with h5py.File(merged_path, 'a') as h5out:
		for step in all_steps:
			src_file = timestep_files[step]
			gname = f'Step#{step}'
			print(f"Merging timestep {step} from {src_file}")
			with h5py.File(src_file, 'r') as h5in:
				if gname not in h5in:
					raise ValueError(f"Timestep {step} not found in {src_file}")
				if gname in h5out:
					del h5out[gname]
				h5in.copy(gname, h5out)
		# Copy non-step datasets/groups from reference file, except timeAttributeCache
		with h5py.File(ref_file, 'r') as h5ref:
			for k in h5ref:
				if k == 'timeAttributeCache':
					continue
				if not k.startswith('Step#') and k not in h5out:
					h5ref.copy(k, h5out)
		# Special handling for timeAttributeCache: concatenate datasets across all files in file order
		timecache_data = {}
		timecache_attrs = {}
		def min_step_in_file(f):
			with h5py.File(f, 'r') as h5:
				steps = [int(g.split('#')[1]) for g in h5 if g.startswith('Step#')]
				return min(steps) if steps else float('inf')
		sorted_files = sorted(files, key=min_step_in_file)
		for f in sorted_files:
			with h5py.File(f, 'r') as h5in:
				if 'timeAttributeCache' in h5in:
					for dset in h5in['timeAttributeCache']:
						arr = h5in['timeAttributeCache'][dset][...]
						if dset not in timecache_data:
							timecache_data[dset] = []
							timecache_attrs[dset] = dict(h5in['timeAttributeCache'][dset].attrs)
						timecache_data[dset].append(arr)
		if timecache_data:
			if 'timeAttributeCache' not in h5out:
				h5out.create_group('timeAttributeCache')
			for dset, arrs in timecache_data.items():
				concat = arrs[0]
				if len(arrs) > 1:
					concat = np.concatenate(arrs, axis=0)
				d = h5out['timeAttributeCache'].create_dataset(dset, data=concat)
				for ak, av in timecache_attrs[dset].items():
					d.attrs[ak] = av

def main():
	MainS = """Merge Gamera HDF5 MPI segment files into merged rank files in an output directory."""
	parser = create_command_line_parser()

	args = parser.parse_args()

	pattern = f'{args.basename}*.gam.h5'
	files = find_segment_files(args.root_dir, pattern=pattern)
	#print(f"Found {len(files)} files to merge.")
	#print(files)
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	# Group files by rank descriptor (everything after basename, before .gam.h5)
	from collections import defaultdict
	import re
	rank_groups = defaultdict(list)
	rank_re = re.compile(re.escape(args.basename) + r'(.*)\.gam\.h5$')
	for f in files:
		m = rank_re.search(os.path.basename(f))
		if m:
			rank_desc = m.group(1)
			rank_groups[rank_desc].append(f)
	if not rank_groups:
		raise ValueError("No rank files found matching pattern.")

	for rank_desc, rank_files in rank_groups.items():
		print(f"Merging rank: {rank_desc} ({len(rank_files)} files)")
		merged_basename = f"{args.basename}_merged{rank_desc}.gam.h5"
		merged_path = os.path.join(args.output_dir, merged_basename)
		merge_all_timesteps_per_rank(rank_files, merged_path)
		print(f"Merged timesteps into {merged_path}")

	# --- SERIAL FILE MERGE LOGIC ---
	serial_types = [
		('gamCpl', '*.gamCpl.h5'),
		('mhdrcm', '*.mhdrcm.h5'),
		('mix', '*.mix.h5'),
		('rcm', '*.rcm.h5'),
		('volt', '*.volt.h5'),
	]

	for typ, pattern in serial_types:
		serial_files = find_serial_files(args.root_dir, pattern)
		if serial_files:
			print(f"Merging {len(serial_files)} files for type {typ}")
			merged_basename = f"{args.basename}_merged.{typ}.h5"
			merged_path = os.path.join(args.output_dir, merged_basename)
			merge_serial_files(serial_files, merged_path)
			print(f"Merged serial files into {merged_path}")
#Uncomment to enable script execution
if __name__ == "__main__":
    main()
