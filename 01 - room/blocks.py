from glob import glob
import pandas as pd
import os, pickle
import numpy as np

from commons import *

def get_blocks_collections(input_folder='input_data'):
	def convertColor(code):
		return (0, ((code % 100) * 10) % 256, ((code // 100) * 10) % 256)

	# ordering
	def add_rotation_bed_orientation(blocks_all, cur_block, reflecting=False):
		if reflecting: cur_block = (cur_block[1], cur_block[0], cur_block[3], cur_block[2])
		blocks_all.append(cur_block)
		for _ in range(3):
			cur_block = (cur_block[1], cur_block[3], cur_block[0], cur_block[2])
			blocks_all.append(cur_block)

	def belongToDoor(cur_block):
		door_els = 0; wall_els = []
		for ind, item in enumerate(cur_block):
			if item in doors: door_els += 1
			elif item == wall: wall_els.append(ind)
		if door_els == 0: return False, None, door_els
		if door_els == 2 and len(wall_els) == 2:
			if wall_els[0] == 0:
				if wall_els[1] == 2: return True, 'left', door_els
				if wall_els[1] == 1: return True, 'top', door_els
			elif wall_els[1] == 3:
				if wall_els[0] == 1: return True, 'right', door_els
				if wall_els[0] == 2: return True, 'bottom', door_els
		return True, None, door_els

	def belongToBed(cur_block):
		bed_els = 0
		for item in cur_block:
			if item in beds: bed_els += 1
		if bed_els == 0: return False, False
		if bed_els < 3: return True, True
		return True, False

	def belongToWindow(cur_block):
		window_els = cur_block.count(window)
		if window_els == 0: return False, None
		return True, window_els == 1
	
	def get_rotation_reflection(data):
		orig_data = data.copy()
		all_data = [data.copy()]
		if not input_data_rotation_reflection: return all_data
		for _ in range(3): # rotation
			new_data = np.zeros((data.shape[1], data.shape[0]), dtype=np.int)
			for i in range(data.shape[0]):
				for j in range(data.shape[1]):
					new_data[data.shape[1] - 1 - j, i] = data[i, j]
			new_data[new_data == window] = wall
			all_data.append(new_data.copy())
			data = new_data.copy()
		# reflection
		data = orig_data
		new_data = data.copy()
		for i in range(data.shape[0]):
			for j in range(data.shape[1]):
				new_data[i, data.shape[1] - 1 - j] = data[i, j]
		all_data.append(new_data.copy())
		new_data = data.copy()
		for i in range(data.shape[0]):
			for j in range(data.shape[1]):
				new_data[data.shape[0] - 1 - i, j] = data[i, j]
		new_data[new_data == window] = wall
		all_data.append(new_data.copy())
		return all_data

	bed_block_orientation = (beds[0], beds[1], beds[bed_size[0]], beds[bed_size[0] + 1])
	bed_blocks_ori = []
	add_rotation_bed_orientation(bed_blocks_ori, bed_block_orientation)
	add_rotation_bed_orientation(bed_blocks_ori, bed_block_orientation, True)
	wall_block = (wall, ) * 4

	# all blocks and frequencies
	all_blocks = []
	all_blocks_freq = {}
	# indices
	door_start_blocks = {
		'left': [],
		'top': [],
		'right': [],
		'bottom': [],
	}
	door_near_blocks = []
	door_center_blocks = []
	window_putted_from_door = []
	window_putted_itself = []
	window_middle_blocks = []
	bed_putted_from_door = []
	bed_putted_itself = []
	bed_oriented = [None] * len(bed_blocks_ori)
	other_blocks = []

	# collecting
	csv_files = glob(os.path.join(input_folder, '*', '*.csv'))
	#cvs_file = 'typical/000038.csv'
	for csv_file in csv_files:
		data = pd.read_csv(csv_file, header=None).values
		all_rotate_reflect_data = get_rotation_reflection(data)
		for data in all_rotate_reflect_data:
			cur_bed_indices = []
			index_ori = None
			for i in range(data.shape[0] - 1):
				for j in range(data.shape[1] - 1):
					cur_block = (data[i, j], data[i, j + 1], data[i + 1, j], data[i + 1, j + 1])
					if cur_block == wall_block: continue
					if cur_block in all_blocks:
						cur_index = all_blocks.index(cur_block)
						new_block = False
						all_blocks_freq[cur_block] += 1
					else:
						cur_index = len(all_blocks)
						new_block = True
						all_blocks_freq[cur_block] = 1
						all_blocks.append(cur_block)
					is_bed, is_bed_st = belongToBed(cur_block)
					if new_block:
						other = True
						is_door, door_st_key, door_els = belongToDoor(cur_block)
						if is_door:
							other = False
							if door_els < 4: door_near_blocks.append(cur_index)
							else: door_center_blocks.append(cur_index)
							if door_st_key is not None:
								door_start_blocks[door_st_key].append(cur_index)
						is_window, is_window_st = belongToWindow(cur_block)
						if is_window:
							other = False
							if is_door: window_putted_from_door.append(cur_index)
							elif is_window_st: window_putted_itself.append(cur_index)
							else: window_middle_blocks.append(cur_index)
						if is_bed:
							other = False
							if is_door: bed_putted_from_door.append(cur_index)
							elif is_bed_st: bed_putted_itself.append(cur_index)
						if other: other_blocks.append(cur_index)
					if is_bed:
						cur_bed_indices.append(cur_index)
						if cur_block in bed_blocks_ori:
							index_ori = bed_blocks_ori.index(cur_block)
			if bed_oriented[index_ori] is None: bed_oriented[index_ori] = cur_bed_indices
			else:
				for ind in cur_bed_indices:
					if ind not in bed_oriented[index_ori]: bed_oriented[index_ori].append(ind)
	for ind in range(len(bed_oriented) - 1, -1, -1):
		if bed_oriented[ind] is None: bed_oriented.pop(ind)
	bed_oriented_search = []
	for cur_list in bed_oriented:
		bed_oriented_search.append([])
		for ind in cur_list:
			if ind not in bed_putted_from_door: bed_oriented_search[-1].append(ind)
	for key in list(door_start_blocks.keys()):
		if len(door_start_blocks[key]) == 0: door_start_blocks.pop(key)
	# add wall
	wall_index = len(all_blocks)
	all_blocks.append(wall_block)
	other_blocks.append(wall_index)
	all_blocks_freq[wall_block] = 1
	# add window
	window_down_start = [len(all_blocks)]
	window_down_start_block = (window, wall, wall, wall)
	all_blocks.append(window_down_start_block)
	all_blocks_freq[window_down_start_block] = 1
	window_down_start.append(len(all_blocks))
	window_down_start_block = (wall, window, wall, wall)
	all_blocks.append(window_down_start_block)
	all_blocks_freq[window_down_start_block] = 1
	window_down_middle = [len(all_blocks)]
	window_down_middle_block = (window, window, wall, wall)
	all_blocks.append(window_down_middle_block)
	all_blocks_freq[window_down_middle_block] = 1

	colors = {}
	min_code = doors[0]
	for code in doors:
		colors[code] = convertColor(code)
		if min_code > code: min_code = code
	for code in beds:
		colors[code] = convertColor(code)
		if min_code > code: min_code = code
	for code in (robe, passage, window, wall):
		colors[code] = convertColor(code)
		if min_code > code: min_code = code
	#for key in all_blocks_freq:
	#	all_blocks_freq[key] = max(int(all_blocks_freq[key] ** 0.5), 1)
	pickle_file_name = os.path.join(input_folder, f'block_collections.pickle')
	with open(pickle_file_name, 'wb') as f:
		pickle.dump((all_blocks, all_blocks_freq,
					 wall_index,
					 bed_oriented, bed_oriented_search, bed_putted_from_door, bed_putted_itself,
					 door_start_blocks, door_center_blocks, door_near_blocks,
					 window_putted_from_door, window_putted_itself, window_middle_blocks,
					 window_down_start, window_down_middle,
					 other_blocks), f)
	return pickle_file_name, colors, min_code

# get_blocks_collections()
