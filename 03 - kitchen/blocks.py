from glob import glob
import pandas as pd
import os, pickle

from commons import *

def get_blocks_collections(input_folder='input_data'):
	def convertColor(code):
		return (0, ((code % 100) * 10) % 256, ((code // 100) * 10) % 256)

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
	other_blocks = []

	# collecting
	csv_files = glob(os.path.join(input_folder, '*', '*.csv'))
	for csv_file in csv_files:
		data = pd.read_csv(csv_file, header=None).values
		cur_bed_indices = []
		for i in range(data.shape[0] - 1):
			for j in range(data.shape[1] - 1):
				cur_block = (data[i, j], data[i, j + 1], data[i + 1, j], data[i + 1, j + 1])
				if cur_block == wall_block: continue
				if window in cur_block:
					cur_block_ = [(wall if item == window else item) for item in cur_block]
					if cur_block_ == wall_block: continue
					del cur_block_
				if cur_block in all_blocks:
					cur_index = all_blocks.index(cur_block)
					new_block = False
					all_blocks_freq[cur_block] += 1
				else:
					cur_index = len(all_blocks)
					new_block = True
					all_blocks_freq[cur_block] = 1
					all_blocks.append(cur_block)
				if new_block:
					is_door, door_st_key, door_els = belongToDoor(cur_block)
					if is_door:
						if door_els < 4: door_near_blocks.append(cur_index)
						else: door_center_blocks.append(cur_index)
						if door_st_key is not None:
							door_start_blocks[door_st_key].append(cur_index)
					else: other_blocks.append(cur_index)
	for key in list(door_start_blocks.keys()):
		if len(door_start_blocks[key]) == 0: door_start_blocks.pop(key)
	# add wall
	wall_index = len(all_blocks)
	all_blocks.append(wall_block)
	other_blocks.append(wall_index)
	all_blocks_freq[wall_index] = 1

	colors = {}
	min_code = doors[0]
	for code in doors:
		colors[code] = convertColor(code)
		if min_code > code: min_code = code
	for code in (passage, window, wall, stove, frig, counter, shaft, sink):
		colors[code] = convertColor(code)
		if min_code > code: min_code = code

	pickle_file_name = os.path.join(input_folder, 'block_collections.pickle')
	with open(pickle_file_name, 'wb') as f:
		pickle.dump((all_blocks, all_blocks_freq,
					 wall_index,
					 door_start_blocks, door_center_blocks, door_near_blocks,
					 other_blocks), f)
	return pickle_file_name, colors, min_code

get_blocks_collections()
