from glob import glob
import pandas as pd
import os, pickle

from commons import *

def get_blocks_collections(input_folder='input_data'):
	def convertColor(code):
		if sitting_all[0] < code < sitting_all[1]: return (((code // 100) * 10) % 256, ((code % 100) * 10) % 256, 0)
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

	def SittingAnalysis(cur_block):
		others = bottoms = walls = 0
		for ind, item in enumerate(cur_block):
			if sitting_all[0] < item < sitting_all[1]:
				if sitting_s[0] < item < sitting_s[1] and 1 < ind < 4: return 'start'
				continue
			if item == wall: walls += 1; continue
			if item == passage_s and 1 < ind < 4 and others == 0: bottoms += 1
			else: others += 1
		if bottoms == 2:
			if others == 0: return 'bottom'
			return 'other'
		if others == 4 - walls: return 'other'
		return 'inside'

	def belongToDining(cur_block):
		for item in cur_block:
			if item in dining: return True
		return False

	def belongToAccess(cur_block):
		for item in cur_block:
			if item in access: return True
		return False

	def cleanBlock(cur_block, colors):
		new_block = []
		for item in cur_block:
			if item == window: item = wall
			if item not in colors: colors[item] = convertColor(item)
			new_block.append(item)
		return tuple(new_block)

	wall_block = (wall, ) * 4
	wall_similar = (wall, access[0], access[1])

	# all blocks and frequencies
	all_blocks = []
	all_blocks_freq = []
	# indices
	door_start_blocks = {
		'left': [],
		'top': [],
		'right': [],
		'bottom': [],
	}
	door_near_blocks = []
	door_center_blocks = []
	tansuo_blocks = []
	dining_starts = []
	dining_from_door = []
	acess_from_door = []
	sitting_starts = []
	access_starts = []
	acess_sitting_common = []

	# collecting
	colors = {}
	sitting_bottom = 0
	min_code = None
	csv_files = glob(os.path.join(input_folder, '*', '*.csv'))
	for csv_file in csv_files:
		data = pd.read_csv(csv_file, header=None).values
		sitting_bottomed = False
		for i in range(data.shape[0] - 1):
			for j in range(data.shape[1] - 1):
				cur_block = cleanBlock((data[i, j], data[i, j + 1], data[i + 1, j], data[i + 1, j + 1]), colors)
				if cur_block == wall_block: continue
				if cur_block in all_blocks:
					cur_index = all_blocks.index(cur_block)
					all_blocks_freq[cur_index] += 1
					continue
				cur_index = len(all_blocks)
				if cur_block == (3020, 0, 3020, 0):
					aaa = 1
				is_door, door_st_key, door_els = belongToDoor(cur_block)
				all_blocks.append(cur_block)
				all_blocks_freq.append(1)
				if min_code is None: min_code = min(cur_block)
				else: min_code = min(min_code, min(cur_block))
				is_dinning = belongToDining(cur_block)
				is_access = belongToAccess(cur_block)
				if is_door:
					if door_st_key is not None: door_start_blocks[door_st_key].append(cur_index)
					elif door_els < 4: door_near_blocks.append(cur_index)
					else: door_center_blocks.append(cur_index)
					if is_dinning: dining_from_door.append(cur_index)
					if is_access: acess_from_door.append(cur_index)
					continue
				if is_dinning:
					dining_starts.append(cur_index)
					continue
				info = SittingAnalysis(cur_block)
				if info == 'start': is_sitting = True
				elif info == 'bottom':
					sitting_bottomed = True
					continue
				elif info == 'inside': is_sitting = True
				else: is_sitting = False
				if is_access:
					if is_sitting: acess_sitting_common.append(cur_index)
					else: access_starts.append(cur_index)
				elif is_sitting: sitting_starts.append(cur_index)
				else: tansuo_blocks.append(cur_index)
		if sitting_bottomed: sitting_bottom += 1
	# add wall
	all_blocks.append(wall_block)
	empty_keys = [key for key in door_start_blocks if len(door_start_blocks[key]) == 0]
	for key in empty_keys: door_start_blocks.pop(key)
	passage_wide = (passage_w, passage_w, passage_w, passage_w)
	cur_index = all_blocks.index(passage_wide)
	all_blocks_freq[cur_index] = all_blocks_freq[cur_index] // 4

	pickle_file_name = os.path.join(input_folder, 'block_collections.pickle')
	with open(pickle_file_name, 'wb') as f:
		pickle.dump((all_blocks, all_blocks_freq, wall_similar,
					door_start_blocks, door_near_blocks, door_center_blocks,
					tansuo_blocks, dining_starts, dining_from_door, acess_from_door,
					sitting_starts, access_starts, acess_sitting_common,
					sitting_bottom / len(csv_files)), f)
	return pickle_file_name, colors, min_code

# get_blocks_collections()
