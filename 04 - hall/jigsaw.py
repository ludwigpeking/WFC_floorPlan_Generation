import cv2
import numpy as np
from datetime import datetime
from scipy import stats
import os

from blocks import get_blocks_collections, pd, os, pickle
from commons import *
from dxf_blocks import draw_dxf


def save_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(filename):
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return {}

def complete_B(val, row, col, k):
    row_n = col_n = None
    if k == 3: row_n, col_n = row, col
    elif k == 2:
        if col > 0: row_n, col_n = row, col - 1
    elif k == 1:
        if row > 0: row_n, col_n = row - 1, col
    elif row > 0 and col > 0: row_n, col_n = row - 1, col - 1
    if row_n is not None:
        if row_n < n - 1 and col_n < n - 1:
            if B[row_n, col_n, 3 - k] == default_code:
                B[row_n, col_n, 3 - k] = val
                if default_code not in B[row_n, col_n]:
                    B_cl[row_n, col_n] = val
                    if val == passage_w: passage_w_new_cells.append((row_n, col_n))
                    elif val in access:
                        global accessings
                        accessings += 1

def setting(src_array, row, col, val_array):
    for k in range(4):
        val = src_array[row, col, k] = val_array[k]
        complete_B(val, row, col, k)

def extend(src_array, row, col, flag_array):
    add_indices = []
    if col > 0:
        if flag_array[row, col - 1] == 0:
            val = src_array[row, col - 1, 1] = src_array[row, col, 0]
            complete_B(val, row, col - 1, 1)
            val = src_array[row, col - 1, 3] = src_array[row, col, 2]
            complete_B(val, row, col - 1, 3)
            add_indices.append((row, col - 1))
    if col < A.shape[1] - 1:
        if flag_array[row, col + 1] == 0:
            val = src_array[row, col + 1, 0] = src_array[row, col, 1]
            complete_B(val, row, col + 1, 0)
            val = src_array[row, col + 1, 2] = src_array[row, col, 3]
            complete_B(val, row, col + 1, 2)
            add_indices.append((row, col + 1))
    if row > 0:
        if flag_array[row - 1, col] == 0:
            val = src_array[row - 1, col, 2] = src_array[row, col, 0]
            complete_B(val, row - 1, col, 2)
            val = src_array[row - 1, col, 3] = src_array[row, col, 1]
            complete_B(val, row - 1, col, 3)
            add_indices.append((row - 1, col))
    if row < A.shape[0] - 1:
        if flag_array[row + 1, col] == 0:
            val = src_array[row + 1, col, 0] = src_array[row, col, 2]
            complete_B(val, row + 1, col, 0)
            val = src_array[row + 1, col, 1] = src_array[row, col, 3]
            complete_B(val, row + 1, col, 1)
            add_indices.append((row + 1, col))
    clean_indices = []
    for (row, col) in add_indices:
        valid = False
        for k in range(4):
            if src_array[row, col, k] < min_code: continue
            if src_array[row, col, k] == wall: continue
            if src_array[row, col, k] in access: continue
            valid = True; break
        if valid: clean_indices.append((row, col))
    return clean_indices  # add_indices

def Show(src_array, src_array_cl, row_=None, col_=None, color=(200, 200, 200), thickness=2, saving_path=None):
    if not showing and saving_path is None: return
    # src_array
    cell_half_size = cell_size // 2
    img = np.zeros((cell_size * n_ex, cell_size * n_ex, 3), dtype=np.uint8)
    img[:] = 255
    img[:cell_half_size, cell_half_size:-cell_half_size] = 0
    img[-cell_half_size:, cell_half_size:-cell_half_size] = 0
    img[cell_half_size:-cell_half_size, :cell_half_size] = 0
    img[cell_half_size:-cell_half_size, -cell_half_size:] = 0
    clean_img = img[cell_half_size:-cell_half_size, cell_half_size:-cell_half_size]
    for row in range(src_array.shape[0]):
        cur_img_row = clean_img[row * cell_size:(row + 1) * cell_size]
        for col in range(src_array.shape[1]):
            cur_img = cur_img_row[:, col * cell_size:(col + 1) * cell_size]
            if src_array[row, col, 0] > default_code:
                for c in range(3):
                    cur_img[:cell_half_size, :cell_half_size, c] = colors[src_array[row, col, 0]][c]
            if src_array[row, col, 1] > default_code:
                for c in range(3):
                    cur_img[:cell_half_size, cell_half_size:, c] = colors[src_array[row, col, 1]][c]
            if src_array[row, col, 2] > default_code:
                for c in range(3):
                    cur_img[cell_half_size:, :cell_half_size, c] = colors[src_array[row, col, 2]][c]
            if src_array[row, col, 3] > default_code:
                for c in range(3):
                    cur_img[cell_half_size:, cell_half_size:, c] = colors[src_array[row, col, 3]][c]
        cur_img_row[-1, :, :] = 0
    for col in range(src_array.shape[1]):
        clean_img[:, col * cell_size, :] = 0
    # src_array_cl
    img_cl = np.zeros((cell_size * n, cell_size * n, 3), dtype=np.uint8)
    img_cl[:] = 255
    clean_img = img_cl[cell_half_size:-cell_half_size, cell_half_size:-cell_half_size]
    for row in range(src_array_cl.shape[0]):
        cur_img_row = clean_img[row * cell_size:(row + 1) * cell_size]
        for col in range(src_array_cl.shape[1]):
            cur_img = cur_img_row[:, col * cell_size:(col + 1) * cell_size]
            if src_array_cl[row, col, 0] > default_code:
                for c in range(3):
                    cur_img[:cell_half_size, :cell_half_size, c] = colors[src_array_cl[row, col, 0]][c]
            if src_array_cl[row, col, 1] > default_code:
                for c in range(3):
                    cur_img[:cell_half_size, cell_half_size:, c] = colors[src_array_cl[row, col, 1]][c]
            if src_array_cl[row, col, 2] > default_code:
                for c in range(3):
                    cur_img[cell_half_size:, :cell_half_size, c] = colors[src_array_cl[row, col, 2]][c]
            if src_array_cl[row, col, 3] > default_code:
                for c in range(3):
                    cur_img[cell_half_size:, cell_half_size:, c] = colors[src_array_cl[row, col, 3]][c]
        cur_img_row[-1, :, :] = 0
    for col in range(src_array_cl.shape[1]):
        clean_img[:, col * cell_size, :] = 0
    clean_img[0, :, :] = 0
    clean_img[:, -1, :] = 0
    if col_ is not None and row_ is not None:
        cv2.rectangle(img, (col_ * cell_size + cell_half_size, row_ * cell_size + cell_half_size), ((col_ + 1) * cell_size + cell_half_size, (row_ + 1) * cell_size + cell_half_size), color, thickness)
    if showing:
        cv2.imshow('111', img)
        cv2.imshow('222', img_cl)
        cv2.waitKey(duration)
    if saving_path is not None:
        main_name, ext_name = os.path.splitext(saving_path)
        if 'failed' in main_name:
            cv2.imwrite(main_name + '-1' + ext_name, img)
            cv2.imwrite(main_name + '-2' + ext_name, img_cl)
        else: cv2.imwrite(saving_path, img_cl)

def calcEntropy(bl_indices):
    if len(bl_indices) == 1: return 0
    data = []
    for ind in bl_indices: data += [ind] * all_blocks_freq[ind]
    pd_series = pd.Series(data)
    counts = pd_series.value_counts()
    entropy = stats.entropy(counts)
    return entropy

def frequencyChoice(bl_indices):
    if len(bl_indices) == 1: return bl_indices[0]
    nums = []
    for ind in bl_indices: nums += [ind] * all_blocks_freq[ind]
    return np.random.choice(nums)

def Complete():
    m, n = B_cl.shape
    B_cl[B_cl == default_code] = wall
    B_cl_o = B_cl.copy()
    for el in access:
        B_cl[B_cl == el] = wall
    Storage = (B_cl == robe).sum()
    if Storage == 0: return
    yy, xx = np.where(B_cl != wall)
    access_outline_cells = get_access_info(B_cl_o)
    outline_cells = []
    additional_cells = []
    for i, j in zip(yy, xx):
        if i > 0:
            if B_cl[i - 1, j] == 0: outline_cells.append((i, j, 0))
        else: outline_cells.append((i, j, 0))
        if i < m - 1:
            if B_cl[i + 1, j] == 0: outline_cells.append((i, j, 1))
        else: outline_cells.append((i, j, 1))
        if j < n - 1:
            if B_cl[i, j + 1] == 0: outline_cells.append((i, j, 2))
        else: outline_cells.append((i, j, 2))
        if j > 0:
            if B_cl[i, j - 1] == 0: outline_cells.append((i, j, 3))
        else: outline_cells.append((i, j, 3))
        if (i, j, 0) in outline_cells:
            if (i, j, 2) in outline_cells: additional_cells.append((i, j, 2))
            if (i, j, 3) in outline_cells: additional_cells.append((i, j, 3))
        if (i, j, 1) in outline_cells:
            if (i, j, 2) in outline_cells: additional_cells.append((i, j, 12))
            if (i, j, 3) in outline_cells: additional_cells.append((i, j, 13))
    AreaCell = len(yy)
    Perimeter = len(outline_cells)
    AreaReality = AreaCell * 0.55 ** 2
    Scoring = -0.1 * (AreaReality - 25) ** 2 + 10
    Storage_s = 5 - 1 / Storage ** 0.8 * 2.5
    PeriEfficiency = AreaCell / Perimeter ** 2
    Scoring_pe = PeriEfficiency * 80
    TotalScore = Scoring + Storage_s + Scoring_pe
    # reduce
    min_i = max_i = None
    for i in range(B_cl_o.shape[0]):
        if np.all(B_cl_o[i] == wall): continue
        min_i = i; break
    for i in range(B_cl_o.shape[0] - 1, -1, -1):
        if np.all(B_cl_o[i] == wall): continue
        max_i = i + 1; break
    min_j = max_j = None
    for j in range(B_cl_o.shape[1]):
        if np.all(B_cl_o[:, j] == wall): continue
        min_j = j; break
    for j in range(B_cl_o.shape[1] - 1, -1, -1):
        if np.all(B_cl_o[:, j] == wall): continue
        max_j = j + 1; break
    B_cl_re = B_cl_o[min_i:max_i, min_j:max_j]
    duplicated_key = AreaCell, Perimeter, Storage, B_cl_re.shape[0], B_cl_re.shape[1]
    found_patterns = load_pickle(pklName)
    if duplicated_key in found_patterns:
        for pattern in found_patterns[duplicated_key]:
            if np.all(pattern == B_cl_re): return
        found_patterns[duplicated_key].append(B_cl_re.copy())
    else: found_patterns[duplicated_key] = [B_cl_re.copy()]

    global success_number, result_folder
    cell_half_size = cell_size // 2
    img_all = np.zeros((cell_size * (m + 1), cell_size * (n + 2) + 300, 3), dtype=np.uint8)
    img_all[:] = 255
    img = img_all[:cell_size * (m + 1), :cell_size * (n + 1)]
    clean_img = img[cell_half_size:-cell_half_size, cell_half_size:-cell_half_size]
    for i, j in zip(yy, xx):
        cur_img = clean_img[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]
        for c in range(3):
            cur_img[:, :, c] = colors[B_cl[i, j]][c]
    no_access_outline_cells = []
    for i, j, k in outline_cells:
        if (i, j, k) in access_outline_cells: continue
        no_access_outline_cells.append((i, j, k))
        top, left = i * cell_size + cell_half_size, j * cell_size + cell_half_size
        bottom, right = top + cell_size, left + cell_size
        if k == 0: cur_img = img[top - cell_half_size:top, left:right]
        elif k == 1: cur_img = img[bottom:bottom + cell_half_size, left:right]
        elif k == 2: cur_img = img[top:bottom, right:right + cell_half_size]
        else: cur_img = img[top:bottom, left - cell_half_size:left]
        for c in range(3):
            cur_img[:, :, c] = colors[wall][c]
    for ii, (i, j, k) in enumerate(access_outline_cells):
        top, left = i * cell_size + cell_half_size, j * cell_size + cell_half_size
        bottom, right = top + cell_size, left + cell_size
        if k == 0: cur_img = img[top - cell_half_size:top, left:right]
        elif k == 1: cur_img = img[bottom:bottom + cell_half_size, left:right]
        elif k == 2: cur_img = img[top:bottom, right:right + cell_half_size]
        else: cur_img = img[top:bottom, left - cell_half_size:left]
        for c in range(3):
            cur_img[:, :, c] = colors[access[ii % 2]][c]
    for i, j, k in additional_cells:
        top, left = i * cell_size + cell_half_size, j * cell_size + cell_half_size
        bottom, right = top + cell_size, left + cell_size
        if k == 2: cur_img = img[top - cell_half_size:top, right:right + cell_half_size]
        elif k == 3: cur_img = img[top - cell_half_size:top, left - cell_half_size:left]
        elif k == 12: cur_img = img[bottom:bottom + cell_half_size, right:right + cell_half_size]
        else: cur_img = img[bottom:bottom + cell_half_size, left - cell_half_size:left]
        for c in range(3):
            cur_img[:, :, c] = colors[wall][c]
    img = img_all[:, cell_size * (n + 2):]
    x1 = 0; x2 = 200; y = 2 * cell_size; y_step = cell_size
    result_str = '%d' % success_number
    cv2.putText(img, 'No:', (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    cv2.putText(img, '%d' % success_number, (x2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    y += y_step
    result_str += ',%d' % AreaCell
    cv2.putText(img, 'Area Cell:', (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    cv2.putText(img, '%d' % AreaCell, (x2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    y += y_step
    result_str += ',%.2f' % AreaReality
    cv2.putText(img, 'Area Reality:', (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    cv2.putText(img, '%.2f' % AreaReality, (x2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    y += y_step
    result_str += ',%.2f' % Scoring
    cv2.putText(img, 'Scoring:', (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    cv2.putText(img, '%.2f' % Scoring, (x2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    y += y_step
    result_str += ',%d' % Storage
    cv2.putText(img, 'Storage:', (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    cv2.putText(img, '%d' % Storage, (x2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    y += y_step
    result_str += ',%.2f' % Storage_s
    cv2.putText(img, 'Scoring:', (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    cv2.putText(img, '%.2f' % Storage_s, (x2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    y += y_step
    result_str += ',%d' % Perimeter
    cv2.putText(img, 'Perimeter:', (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    cv2.putText(img, '%d' % Perimeter, (x2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    y += y_step
    result_str += ',%.4f' % PeriEfficiency
    cv2.putText(img, 'Peri-Efficiency:', (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    cv2.putText(img, '%.4f' % PeriEfficiency, (x2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    y += y_step
    result_str += ',%.2f' % Scoring_pe
    cv2.putText(img, 'Scoring:', (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    cv2.putText(img, '%.2f' % Scoring_pe, (x2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    y += y_step * 2
    result_str += ',%.2f' % TotalScore
    cv2.putText(img, 'Total Score:', (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
    cv2.putText(img, '%.2f' % TotalScore, (x2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
    result_str += '\n'
    if showing:
        cv2.imshow('111', img_all)
        cv2.waitKey(duration)
    if success_number == 1:
        os.makedirs(result_folder)
        with open(csvName, 'wt') as f:
            f.write('No,Area Cell,Area Reality,Scoring,Storage,Scoring,Perimeter,Peri-Efficiency,Scoring,Total Score\n')
    baseName = result_folder + '/%06d' % success_number
    imgName = baseName + '.png'
    cv2.imwrite(imgName, img_all)
    with open(csvName, 'at') as f:
        f.write(result_str)
    save_pickle(found_patterns, pklName)
    pickleName = baseName + '.pkl'
    save_pickle((B_cl_o.copy(), access_outline_cells.copy()), pickleName)
    dxf_name = baseName + '.dxf'
    draw_dxf(B_cl_o, no_access_outline_cells, access_outline_cells, additional_cells, dxf_name)
    success_number += 1

def check_B(B_cell):
    for i in B_cell:
        if i not in (default_code, passage_w): return False
    return True

def get_access_pos(B_cl):
    yy, xx = np.where(B_cl == access[0])
    pos1 = [(i, j) for i, j in zip(yy, xx)]
    yy, xx = np.where(B_cl == access[1])
    pos2 = [(i, j) for i, j in zip(yy, xx)]
    return pos1, pos2

def check_access(B_cl):
    pos1, pos2 = get_access_pos(B_cl)
    if len(pos1) != 4: return False
    for i, j in pos1:
        near_pos = i, j + 1
        if near_pos in pos2:
            pos2.remove(near_pos)
            continue
        near_pos = i + 1, j
        if near_pos not in pos2: return False
        pos2.remove(near_pos)
    return True

def get_access_info(B_cl):
    pos1, pos2 = get_access_pos(B_cl)
    access_outline_cells = []
    for i, j in pos1:
        found = False
        near_pos = i, j + 1
        if near_pos in pos2:
            pos2.remove(near_pos)
            if i > 0:
                if B_cl[i - 1, j] != wall:
                    access_outline_cells.append((i - 1, j, 1))
                    access_outline_cells.append((i - 1, j + 1, 1))
                    found = True
            if not found:
                access_outline_cells.append((i + 1, j, 0))
                access_outline_cells.append((i + 1, j + 1, 0))
        else:
            near_pos = i + 1, j
            pos2.remove(near_pos)
            if j > 0:
                if B_cl[i, j - 1] != wall:
                    access_outline_cells.append((i, j - 1, 2))
                    access_outline_cells.append((i + 1, j - 1, 2))
                    found = True
            if not found:
                access_outline_cells.append((i, j + 1, 3))
                access_outline_cells.append((i + 1, j + 1, 3))
    return access_outline_cells
    
print('Started at ', datetime.now())
result_folder = 'results(' + str(datetime.now()).replace(':', '_')[:19] + ')'
success_number = 1
n_ex = n + 1
pickle_file_name, colors, min_code = get_blocks_collections()
default_code = min_code - 1
csvName = result_folder + '/all.csv'
pklName = result_folder + '/all.pkl'
result_folder += '/dxf_png'

for iii in range(loops):
    with open(pickle_file_name, 'rb') as f:
        blocks, all_blocks_freq, wall_similar,\
        door_start_blocks, door_near_blocks, door_center_blocks,\
        tansuo_blocks, dining_starts, dining_from_door, acess_from_door, \
        sitting_starts, access_starts, acess_sitting_common,\
        sitting_bottom_ratio = pickle.load(f)
        passage_w_block = (passage_w, passage_w, passage_w, passage_w)

    # initialize
    C = np.zeros((n, n), dtype=np.uint8)
    A = np.ones((n, n, 4), dtype=np.int) * default_code
    B = np.ones((n - 1, n - 1, 4), dtype=np.uint8) * default_code
    B_cl = np.ones((n - 1, n - 1), dtype=np.uint8) * default_code
    passage_w_new_cells = []
    A[0, :, :2] = wall
    A[:, 0, 0] = wall
    A[:, 0, 2] = wall
    A[:, -1, 1] = wall
    A[:, -1, 3] = wall
    A[-1, :, 2:] = wall
    #A[-1, 0, 3] = wall
    #A[-1, -1, 2] = wall
    Show(A, B)

    # starting
    door_start_block_keys = list(door_start_blocks.keys())
    start_key = np.random.choice(door_start_block_keys)
    i_j = np.random.randint(1, n - 2)
    if i_j == n - 2: i_j = n - 3
    if start_key == 'left':
        i, j = i_j, 0
        new_index = i, j + 1
    elif start_key == 'right':
        i, j = np.random.randint(1, 5), n - 1
        new_index = i, j - 1
    elif start_key == 'top':
        i, j = 1, np.random.randint(n - 5, n - 2)
        new_index = i + 1, j
    else:
        i, j = n - 1, i_j
        new_index = i - 1, j
    bl_indices = door_start_blocks[start_key]
    del door_start_block_keys, start_key
    dinnings = sittings = accessings = 0

    # complete doors
    door_failed = False
    new_indices = []
    while True:
        bl_index = frequencyChoice(bl_indices)
        if bl_index in door_near_blocks: door_near_blocks.remove(bl_index)
        setting(A, i, j, blocks[bl_index])
        Show(A, B, i, j)
        C[i, j] = 1
        new_indices_t = extend(A, i, j, C)
        for index in new_indices_t:
            if index not in new_indices: new_indices.append(index)
        Show(A, B, i, j)
        if bl_index in dining_from_door:
            dinnings += 1
        # new search
        if new_index is not None:
            door_existing = True
            i, j = new_index
            new_index = None
            bl_indices = []
            for ind in door_center_blocks:
                block = blocks[ind]
                same = True
                for k in range(4):
                    if A[i, j, k] < min_code: continue
                    if A[i, j, k] != block[k]: same = False; break
                if same: bl_indices.append(ind)
        else:
            door_existing = False
            for i, j in new_indices:
                for k in range(4):
                    if A[i, j, k] in doors: door_existing = True; break
                if door_existing: break
            if not door_existing: break
            bl_indices = []
            for ind in door_near_blocks:
                block = blocks[ind]
                same = True
                for k in range(4):
                    if A[i, j, k] < min_code:
                        if block[k] in doors: same = False; break
                    elif A[i, j, k] != block[k]: same = False; break
                if same: bl_indices.append(ind)
        if len(bl_indices) < 1:
            Show(A, B, i, j, (0, 0, 255), 2)
            door_failed = True; break
        new_indices.remove((i, j))
    if door_failed:
        if showing: cv2.waitKey(1000)
        continue

    # next
    #cv2.waitKey(0)
    collaps = {}
    while True:
        if len(passage_w_new_cells) > 0:
            candidates_all = []
            single_cell = False
            for row, col in passage_w_new_cells:
                found, candidates = False, []
                if row > 0:
                    if B_cl[row - 1, col] == default_code:
                        if check_B(B[row - 1, col]): candidates.append((row - 1, col))
                    else: found = B_cl[row - 1, col] == passage_w
                if not found and row < n - 2:
                    if B_cl[row + 1, col] == default_code:
                        if check_B(B[row + 1, col]): candidates.append((row + 1, col))
                    else: found = B_cl[row + 1, col] == passage_w
                if not found:
                    if len(candidates) == 0: single_cell = True; break
                    candidates_all.append(candidates[0])
                found, candidates = False, []
                if col > 0:
                    if B_cl[row, col - 1] == default_code:
                        if check_B(B[row, col - 1]): candidates.append((row, col - 1))
                    else: found = B_cl[row, col - 1] == passage_w
                if not found and col < n - 2:
                    if B_cl[row, col + 1] == default_code:
                        if check_B(B[row, col + 1]): candidates.append((row, col + 1))
                    else: found = B_cl[row, col + 1] == passage_w
                if not found:
                    if len(candidates) == 0: single_cell = True; break
                    candidates_all.append(candidates[0])
            if single_cell: break
            i_s, j_s = (0, 0, 1, 1), (0, 1, 0, 1)
            for i, j in candidates_all:
                B_cl[i, j] = passage_w
                A[i, j, 3] = A[i, j + 1, 2] = A[i + 1, j, 1] = A[i + 1, j + 1, 0] = passage_w
                for k in range(4):
                    if B[i, j, k] == passage_w: continue
                    B[i, j, k] = passage_w
                    row, col = i + i_s[k], j + j_s[k]
                    A[row, col, 3 - k] = passage_w
                    index = (row, col)
                    if default_code in A[row, col]:
                        if index not in new_indices: new_indices.append(index)
                    else:
                        C[row, col] = 1
                        if index in new_indices: new_indices.remove(index)
                        if index in collaps: collaps.pop(index)
            Show(A, B)
            passage_w_new_cells = []
        impasse_indices = []; good_indices = []
        for index in new_indices:
            i, j = index
            is_impasse = True
            for k in range(4):
                if A[i, j, k] < min_code: continue
                #if A[i, j, k] in access: continue
                if A[i, j, k] not in (wall, robe): is_impasse = False; break
            if is_impasse: impasse_indices.append(index)
            else: good_indices.append(index)
        if len(good_indices) > 0: new_indices = good_indices
        else: new_indices = impasse_indices; impasse_indices = []
        blocks_indices = tansuo_blocks.copy()
        if dinnings < 12: blocks_indices += dining_starts
        if sittings < 42:
            if accessings < 8: blocks_indices += sitting_starts + access_starts + acess_sitting_common
            else: blocks_indices += sitting_starts
        elif accessings < 8: blocks_indices += access_starts
        for index in new_indices:
            i, j = index
            A_have_pass_w = passage_w in A[i, j]
            A_have_pass_s = passage_s in A[i, j]
            A_have_pass = A_have_pass_w or A_have_pass_s
            bl_indices = []
            for ind in blocks_indices:
                block = blocks[ind]
                same = True
                bl_have_pass_w = bl_have_pass_s = bl_have_robe = False
                for k in range(4):
                    if A[i, j, k] < min_code:
                        if block[k] == robe: bl_have_robe = True
                        elif block[k] == passage_w: bl_have_pass_w = True
                        elif block[k] == passage_s: bl_have_pass_s = True
                    elif A[i, j, k] != block[k]: same = False; break
                if same:
                    if bl_have_pass_w: same = A_have_pass_w
                    elif bl_have_pass_s: same = A_have_pass
                    elif bl_have_robe: same = A_have_pass
                    if same: bl_indices.append(ind)
                    elif len(bl_indices) == 0: bl_indices = [None]
            if len(bl_indices) > 0:
                if bl_indices[0] is None:
                    bl_indices.pop(0)
                if len(bl_indices) > 0: collaps[index] = bl_indices
                elif index in collaps: collaps.pop(index)
            else:
                collaps[index] = bl_indices
        if len(collaps) == 0:
            if sittings == 42 and dinnings == 12 and accessings == 8:
                # Show(A, B, saving_path='temp.png')
                # A[A == default_code] = wall
                if check_access(B_cl): Complete()
            break
        cur_keys = tuple(collaps.keys())
        [calcEntropy(collaps[key]) for key in cur_keys]
        Entropies = [calcEntropy(collaps[key]) for key in cur_keys]
        min_entropy = min(Entropies)
        min_indices = [cur_keys[ind] for ind, entropy in enumerate(Entropies) if entropy == min_entropy]
        # random choice
        next_index = min_indices[np.random.choice(range(len(min_indices)))]
        bl_indices = collaps.pop(next_index)
        i, j = next_index
        if len(bl_indices) == 0:
            break
        bl_index = frequencyChoice(bl_indices)
        # setting
        setting(A, i, j, blocks[bl_index])
        Show(A, B, i, j)
        C[i, j] = 1
        if bl_index in dining_starts:
            dinnings += 1
        if bl_index in sitting_starts:
            sittings += 1
        if bl_index in acess_sitting_common:
            sittings += 1
        new_indices = extend(A, i, j, C); Show(A, B, i, j)
        if len(impasse_indices) > 0:
            for index in impasse_indices:
                if index not in new_indices: new_indices.append(index)
    if showing: cv2.waitKey(1000)
print('Finished at ', datetime.now())
