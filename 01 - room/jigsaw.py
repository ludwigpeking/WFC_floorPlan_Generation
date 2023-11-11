import cv2
from datetime import datetime
from scipy import stats

from blocks import get_blocks_collections, pd, os, pickle
from commons import *
from dxf_blocks import draw_dxf, np


def save_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(filename):
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return {}

def setting(src_array, row, col, val_array):
    for k in range(4):
        src_array[row, col, k] = val_array[k]

def extend(src_array, row, col, flag_array):
    add_indices = []
    if col > 0:
        if flag_array[row, col - 1] == 0:
            src_array[row, col - 1, 1] = src_array[row, col, 0]
            src_array[row, col - 1, 3] = src_array[row, col, 2]
            add_indices.append((row, col - 1))
    if col < src_array.shape[1] - 1:
        if flag_array[row, col + 1] == 0:
            src_array[row, col + 1, 0] = src_array[row, col, 1]
            src_array[row, col + 1, 2] = src_array[row, col, 3]
            add_indices.append((row, col + 1))
    if row > 0:
        if flag_array[row - 1, col] == 0:
            src_array[row - 1, col, 2] = src_array[row, col, 0]
            src_array[row - 1, col, 3] = src_array[row, col, 1]
            add_indices.append((row - 1, col))
    if row < src_array.shape[0] - 1:
        if flag_array[row + 1, col] == 0:
            src_array[row + 1, col, 0] = src_array[row, col, 2]
            src_array[row + 1, col, 1] = src_array[row, col, 3]
            add_indices.append((row + 1, col))
    clean_indices = []
    for (row, col) in add_indices:
        valid = False
        for k in range(4):
            if src_array[row, col, k] < min_code: continue
            if src_array[row, col, k] not in (wall, window): valid = True; break
        if valid: clean_indices.append((row, col))
    return clean_indices  # add_indices

def Show(src_array, row_=None, col_=None, color=(200, 200, 200), thickness=2):
    if not showing: return
    cell_size = 56
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
    if col_ is not None and row_ is not None:
        cv2.rectangle(img, (col_ * cell_size + cell_half_size, row_ * cell_size + cell_half_size), ((col_ + 1) * cell_size + cell_half_size, (row_ + 1) * cell_size + cell_half_size), color, thickness)
    cv2.imshow('111', img)
    cv2.waitKey(duration)

def calcEntropy(bl_indices, all_blocks_freq, blocks):
    if len(bl_indices) == 1: return 0
    data = []
    for ind in bl_indices: data += [ind] * all_blocks_freq[blocks[ind]]
    pd_series = pd.Series(data, dtype='int')
    counts = pd_series.value_counts()
    entropy = stats.entropy(counts)
    return entropy

def frequencyChoice(bl_indices, all_blocks_freq, blocks):
    if len(bl_indices) == 1: return bl_indices[0]
    nums = []
    for ind in bl_indices: nums += [ind] * all_blocks_freq[blocks[ind]]
    return np.random.choice(nums)

def Complete(A, ScoreThreshold):
    global success_number, result_folder

    AA = A.copy()
    A[A == window] = wall
    C = np.zeros((n + 1, n + 1), dtype=np.int)
    B = C[1:-1, 1:-1]
    Perimeter = AreaCell = Storage = 0
    for i in range(n - 1):
        for j in range(n - 1):
            collections = A[i, j, 3], A[i, j + 1, 2], A[i + 1, j, 1], A[i + 1, j + 1, 0]
            if min(collections) < max(collections): return
            B[i, j] = A[i, j, 3]
            if B[i, j] == robe: Storage += 1
    if Storage == 0: return
    for i in range(1, n):
        for j in range(1, n):
            if C[i, j] == robe:
                if C[i - 1, j] in robe_in_passages: continue
                if C[i + 1, j] in robe_in_passages: continue
                if C[i, j + 1] in robe_in_passages: continue
                if C[i, j - 1] in robe_in_passages: continue
                return
    A[AA == window] = window
    if len(np.unique(np.where(A == window)[0])) == 2:
        for i in range(A.shape[0] - 1, -1, -1):
            if window in A[i]:
                last_row = A[i]
                last_row[last_row == window] = wall
                break
    del AA
    B = np.ones_like(C)
    B[0, 0] = 0
    C[0, 0] = 1
    old_indices = [(0, 0)]
    while len(old_indices) > 0:
        new_indices = []
        for index in old_indices:
            i, j = index
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                new_i = i + di
                new_j = j + dj
                if new_i < 0: continue
                if new_j < 0: continue
                if new_i > n: continue
                if new_j > n: continue
                if C[new_i, new_j] == 0:
                    B[new_i, new_j] = 0
                    new_indices.append((new_i, new_j))
                    C[new_i, new_j] = 1
        old_indices = new_indices.copy()
    for i in range(1, n):
        for j in range(1, n):
            if B[i, j] > 0:
                AreaCell += 1
                if B[i - 1, j] == 0: Perimeter += 1
                if B[i + 1, j] == 0: Perimeter += 1
                if B[i, j + 1] == 0: Perimeter += 1
                if B[i, j - 1] == 0: Perimeter += 1
    AreaReality = AreaCell * 0.55 ** 2
    Scoring = -0.04 * (AreaReality - 10) ** 2 + 10
    Storage_s = Storage ** 0.7
    PeriEfficiency = AreaCell / Perimeter ** 2
    Scoring_pe = PeriEfficiency * 180
    TotalScore = Scoring + Storage_s + Scoring_pe
    if TotalScore < ScoreThreshold: return
    # checking duplicated
    B_cl = np.zeros((n - 1, n - 1), dtype=np.int)
    for i in range(n - 1):
        for j in range(n - 1):
            collections = A[i, j, 3], A[i, j + 1, 2], A[i + 1, j, 1], A[i + 1, j + 1, 0]
            if min(collections) < max(collections): 
                if window in collections: B_cl[i, j] = window
            else: B_cl[i, j] = A[i, j, 3]
    min_i = max_i = None
    for i in range(B_cl.shape[0]):
        if np.all(B_cl[i] == wall): continue
        min_i = i; break
    for i in range(B_cl.shape[0] - 1, -1, -1):
        if np.all(B_cl[i] == wall): continue
        max_i = i + 1; break
    min_j = max_j = None
    for j in range(B_cl.shape[1]):
        if np.all(B_cl[:, j] == wall): continue
        min_j = j; break
    for j in range(B_cl.shape[1] - 1, -1, -1):
        if np.all(B_cl[:, j] == wall): continue
        max_j = j + 1; break
    B_cl_re = B_cl[min_i:max_i, min_j:max_j]

    if success_number == 1:
        os.makedirs(result_folder)
        with open(csvName, 'wt') as f:
            f.write('No,Area Cell,Area Reality,Scoring,Storage,Scoring,Perimeter,Peri-Efficiency,Scoring,Total Score\n')
        with open(found_patterns_name, 'wb') as f: 
            pickle.dump(dict(), f)
        with open(found_sub_patterns_name, 'wb') as f: 
            pickle.dump(dict(), f)

    with open(found_patterns_name, 'rb') as f: 
        found_patterns = pickle.load(f)
    min_i, max_i = A.shape[0], 0
    min_j, max_j = A.shape[0], 0
    for row in range(A.shape[0]):
        for col in range(A.shape[1]):
            if np.all(A[row, col] == wall): continue
            if row > max_i: max_i = row
            elif row < min_i: min_i = row
            if col > max_j: max_j = col
            elif col < min_j: min_j = col
    clean_a = A[min_i:max_i + 1, min_j:max_j + 1]
    duplicated_key = (AreaCell, Perimeter, Storage, clean_a.shape[0], clean_a.shape[1])
    if duplicated_key in found_patterns:
        for pattern in found_patterns[duplicated_key]:
            if np.all(pattern == clean_a): return
        found_patterns[duplicated_key].append(clean_a.copy())
    else: found_patterns[duplicated_key] = [clean_a.copy()]
    with open(found_patterns_name, 'wb') as f:
        pickle.dump(found_patterns, f)
    global newTotalScore, newCollections
    newTotalScore.append(TotalScore)
    newCollections.append(clean_a.copy())

    cell_half_size = cell_size // 2
    img_all = np.zeros((cell_size * n_ex, cell_size * n_ex + 300, 3), dtype=np.uint8)
    img_all[:] = 255
    img = img_all[:cell_size * n_ex, :cell_size * n_ex]
    clean_img = img[cell_half_size:-cell_half_size, cell_half_size:-cell_half_size]
    for row in range(A.shape[0]):
        cur_img_row = clean_img[row * cell_size:(row + 1) * cell_size]
        for col in range(A.shape[1]):
            if np.all(A[row, col] == wall): continue
            cur_img = cur_img_row[:, col * cell_size:(col + 1) * cell_size]
            if A[row, col, 0] > default_code:
                for c in range(3):
                    cur_img[:cell_half_size, :cell_half_size, c] = colors[A[row, col, 0]][c]
            if A[row, col, 1] > default_code:
                for c in range(3):
                    cur_img[:cell_half_size, cell_half_size:, c] = colors[A[row, col, 1]][c]
            if A[row, col, 2] > default_code:
                for c in range(3):
                    cur_img[cell_half_size:, :cell_half_size, c] = colors[A[row, col, 2]][c]
            if A[row, col, 3] > default_code:
                for c in range(3):
                    cur_img[cell_half_size:, cell_half_size:, c] = colors[A[row, col, 3]][c]
    img = img_all[:, cell_size * n_ex:]
    img[:] = 255
    x1 = 0; x2 = 200; y = cell_size; y_step = cell_size * 2 // 3
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
    baseName = result_folder + '/%06d' % success_number
    imgName = baseName + '.png'
    cv2.imwrite(imgName, img_all)
    pickleName = baseName + '.pkl'
    with open(pickleName, 'wb') as f:
        pickle.dump((A.copy(), B_cl_re.copy(), C.copy()), f)
    with open(csvName, 'at') as f:
        f.write(result_str)
    draw_dxf(B_cl_re, result_folder + '/%06d.dxf' % success_number)
    print('\tAt ' + get_time_str() + ', Got %d-th scheme (score: %.2f).' % (success_number, TotalScore))
    success_number += 1

def extend_generating(clean_a):
    new_seeds = []
    with open(found_sub_patterns_name, 'rb') as f: 
        found_sub_patterns = pickle.load(f)
    door_area = find_door_area(clean_a)
    hh, ww = clean_a.shape[:2]
    splits = [(0, hh // 2, 0, ww - 1), (hh // 2, hh - 1, 0, ww - 1), (0, hh - 1, 0, ww // 2), (0, hh - 1, ww // 2, ww - 1)]
    for ind, (i_min, i_max, j_min, j_max) in enumerate(splits):
        if in_door_checking(i_min, i_max, j_min, j_max, door_area): continue
        if i_min > door_area[0]: i_min = door_area[0]
        if i_max < door_area[1]: i_max = door_area[1]
        if j_min > door_area[2]: j_min = door_area[2]
        if j_max < door_area[3]: j_max = door_area[3]
        sub_a = clean_a[i_min:i_max + 1, j_min:j_max + 1]
        bed_area = find_bed_area(sub_a)
        if i_min == bed_area[0] and i_max == bed_area[1]:
            js = [j_min, j_max, bed_area[2], bed_area[3]]
            js.sort()
            for j_min, j_max in (js[:2], js[2:]):
                if not in_door_checking(i_min, i_max, j_min, j_max, door_area): break
        elif j_min == bed_area[2] and j_max == bed_area[3]:
            i_s = [i_min, i_max, bed_area[0], bed_area[1]]
            i_s.sort()
            for i_min, i_max in (i_s[:2], i_s[2:]):
                if not in_door_checking(i_min, i_max, j_min, j_max, door_area): break
        if i_max == hh - 1 and 0 in door_area[:2]: i_max = hh - 2
        if i_min == 0 and hh - 1 in door_area[:2]: i_min = 1
        if j_max == ww - 1 and 0 in door_area[2:]: j_max = ww - 2
        if j_min == 0 and ww - 1 in door_area[2:]: j_min = 1
        sub_a = clean_a[i_min:i_max + 1, j_min:j_max + 1].copy()
        sub_door_area = find_door_area(sub_a)
        for i in range(sub_a.shape[0]):
            for j in range(sub_a.shape[1]):
                beds_num = wall_num = 0
                for k in range(4):
                    if sub_a[i, j, k] in beds: sub_a[i, j, k] = passage; beds_num += 1
                    elif sub_a[i, j, k] == wall: wall_num += 1
                if beds_num > 0 and not cell_in_door_checking(i, j, sub_door_area):
                    if wall_num > 0:
                        if beds_num + wall_num == 4: sub_a[i, j, :] = default_code
                    else: sub_a[i, j, :] = default_code
        duplicated_key = (sub_a.shape[0], sub_a.shape[1])
        if duplicated_key in found_sub_patterns:
            duplicated = False
            for pattern in found_sub_patterns[duplicated_key]:
                if np.all(pattern == sub_a): duplicated = True; break
            if duplicated: continue
            found_sub_patterns[duplicated_key].append(sub_a.copy())
        else: found_sub_patterns[duplicated_key] = [sub_a.copy()]
        C = np.zeros((n, n), dtype=np.uint8)
        A = np.ones((n, n, 4), dtype=np.int) * default_code
        if door_area[0] == 0:
            if door_area[2] == 0: A[:sub_a.shape[0], :sub_a.shape[1]] = sub_a
            else: A[:sub_a.shape[0], -sub_a.shape[1]:] = sub_a
        else:
            if door_area[2] == 0: A[-sub_a.shape[0] - 1:-1, :sub_a.shape[1]] = sub_a
            else: A[-sub_a.shape[0] - 1:-1, -sub_a.shape[1]:] = sub_a
            A[-1, :, :2] = wall
        if window in A:
            blocks_num = len(np.where(A == window)[0])
            if blocks_num % 2 == 0: cur_windows = -1
            else: cur_windows = (blocks_num + 1) // 2
        else: cur_windows = 0
        A[0, :, :2] = wall
        A[:, 0, 0] = wall
        A[:, 0, 2] = wall
        A[:, -1, 1] = wall
        A[:, -1, 3] = wall
        A[-1, :, 2:] = wall
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if np.all(A[i, j] > default_code): C[i, j] = 1
        new_indices = []
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if C[i, j] > 0: new_indices += [indice for indice in extend(A, i, j, C) if indice not in new_indices]
        new_seeds.append((A.copy(), C.copy(), new_indices.copy(), cur_windows))
    with open(found_sub_patterns_name, 'wb') as f: 
        pickle.dump(found_sub_patterns, f)
    return new_seeds

def find_door_area(clean_a):
    for i in range(clean_a.shape[0]):
        for j in range(clean_a.shape[1]):
            if doors[0] not in clean_a[i, j]: continue
            if doors[1] not in clean_a[i, j]: continue
            if doors[2] not in clean_a[i, j]: continue
            if doors[3] not in clean_a[i, j]: continue
            return (i - 1, i + 1, j - 1, j + 1)

def in_door_checking(i_min, i_max, j_min, j_max, door_area):
    if i_min > door_area[0] or i_max < door_area[1]: return True
    if j_min > door_area[2] or j_max < door_area[3]: return True
    return False

def cell_in_door_checking(i, j, door_area):
    return door_area[0] <= i <= door_area[1] and door_area[2] <= j <= door_area[3]

def find_bed_area(clean_a):
    i_min, i_max = clean_a.shape[0], 0
    j_min, j_max = clean_a.shape[1], 0
    for i in range(clean_a.shape[0]):
        for j in range(clean_a.shape[1]):
            found = False
            for k in range(clean_a.shape[2]):
                if clean_a[i, j, k] in beds: found = True; break
            if found:
                if i > i_max: i_max = i
                elif i < i_min: i_min = i
                if j > j_max: j_max = j
                elif j < j_min: j_min = j
    return (i_min, i_max, j_min, j_max)

def get_time_str():
    cur = datetime.now()
    return '%04d-%02d-%02d %02d:%02d:%02d' % (cur.year, cur.month, cur.day, cur.hour, cur.minute, cur.second)

#print('At ' + get_time_str() + ', Started')
result_folder = 'results(' + str(datetime.now()).replace(':', '_')[:19] + ')'
success_number = 1
n_ex = n + 1
pickle_file_name, colors, min_code = get_blocks_collections()
default_code = min_code - 1
csvName = result_folder + '/all.csv'
pklName = result_folder + '/all.pkl'
found_patterns_name = result_folder + '/found_patterns.pickle'
found_sub_patterns_name = result_folder + '/found_sub_patterns.pickle'
new_seeds_name = result_folder + '/new_seeds.pickle'
cur_seed_name = result_folder + '/cur_seed.pickle'
robe_in_passages = [passage] + list(doors)
tried_num = 0
mother_number_prev = None
result_folder += '/dxf_png'

def Generate(extending, ScoreThreshold):
    with open(pickle_file_name, 'rb') as f:
        blocks, all_blocks_freq, \
        wall_index,\
        bed_oriented, bed_oriented_search, bed_putted_from_door, bed_putted_itself,\
        door_start_blocks, door_center_blocks, door_near_blocks,\
        window_putted_from_door, window_putted_itself, window_middle_blocks, \
        window_down_start, window_down_middle,\
        other_blocks = pickle.load(f)
    #
    cur_beds = 0
    beds_order_index = None
    if extending is not None:
        A, C, new_indices, cur_windows = extending
        extending_ = True
    else:
        extending_ = False
        # initialize
        C = np.zeros((n, n), dtype=np.uint8)
        A = np.ones((n, n, 4), dtype=np.int) * default_code
        A[0, :, :2] = wall
        A[:, 0, 0] = wall
        A[:, 0, 2] = wall
        A[:, -1, 1] = wall
        A[:, -1, 3] = wall
        A[-1, :, 2:] = wall
        #A[-1, 0, 3] = wall
        #A[-1, -1, 2] = wall
        Show(A)
        # starting
        door_start_block_keys = list(door_start_blocks.keys())
        start_key = np.random.choice(door_start_block_keys)
        i_j = np.random.randint(1, n - 2)
        if i_j == n - 2: i_j = n - 3
        if start_key == 'left':
            i, j = i_j, 0
            new_index = i, j + 1
        elif start_key == 'right':
            i, j = i_j, n - 1
            new_index = i, j - 1
        elif start_key == 'top':
            i, j = 0, i_j
            new_index = i + 1, j
        else:
            i, j = n - 1, i_j
            new_index = i - 1, j
        bl_indices = door_start_blocks[start_key]
        del door_start_block_keys, start_key
        cur_windows = 0
        # complete doors
        door_failed = False
        new_indices = []
        while True:
            bl_index = frequencyChoice(bl_indices, all_blocks_freq, blocks)
            if bl_index in door_near_blocks: door_near_blocks.remove(bl_index)
            if bl_index in bed_putted_from_door:
                cur_beds += 1
                if cur_beds == 1:
                    for beds_order_index in range(len(bed_oriented)):
                        if bl_index in bed_oriented[beds_order_index]: break
            elif bl_index in window_putted_from_door: cur_windows += 1
            setting(A, i, j, blocks[bl_index])
            Show(A, i, j)
            C[i, j] = 1
            new_indices_t = extend(A, i, j, C)
            for index in new_indices_t:
                if index not in new_indices: new_indices.append(index)
            Show(A, i, j)
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
                Show(A, i, j, (0, 0, 255), 2)
                door_failed = True; break
            new_indices.remove((i, j))
        if door_failed:
            if showing: cv2.waitKey(1000)
            return
    # next
    collaps = {}
    while True:
        if cur_beds > 0:
            bed_nears = []; other_nears = []
            for (i, j) in tuple(collaps.keys()):
                is_bed_near = False
                for k in range(4):
                    if A[i, j, k] in beds: is_bed_near = True; break
                if is_bed_near: bed_nears.append((i, j)); collaps.pop((i, j))
                else: other_nears.append((i, j))
            bed_failed = False
            while True:
                for (i, j) in new_indices:
                    is_bed_near = False
                    for k in range(4):
                        if A[i, j, k] in beds: is_bed_near = True; break
                    if is_bed_near:
                        if (i, j) not in bed_nears: bed_nears.append((i, j))
                    elif (i, j) not in other_nears: other_nears.append((i, j))
                if len(bed_nears) == 0: break
                i, j = bed_nears.pop(0)
                bl_indices = []
                for ind in bed_oriented_search[beds_order_index]:
                    block = blocks[ind]
                    same = True
                    for k in range(4):
                        if A[i, j, k] < min_code: continue
                        if A[i, j, k] != block[k]: same = False; break
                    if same: bl_indices.append(ind)
                if len(bl_indices) < 1:
                    Show(A, i, j, (0, 0, 255), 2)
                    bed_failed = True; break
                bl_index = frequencyChoice(bl_indices, all_blocks_freq, blocks)
                bed_oriented_search[beds_order_index].remove(bl_index)
                setting(A, i, j, blocks[bl_index])
                Show(A, i, j)
                C[i, j] = 1
                new_indices = extend(A, i, j, C); Show(A, i, j)
            if bed_failed: break
            collaps = {}
            new_indices = [(i, j) for (i, j) in other_nears if C[i, j] == 0]
            cur_beds = -1
        for index in new_indices:
            i, j = index
            A_have_pass = passage in A[i, j]
            if window in A[i, j]:
                indices = []
                if cur_windows > 0: indices += window_middle_blocks
                if cur_windows >= windows_min: indices += window_putted_itself
            else:
                indices = other_blocks.copy()
                if cur_beds == 0: indices += bed_putted_itself
                if cur_windows == 0: indices += window_putted_itself
                elif cur_windows > 0:
                    indices += window_middle_blocks
                    if cur_windows >= windows_min: indices += window_putted_itself
            bl_indices = []
            for ind in indices:
                block = blocks[ind]
                same = True
                bl_have_pass = bl_have_robe = False
                for k in range(4):
                    if A[i, j, k] < min_code:
                        if block[k] == robe: bl_have_robe = True
                        elif block[k] == passage: bl_have_pass = True
                    elif A[i, j, k] != block[k]: same = False; break
                if same:
                    if bl_have_pass:
                        same = A_have_pass
                    if same: bl_indices.append(ind)
                    elif len(bl_indices) == 0: bl_indices = [None]
            if len(bl_indices) > 0:
                if bl_indices[0] is None:
                    bl_indices.pop(0)
                if len(bl_indices) > 0: collaps[index] = bl_indices
                elif index in collaps: collaps.pop(index)
            else: collaps[index] = bl_indices
        if len(collaps) == 0:
            if max(cur_beds, cur_windows) < 0:
                A[A == default_code] = wall
                Complete(A, ScoreThreshold)
            break
        cur_keys = tuple(collaps.keys())
        if extending_:
            frequencies = [len(collaps[key]) for key in cur_keys]
            min_freq = min(frequencies)
            min_indices = [cur_keys[ind] for ind, freq in enumerate(frequencies) if freq == min_freq]
            # random choice
            next_index = min_indices[np.random.choice(range(len(min_indices)))]
        else:
            Entropies = [calcEntropy(collaps[key], all_blocks_freq, blocks) for key in cur_keys]
            min_entropy = min(Entropies)
            min_indices = [cur_keys[ind] for ind, entropy in enumerate(Entropies) if entropy == min_entropy]
            # random choice
            next_index = min_indices[np.random.choice(range(len(min_indices)))]
        bl_indices = collaps.pop(next_index)
        i, j = next_index
        if len(bl_indices) == 0:
            Show(A, i, j, (0, 0, 255), 2)
            break
        bl_index = frequencyChoice(bl_indices, all_blocks_freq, blocks)
        if bl_index in bed_putted_itself:
            cur_beds += 1
            if cur_beds == 1:
                for beds_order_index in range(len(bed_oriented)):
                    if bl_index in bed_oriented[beds_order_index]: break
                bed_oriented_search[beds_order_index].remove(bl_index)
        elif bl_index in window_putted_itself:
            if cur_windows > 1: cur_windows = -1
            else: cur_windows = 1
        elif bl_index in window_middle_blocks: cur_windows += 1
        # setting
        setting(A, i, j, blocks[bl_index])
        Show(A, i, j)
        C[i, j] = 1
        if bl_index == wall_index: new_indices = []
        else:
            new_indices = extend(A, i, j, C); Show(A, i, j)
    if showing: cv2.waitKey(1000)

# step 1
ScoreThreshold = TotalScoreThreshold
newTotalScore, newCollections = [], []
while True:
    # step 2
    print('At ' + get_time_str() + ', Generating from scratch...')
    while True:
        for _ in range(each_tries): Generate(None, ScoreThreshold)
        if len(newTotalScore) > 0: break
        # step 3
        print('At ' + get_time_str() + ', Retrying to generate from scratch...')
    print('At ' + get_time_str() + ', got %d scheme(s).' % len(newTotalScore))
    # extending
    while True:
        # step 4
        selections = max(1, int(len(newTotalScore) * (1 - removal / 100) + 0.3))
        np.random.shuffle(newTotalScore)
        newTotalScore = newTotalScore[:selections]
        newCollections = newCollections[:selections]
        ScoreThreshold0 = ScoreThreshold
        if len(newTotalScore) > 1:
            #max_score = max(newTotalScore)
            #min_score = min(newTotalScore)
            #ScoreThreshold = max_score * max_score_co + min_score * (1 - max_score_co)
            ScoreThreshold = sum(newTotalScore) / len(newTotalScore)
        else: ScoreThreshold = newTotalScore[0]
        print('ScoreThreshold is updated from %.2f to %.2f !' % (ScoreThreshold0, ScoreThreshold))
        # step 5
        totalSeeds = []
        for clean_a in newCollections:
            totalSeeds += extend_generating(clean_a)
        print('Got %d mother scheme(s).' % len(totalSeeds))
        newTotalScore.clear()
        newCollections.clear()
        for new_seed in totalSeeds:
            for _ in range(extend_tries): Generate(new_seed, ScoreThreshold)
        print('At ' + get_time_str() + ', got %d scheme(s).' % len(newTotalScore))
        # step 6
        if len(newTotalScore) == 0: break
    # step 7
    print()
