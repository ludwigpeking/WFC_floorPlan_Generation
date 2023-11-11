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
    for k in range(4): src_array[row, col, k] = val_array[k]

def extend(src_array, row, col, flag_array):
    add_indices = []
    if col > 0:
        if flag_array[row, col - 1] == 0:
            for ind in (1, 3):
                if src_array[row, col - 1, ind] < min_code: src_array[row, col - 1, ind] = src_array[row, col, ind - 1]
            add_indices.append((row, col - 1))
    if col < A.shape[1] - 1:
        if flag_array[row, col + 1] == 0:
            for ind in (0, 2):
                if src_array[row, col + 1, ind] < min_code: src_array[row, col + 1, ind] = src_array[row, col, ind + 1]
            add_indices.append((row, col + 1))
    if row > 0:
        if flag_array[row - 1, col] == 0:
            for ind in (2, 3):
                if src_array[row - 1, col, ind] < min_code: src_array[row - 1, col, ind] = src_array[row, col, ind - 2]
            add_indices.append((row - 1, col))
    if row < A.shape[0] - 1:
        if flag_array[row + 1, col] == 0:
            for ind in (0, 1):
                if src_array[row + 1, col, ind] < min_code: src_array[row + 1, col, ind] = src_array[row, col, ind + 2]
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
    img = np.zeros((cell_size * n_ex, cell_size * m_ex, 3), dtype=np.uint8)
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

def calcEntropy(bl_indices):
    if len(bl_indices) == 1: return 0
    data = []
    for ind in bl_indices: data += [ind] * all_blocks_freq[blocks[ind]]
    pd_series = pd.Series(data)
    counts = pd_series.value_counts()
    entropy = stats.entropy(counts)
    return entropy

def frequencyChoice(bl_indices):
    if len(bl_indices) == 1: return bl_indices[0]
    nums = []
    for ind in bl_indices: nums += [ind] * all_blocks_freq[blocks[ind]]
    return np.random.choice(nums)

def line_array(array):
    array.sort()
    for i in range(len(array) - 1):
        if array[i + 1] > array[i] + 1: return False
    return True

def line_window(array):
    array.sort()
    for i in range(1, len(array) - 1, 2):
        if array[i + 1] != array[i]: return False
    return True

def CheckAndComplete():
    global success_number, result_folder

    # checking window
    if window not in A: return
    # boundary
    for item in np.unique(A[0, :, :2]):
        if item not in (wall, window): return
    for item in np.unique(A[-1, :, 2:]):
        if item not in (wall, window): return
    for item in np.unique(A[:, 0, ::2]):
        if item not in (wall, window): return
    for item in np.unique(A[:, -1, 1::2]):
        if item not in (wall, window): return
    AA = A.copy()
    A[A == window] = wall
    C = np.zeros((n + 1, m + 1), dtype=np.int)
    B = C[1:-1, 1:-1]
    for i in range(n - 1):
        for j in range(m - 1):
            collections = A[i, j, 3], A[i, j + 1, 2], A[i + 1, j, 1], A[i + 1, j + 1, 0]
            if min(collections) < max(collections): return
            B[i, j] = A[i, j, 3]
    A[AA == window] = window
    # cut window side
    wall_block_sum = 4 * wall
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j].sum() == wall_block_sum: continue
            can_be_wall = True
            for item in A[i, j]:
                if item not in (wall, window): can_be_wall = False
            if can_be_wall: A[i, j, :] = wall
    # checking unique
    if len(np.where(C == shaft)[0]) != 1: return
    if counter not in C: return
    # shape of pins
    pins = {}
    for item in (stove, frig, sink):
        ys, xs = np.where(C == item)
        if len(ys) != 2: return
        ys_unique = np.unique(ys)
        xs_unique = np.unique(xs)
        if len(ys_unique) == 1:
            if len(xs_unique) > 1:
                if not line_array(xs_unique): return
            pins[item] = ('y', ys_unique[0], xs_unique)
        elif len(xs_unique) > 1: return
        elif not line_array(ys_unique): return 
        else: pins[item] = ('x', xs_unique[0], ys_unique)
    # window
    ys, xs, _ = np.where(A == window)
    ys_unique = np.unique(ys)
    xs_unique = np.unique(xs)
    if len(ys_unique) == 1:
        if len(xs_unique) > 1:
            if not line_window(xs): return
        pins[window] = ('y', ys_unique[0], (xs[1], xs[-1]))
    elif len(xs_unique) > 1: return
    elif not line_window(ys): return 
    else: pins[window] = ('x', xs_unique[0], (ys[1], ys[-1]))
    # find Perimeter
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
                if new_j > m: continue
                if C[new_i, new_j] == 0:
                    B[new_i, new_j] = 0
                    new_indices.append((new_i, new_j))
                    C[new_i, new_j] = 1
        old_indices = new_indices.copy()
    Perimeter = AreaCell = 0
    for i in range(1, n):
        for j in range(1, m):
            if B[i, j] > 0:
                AreaCell += 1
                if B[i - 1, j] == 0: Perimeter += 1
                if B[i + 1, j] == 0: Perimeter += 1
                if B[i, j + 1] == 0: Perimeter += 1
                if B[i, j - 1] == 0: Perimeter += 1
    # scoring
    AreaReality = C2 = AreaCell * 0.55 ** 2
    score1 = C2 ** 4 * (-0.0017) + 0.0778 * C2 ** 3 - 1.1921 * C2 ** 2 + 6.048 * C2 + 0.3002
    Counters = J2 = len(np.where(C == counter)[0])
    score2 = -0.0689 * J2 ** 2 + 1.0218 * J2 + 0.0741
    windows_len = O4 = len(np.where(A == window)[0]) // 2
    score3 = 0.5816 * np.log(O4) + 1.0004
    PeriEfficiency = AreaCell / Perimeter ** 2
    score4 = PeriEfficiency * 60
    TotalScore = score1 + score2 + score3 + score4
    if TotalScore < TotalScoreThreshold: return
    # reduce
    B_cl = np.zeros((n - 1, m - 1), dtype=np.int)
    for i in range(n - 1):
        for j in range(m - 1):
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
    B_cl = B_cl[min_i:max_i, min_j:max_j]
    j_9090 = np.where(B_cl == 9090)[1][0]
    j_9390 = np.where(B_cl == 9390)[1][0]
    if j_9090 < j_9390:
        B_cl_re = B_cl[:, ::-1]
        reflected = True
    else:
        B_cl_re = B_cl.copy()
        reflected = True
    duplicated_key = AreaCell, Perimeter, windows_len, B_cl_re.shape[0], B_cl_re.shape[1]
    found_patterns = load_pickle(pklName)
    if duplicated_key in found_patterns:
        for pattern in found_patterns[duplicated_key]:
            if np.all(pattern == B_cl_re): return
        found_patterns[duplicated_key].append(B_cl_re.copy())
    else: found_patterns[duplicated_key] = [B_cl_re.copy()]
    
    cell_half_size = cell_size // 2
    # draw & detect duplicated
    x1 = 0; x2 = 200; y = cell_size; y_step = cell_size * 2 // 3
    img_all = np.zeros((max(y * 2 + y_step * 14, cell_size * n_ex), cell_size * m_ex + 300, 3), dtype=np.uint8)
    img_all[:] = 255
    img = img_all[:cell_size * n_ex, :cell_size * m_ex]
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
    img = img_all[:, -300:]
    img[:] = 255
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
    result_str += ',%.2f' % score1
    cv2.putText(img, 'Scoring:', (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    cv2.putText(img, '%.2f' % score1, (x2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    y += y_step
    result_str += ',%d' % Counters
    cv2.putText(img, 'Counter Area:', (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    cv2.putText(img, '%d' % Counters, (x2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    y += y_step
    result_str += ',%.2f' % score2
    cv2.putText(img, 'Scoring:', (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    cv2.putText(img, '%.2f' % score2, (x2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    y += y_step
    result_str += ',%d' % windows_len
    cv2.putText(img, 'Window Width:', (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    cv2.putText(img, '%d' % windows_len, (x2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    y += y_step
    result_str += ',%.2f' % score3
    cv2.putText(img, 'Scoring:', (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    cv2.putText(img, '%.2f' % score3, (x2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    y += y_step
    result_str += ',%d' % Perimeter
    cv2.putText(img, 'Perimeter:', (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    cv2.putText(img, '%d' % Perimeter, (x2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    y += y_step
    result_str += ',%.4f' % PeriEfficiency
    cv2.putText(img, 'Peri-Efficiency:', (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    cv2.putText(img, '%.4f' % PeriEfficiency, (x2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    y += y_step
    result_str += ',%.2f' % score4
    cv2.putText(img, 'Scoring:', (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    cv2.putText(img, '%.2f' % score4, (x2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    y += y_step * 2
    result_str += ',%.2f' % TotalScore
    cv2.putText(img, 'Total Score:', (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
    cv2.putText(img, '%.2f' % TotalScore, (x2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
    result_str += '\n'
    if success_number == 1:
        os.makedirs(result_folder)
        with open(csvName, 'wt') as f:
            f.write('No,Area Cell,Area Reality,Scoring,Counter Area,Scoring,Window Width,Scoring,Perimeter,Peri-Efficiency,Scoring,Total Score\n')
    baseName = result_folder + '/%06d' % success_number
    imgName = baseName + '.png'
    cv2.imwrite(imgName, img_all)
    pickleName = baseName + '.pkl'
    #B_cl_r = B_cl[:, ::-1]
    #B_cls_0 = B_cl.copy(), B_cl_r.copy()
    #B_cls_1 = B_cl[::-1], B_cl_r[::-1]
    #B_cls_2 = B_cl.T[::-1][::-1, ::-1], B_cl_r.T[::-1][::-1, ::-1]
    #B_cls_3 = B_cl.T[::-1], B_cl_r.T[::-1]
    save_pickle((A.copy(), B_cl.copy(), C.copy()), pickleName)
    save_pickle(found_patterns, pklName)
    with open(csvName, 'at') as f:
        f.write(result_str)
    dxf_name = baseName + '.dxf'
    draw_dxf(A, B, C, pins, windows_len, dxf_name)
    success_number += 1

print('Started at ', datetime.now())
result_folder = 'results(' + str(datetime.now()).replace(':', '_')[:19] + ')'
success_number = 1
m_ex, n_ex = m + 1, n + 1
pickle_file_name, colors, min_code = get_blocks_collections()
default_code = min_code - 1
csvName = result_folder + '/all.csv'
pklName = result_folder + '/all.pkl'
result_folder += '/dxf_png'

for iii in range(loops):
    with open(pickle_file_name, 'rb') as f:
        blocks, all_blocks_freq, \
        wall_index,\
        door_start_blocks, door_center_blocks, door_near_blocks,\
        other_blocks = pickle.load(f)

    # initialize
    C = np.zeros((n, m), dtype=np.uint8)
    A = np.ones((n, m, 4), dtype=np.int) * default_code
    for ind in (0, 1, 2): A[0, 0, ind] = wall
    for ind in (0, 1, 3): A[0, -1, ind] = wall
    for ind in (0, 2, 3): A[-1, 0, ind] = wall
    for ind in (1, 2, 3): A[-1, -1, ind] = wall
    Show(A)

    # starting
    door_start_block_keys = list(door_start_blocks.keys())
    start_key = np.random.choice(door_start_block_keys)
    if start_key == 'left':
        i, j = np.random.randint(1, n - 2), 0
        if i == n - 2: i = n - 3
        new_index = i, j + 1
    elif start_key == 'right':
        i, j = np.random.randint(1, n - 2), m - 1
        if i == n - 2: i = m - 3
        new_index = i, j - 1
    elif start_key == 'top':
        i, j = 0, np.random.randint(1, m - 2)
        if j == m - 2: j = m - 3
        new_index = i + 1, j
    else:
        i, j = n - 1, np.random.randint(1, m - 2)
        if j == m - 2: j = m - 3
        new_index = i - 1, j
    bl_indices = door_start_blocks[start_key]
    del door_start_block_keys, start_key

    # complete doors
    door_failed = False
    new_indices = []
    while True:
        bl_index = frequencyChoice(bl_indices)
        if bl_index in door_near_blocks: door_near_blocks.remove(bl_index)
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
        continue

    # next
    collaps = {}
    while True:
        for index in new_indices:
            i, j = index
            A_have_pass = passage in A[i, j]
            bl_indices = []
            for ind in other_blocks:
                block = blocks[ind]
                same = True
                bl_have_new_pass = False
                for k in range(4):
                    if A[i, j, k] < min_code:
                        if block[k] == passage: bl_have_new_pass = True
                    elif A[i, j, k] != block[k]: same = False; break
                if same:
                    if bl_have_new_pass: same = A_have_pass
                    if same: bl_indices.append(ind)
                    elif len(bl_indices) == 0: bl_indices = [None]
            if len(bl_indices) > 0:
                if bl_indices[0] is None:
                    bl_indices.pop(0)
                if len(bl_indices) > 0: collaps[index] = bl_indices
                elif index in collaps: collaps.pop(index)
            else: collaps[index] = bl_indices
        if len(collaps) == 0:
            A[A == default_code] = wall
            CheckAndComplete()
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
            Show(A, i, j, (0, 0, 255), 2)
            break
        bl_index = frequencyChoice(bl_indices)
        # setting
        setting(A, i, j, blocks[bl_index])
        Show(A, i, j)
        C[i, j] = 1
        if bl_index == wall_index: new_indices = []
        else:
            new_indices = extend(A, i, j, C); Show(A, i, j)
    if showing: cv2.waitKey(1000)
print('Finished at ', datetime.now())
