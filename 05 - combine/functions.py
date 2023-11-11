import pickle, os
from glob import glob
import pandas as pd
import numpy as np
import shutil

import ezdxf
import matplotlib.pyplot as plt
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
from ezdxf.addons import Importer

from common import *

cell_size = 550

def save_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def choose_general(src_path, dst_path, num):
    if not os.path.isdir(dst_path):
        os.makedirs(dst_path)
    else:
        for file in os.listdir(dst_path):
            os.remove(os.path.join(dst_path, file))
    csv_path = glob(os.path.join(src_path, '*', 'all.csv'))[-1]
    data_path = os.path.join(os.path.dirname(csv_path), 'dxf_png')
    df = pd.read_csv(csv_path)
    with open(csv_path, 'rt') as f:
        cc = f.read().split('\n')
    head, body = cc[0], cc[1:]
    sorting = np.argsort(list(df['Total Score']))[::-1]
    cur_num = 0
    total_patterns = {}
    for i, index in enumerate(sorting):
        src_base_name = os.path.join(data_path, '%06d' % df['No'][index])
        src_pkl_name = src_base_name + '.pkl'
        src_dxf_name = src_base_name + '.dxf'
        src_png_name = src_base_name + '.png'
        A, B, C = load_pickle(src_pkl_name)
        cur_pattern = B.copy()
        cur_pattern[cur_pattern > 0] = 1
        cur_key = cur_pattern.shape
        if cur_key in total_patterns:
            duplicated = False
            for pp in total_patterns[cur_key]:
                if np.all(pp == cur_pattern): duplicated = True; break
            if duplicated: continue
            total_patterns[cur_key].append(cur_pattern)
        else: total_patterns[cur_key] = [cur_pattern]
        dst_base_name = os.path.join(dst_path, '%04d' % cur_num)
        dst_pkl_name = dst_base_name + '.pkl'
        dst_dxf_name = dst_base_name + '.dxf'
        dst_png_name = dst_base_name + '.png'
        dst_csv_name = dst_base_name + '.csv'
        with open(dst_csv_name, 'wt') as f:
            f.write(head + '\n' + body[index])
        shutil.copy(src_pkl_name, dst_pkl_name)
        shutil.copy(src_dxf_name, dst_dxf_name)
        doc = ezdxf.readfile(dst_dxf_name)
        msp = doc.modelspace()
        remove_all_entities(msp)
        doc.saveas(dst_dxf_name)
        shutil.copy(src_png_name, dst_png_name)
        cur_num += 1
        if cur_num == num: break

def dxf2img(dxf_name, img_name, dpi=300):
    doc = ezdxf.readfile(dxf_name)
    msp = doc.modelspace()
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ctx = RenderContext(doc)
    ctx.set_current_layout(msp)
    out = MatplotlibBackend(ax)
    Frontend(ctx, out).draw_layout(msp, finalize=True)
    fig.savefig(img_name, dpi=dpi)
    plt.close(fig)

def cells(map):
    for i in range(-5, 6):
        msp.add_line((-5 * cell_size, i * cell_size), (5 * cell_size, i * cell_size), dxfattribs={"color": 3})
        msp.add_line((i * cell_size, -5 * cell_size), (i * cell_size, 5 * cell_size), dxfattribs={"color": 3})

def get_pos_clean(np_pos):
    return np_pos[0][0], np_pos[1][0]

def dxftype(entity):
    notHandled = []
    if entity.dxftype() in notHandled:
        return None
    else:
        return entity.dxftype()

def remove_all_entities(msp):
    group = msp.groupby(key=dxftype)
    for entityType, entities in group.items():
        for e in entities:
            msp.delete_entity(e)

def door_pos_index_01(B):
    pos_9390 = get_pos_clean(np.where(B == 9390))
    pos_9090 = get_pos_clean(np.where(B == 9090))
    return pos_9390[0], min(pos_9390[1], pos_9090[1])

def door_pos_index_23(B):
    pos_9390 = get_pos_clean(np.where(B == 9390))
    pos_9090 = get_pos_clean(np.where(B == 9090))
    return min(pos_9390[0], pos_9090[0]), pos_9390[1]

def index_0_block(B, C):
    if C is None:
        pos_9390 = get_pos_clean(np.where(B == 9390))
        pos_9090 = get_pos_clean(np.where(B == 9090))
        xx = min(pos_9390[1], pos_9090[1]) * cell_size
        yy = (B.shape[0] - 2 - pos_9390[0]) * cell_size
    else:
        pos_9390 = get_pos_clean(np.where(C == 9390))
        pos_9090 = get_pos_clean(np.where(C == 9090))
        xx = min(pos_9390[1], pos_9090[1]) * cell_size
        yy = (C.shape[0] - 3 - pos_9390[0]) * cell_size
    B0 = B.copy()
    B0_ref = B0[:, ::-1]
    no_ref = np.all((B0 > 0) == (B0_ref > 0))
    if no_ref:
        pos_9390 = get_pos_clean(np.where(B == 9390))
        pos_9090 = get_pos_clean(np.where(B == 9090))
        left_x, right_x = (pos_9390[1], pos_9090[1]) if pos_9390[1] < pos_9090[1] else (pos_9090[1], pos_9390[1])
        no_ref = (left_x == (B.shape[1] - 1 - right_x))
    if no_ref:
        B0s = (B0,)
        B0s_door_index = (door_pos_index_01(B0s[0]),)
        B0s_pos = ((-xx, -yy),)
        B0s_dict = ({},)
    else:
        B0s = (B0, B0[:, ::-1])
        B0s_door_index = (door_pos_index_01(B0s[0]), door_pos_index_01(B0s[1]))
        B0s_pos = ((-xx, -yy), (xx + 2 * cell_size, -yy))
        B0s_dict = ({}, {'xscale': -1})
    # msp.add_blockref('AllBlock', (-xx, -yy))
    # msp.add_blockref('AllBlock', (xx + 2 * cell_size, -yy), dxfattribs={'xscale': -1})
    B1 = B[::-1]
    if no_ref:
        B1s = (B1[:, ::-1],)
        B1s_door_index = (door_pos_index_01(B1s[0]),)
        B1s_pos = ((xx + 2 * cell_size, yy),)
        B1s_dict = ({'xscale': -1, 'yscale': -1},)
    else:
        B1s = (B1, B1[:, ::-1])
        B1s_door_index = (door_pos_index_01(B1s[0]), door_pos_index_01(B1s[1]))
        B1s_pos = ((-xx, yy), (xx + 2 * cell_size, yy))
        B1s_dict = ({'yscale': -1}, {'xscale': -1, 'yscale': -1})
    # msp.add_blockref('AllBlock', (0, 0), dxfattribs={'rotation': 180})
    # msp.add_blockref('AllBlock', (0, 0), dxfattribs={'xscale': -1, 'rotation': 180})
    B3 = B.T[::-1]
    if no_ref:
        B3s = (B3,)
        B3s_door_index = (door_pos_index_23(B3s[0]),)
        B3s_pos = ((yy, -(xx + 2 * cell_size)),)
        B3s_dict = ({'rotation': 90},)
    else:
        B3s = (B3, B3[::-1])
        B3s_door_index = (door_pos_index_23(B3s[0]), door_pos_index_23(B3s[1]))
        B3s_pos = ((yy, -(xx + 2 * cell_size)), (yy, xx))
        B3s_dict = ({'rotation': 90}, {'xscale': -1, 'rotation': 90})
    # msp.add_blockref('AllBlock', (yy, -(xx + 2 * cell_size)), dxfattribs={'rotation': 90})
    # msp.add_blockref('AllBlock', (yy, xx), dxfattribs={'xscale': -1, 'rotation': 90})
    B2 = B3[::-1, ::-1]
    if no_ref:
        B2s = (B2,)
        B2s_door_index = (door_pos_index_23(B2s[0]),)
        B2s_pos = ((-yy, xx),)
        B2s_dict = ({'rotation': 270},)
    else:
        B2s = (B2, B3[::-1])
        B2s_door_index = (door_pos_index_23(B2s[0]), door_pos_index_23(B2s[1]))
        B2s_pos = ((-yy, xx), (-yy, -xx - 2 * cell_size))
        B2s_dict = ({'rotation': 270}, {'xscale': -1, 'rotation': 270})
    # msp.add_blockref('AllBlock', (-yy, -(xx + 2 * cell_size)), dxfattribs={'rotation': 270})
    # msp.add_blockref('AllBlock', (-yy, xx), dxfattribs={'xscale': -1, 'rotation': 270})
    return (B0s, B0s_door_index, B0s_pos, B0s_dict), (B1s, B1s_door_index, B1s_pos, B1s_dict), (B2s, B2s_door_index, B2s_pos, B2s_dict), (B3s, B3s_door_index, B3s_pos, B3s_dict)

def index_1_block(B, C):
    if C is None:
        pos_9390 = get_pos_clean(np.where(B == 9390))
        pos_9090 = get_pos_clean(np.where(B == 9090))
        xx = min(pos_9390[1], pos_9090[1]) * cell_size
        yy = (B.shape[0] - 2 - pos_9390[0]) * cell_size
    else:
        pos_9390 = get_pos_clean(np.where(C == 9390))
        pos_9090 = get_pos_clean(np.where(C == 9090))
        xx = min(pos_9390[1], pos_9090[1]) * cell_size
        yy = (C.shape[0] - 3 - pos_9390[0]) * cell_size
    B1 = B.copy()
    B1_ref = B1[:, ::-1]
    no_ref = np.all((B1 > 0) == (B1_ref > 0))
    if no_ref:
        pos_9390 = get_pos_clean(np.where(B == 9390))
        pos_9090 = get_pos_clean(np.where(B == 9090))
        left_x, right_x = (pos_9390[1], pos_9090[1]) if pos_9390[1] < pos_9090[1] else (pos_9090[1], pos_9390[1])
        no_ref = (left_x == (B.shape[1] - 1 - right_x))
    if no_ref:
        B1s = (B1,)
        B1s_door_index = (door_pos_index_01(B1s[0]),)
        B1s_pos = ((-xx, -yy - cell_size),)
        B1s_dict = ({},)
    else:
        B1s = (B1, B1_ref)
        B1s_door_index = (door_pos_index_01(B1s[0]), door_pos_index_01(B1s[1]))
        B1s_pos = ((-xx, -yy - cell_size), (xx + 2 * cell_size, -yy - cell_size))
        B1s_dict = ({}, {'xscale': -1})
    # msp.add_blockref('AllBlock', (-xx, -yy - cell_size))
    # msp.add_blockref('AllBlock', (xx + 2 * cell_size, -yy - cell_size), dxfattribs={'xscale': -1})
    B0 = B[::-1, ::-1]
    if no_ref:
        B0s = (B0,)
        B0s_door_index = (door_pos_index_01(B0s[0]),)
        B0s_pos = ((xx + 2 * cell_size, yy + cell_size),)
        B0s_dict = ({'rotation': 180})
    else:
        B0s = (B0, B0[:, ::-1])
        B0s_door_index = (door_pos_index_01(B0s[0]), door_pos_index_01(B0s[1]))
        B0s_pos = ((xx + 2 * cell_size, yy + cell_size), (-xx, yy + cell_size))
        B0s_dict = ({'rotation': 180}, {'xscale': -1, 'rotation': 180})
    # msp.add_blockref('AllBlock', (xx + 2 * cell_size, yy + cell_size), dxfattribs={'rotation': 180})
    # msp.add_blockref('AllBlock', (-xx, yy + cell_size), dxfattribs={'xscale': -1, 'rotation': 180})
    B2 = B.T[::-1]
    if no_ref:
        B2s = (B2,)
        B2s_door_index = (door_pos_index_23(B2s[0]),)
        B2s_pos = ((yy + cell_size, -xx - 2 * cell_size),)
        B2s_dict = ({'rotation': 90},)
    else:
        B2s = (B2, B2[::-1])
        B2s_door_index = (door_pos_index_23(B2s[0]), door_pos_index_23(B2s[1]))
        B2s_pos = ((yy + cell_size, -xx - 2 * cell_size), (yy + cell_size, xx))
        B2s_dict = ({'rotation': 90}, {'xscale': -1, 'rotation': 90})
    # msp.add_blockref('AllBlock', (yy + cell_size, -xx - 2 * cell_size), dxfattribs={'rotation': 90})
    # msp.add_blockref('AllBlock', (yy + cell_size, xx), dxfattribs={'xscale': -1, 'rotation': 90})
    B3 = B2[::-1, ::-1]
    if no_ref:
        B3s = (B3,)
        B3s_door_index = (door_pos_index_23(B3s[0]),)
        B3s_pos = ((-yy - cell_size, xx),)
        B3s_dict = ({'rotation': 270},)
    else:
        B3s = (B3, B3[::-1])
        B3s_door_index = (door_pos_index_23(B3s[0]), door_pos_index_23(B3s[1]))
        B3s_pos = ((-yy - cell_size, xx), (-yy - cell_size, -xx - 2 * cell_size))
        B3s_dict = ({'rotation': 270}, {'xscale': -1, 'rotation': 270})
    # msp.add_blockref('AllBlock', (-yy - cell_size, xx), dxfattribs={'rotation': 270})
    # msp.add_blockref('AllBlock', (-yy - cell_size, -xx - 2 * cell_size), dxfattribs={'xscale': -1, 'rotation': 270})
    return (B0s, B0s_door_index, B0s_pos, B0s_dict), (B1s, B1s_door_index, B1s_pos, B1s_dict), (B2s, B2s_door_index, B2s_pos, B2s_dict), (B3s, B3s_door_index, B3s_pos, B3s_dict)

def index_3_block(B, C):
    if C is None:
        pos_9390 = get_pos_clean(np.where(B == 9390))
        pos_9090 = get_pos_clean(np.where(B == 9090))
        xx = pos_9390[1] * cell_size
        yy = (B.shape[0] - 2 - max(pos_9390[0], pos_9090[0])) * cell_size
    else:
        pos_9390 = get_pos_clean(np.where(C == 9390))
        pos_9090 = get_pos_clean(np.where(C == 9090))
        xx = pos_9390[1] * cell_size
        yy = (C.shape[0] - 3 - max(pos_9390[0], pos_9090[0])) * cell_size
    B3 = B.copy()
    B3_ref = B3[::-1]
    no_ref = np.all((B3 > 0) == (B3_ref > 0))
    if no_ref:
        pos_9390 = get_pos_clean(np.where(B == 9390))
        pos_9090 = get_pos_clean(np.where(B == 9090))
        upper, lower = (pos_9390[0], pos_9090[0]) if pos_9390[0] < pos_9090[0] else (pos_9090[0], pos_9390[0])
        no_ref = (upper == (B.shape[0] - 1 - lower))
    if no_ref:
        B3s = (B3,)
        B3s_door_index = (door_pos_index_23(B3s[0]),)
        B3s_pos = ((-xx - cell_size, -yy - 2 * cell_size),)
        B3s_dict = ({},)
    else:
        B3s = (B3, B3[::-1])
        B3s_door_index = (door_pos_index_23(B3s[0]), door_pos_index_23(B3s[1]))
        B3s_pos = ((-xx - cell_size, -yy - 2 * cell_size), (-xx - cell_size, yy))
        B3s_dict = ({}, {'yscale': -1})
    # msp.add_blockref('AllBlock', (-xx - cell_size, -yy - 2 * cell_size))
    # msp.add_blockref('AllBlock', (-xx - cell_size, yy), dxfattribs={'yscale': -1})
    B2 = B3[::-1, ::-1]
    if no_ref:
        B2s = (B2,)
        B2s_door_index = (door_pos_index_23(B2s[0]),)
        B2s_pos = ((xx + cell_size, yy),)
        B2s_dict = ({'rotation': 180},)
    else:
        B2s = (B2, B3[::-1])
        B2s_door_index = (door_pos_index_23(B2s[0]), door_pos_index_23(B2s[1]))
        B2s_pos = ((xx + cell_size, yy), (xx + cell_size, -yy - 2 * cell_size))
        B2s_dict = ({'rotation': 180}, {'yscale': -1, 'rotation': 180})
    # msp.add_blockref('AllBlock', (xx + cell_size, yy), dxfattribs={'rotation': 180})
    # msp.add_blockref('AllBlock', (xx + cell_size, -yy - 2 * cell_size), dxfattribs={'yscale': -1, 'rotation': 180})
    B0 = B.T[:, ::-1]
    if no_ref:
        B0s = (B0,)
        B0s_door_index = (door_pos_index_01(B0s[0]),)
        B0s_pos = ((-yy, xx + cell_size),)
        B0s_dict = ({'rotation': 270},)
    else:
        B0s = (B0, B0[:, ::-1])
        B0s_door_index = (door_pos_index_01(B0s[0]), door_pos_index_01(B0s[1]))
        B0s_pos = ((-yy, xx + cell_size), (yy + 2 * cell_size, xx + cell_size))
        B0s_dict = ({'rotation': 270}, {'yscale': -1, 'rotation': 270})
    # msp.add_blockref('AllBlock', (-yy, xx + cell_size), dxfattribs={'rotation': 270})
    # msp.add_blockref('AllBlock', (yy + 2 * cell_size, xx + cell_size), dxfattribs={'yscale': -1, 'rotation': 270})
    B1 = B0[::-1, ::-1]
    if no_ref:
        B1s = (B1,)
        B1s_door_index = (door_pos_index_01(B1s[0]),)
        B1s_pos = ((yy + 2 * cell_size, -xx - cell_size),)
        B1s_dict = ({'rotation': 90},)
    else:
        B1s = (B1, B1[:, ::-1])
        B1s_door_index = (door_pos_index_01(B1s[0]), door_pos_index_01(B1s[1]))
        B1s_pos = ((yy + 2 * cell_size, -xx - cell_size), (-yy, -xx - cell_size))
        B1s_dict = ({'rotation': 90}, {'yscale': -1, 'rotation': 90})
    # msp.add_blockref('AllBlock', (yy + 2 * cell_size, -xx - cell_size), dxfattribs={'rotation': 90})
    # msp.add_blockref('AllBlock', (-yy, -xx - cell_size), dxfattribs={'yscale': -1, 'rotation': 90})
    return (B0s, B0s_door_index, B0s_pos, B0s_dict), (B1s, B1s_door_index, B1s_pos, B1s_dict), (B2s, B2s_door_index, B2s_pos, B2s_dict), (B3s, B3s_door_index, B3s_pos, B3s_dict)

def analysis_blocks(path):
    pkls = glob(os.path.join(path, '*.pkl'))
    for pkl in pkls:
        A, B, C = load_pickle(pkl)
        pos_9390 = get_pos_clean(np.where(C == 9390))
        pos_9090 = get_pos_clean(np.where(C == 9090))
        dst_func = None
        if pos_9090[0] == pos_9390[0]:
            if C[pos_9390[0] + 1, pos_9390[1]] == 1: dst_func = index_0_block
            else: dst_func = index_1_block
        elif C[pos_9390[0], pos_9390[1] + 1] == 1: dst_func = index_3_block
        else: dst_func = index_2_block
        if path == room_dst_path: C = None
        save_pickle(dst_func(B, C), pkl + '.pickle')

def check_pkl_pkl(pkl):
    dxf = pkl.replace('.pkl.pickle', '.dxf')
    doc = ezdxf.readfile(dxf)
    msp = doc.modelspace()
    door_info = load_pickle(pkl)
    for ii, ind_info in enumerate(door_info):
        for i in range(len(ind_info[0])):
            remove_all_entities(msp)
            for j in range(-10, 11):
                msp.add_line((-10 * cell_size, j * cell_size), (10 * cell_size, j * cell_size), dxfattribs={"color": 3})
                msp.add_line((j * cell_size, -10 * cell_size), (j * cell_size, 10 * cell_size), dxfattribs={"color": 3})
            pos, dxfattribs = ind_info[2][i], ind_info[3][i]
            msp.add_blockref('AllBlock', pos, dxfattribs=dxfattribs)
            msp.add_circle((0, 0), radius=50, dxfattribs={"color": 3})
            doc.saveas('11.dxf')
            dxf2img('11.dxf', f'11-{ii}-{i}.png')

def basic_working():
    choose_general(room_src_path, room_dst_path, rooms)
    choose_general(toilet_src_path, toilet_dst_path, toilets)
    choose_general(kitchen_src_path, kitchen_dst_path, kitchens)
    analysis_blocks(room_dst_path)
    analysis_blocks(toilet_dst_path)
    analysis_blocks(kitchen_dst_path)

def not_fit_put(mother_B, poses, B1, door_index1):
    pos1, pos2 = poses
    di, dj = pos1[0] - door_index1[0], pos1[1] - door_index1[1]
    #print(di, dj)
    for i in range(B1.shape[0]):
        for j in range(B1.shape[1]):
            if B1[i, j] == 0: continue
            new_i, new_j = i + di, j + dj
            if new_i == pos1[0] and new_j == pos1[1]: continue
            if new_i == pos2[0] and new_j == pos2[1]: continue
            if mother_B[new_i, new_j] != 0: return True#print(mother_B[new_i, new_j], new_i, new_j, i, j); return True
            mother_B[new_i, new_j] = B1[i, j]
    return False

def not_fit_put_op(mother_B, poses, B1, door_index1):
    pos1, pos2 = poses
    di, dj = pos1[0] - door_index1[0], pos1[1] - door_index1[1]
    #print(di, dj)
    for i in range(B1.shape[0]):
        for j in range(B1.shape[1]):
            if B1[i, j] == 0: continue
            new_i, new_j = i + di, j + dj
            if mother_B[new_i, new_j] != 0: 
                if mother_B[new_i, new_j] in access: 
                    if new_i == pos1[0] and new_j == pos1[1]: continue
                    if new_i == pos2[0] and new_j == pos2[1]: continue
                return True#print(mother_B[new_i, new_j], new_i, new_j, i, j); return True
            mother_B[new_i, new_j] = B1[i, j]
    return False

def check_matrix(C):
    C[C == window] = 0
    #mask = np.zeros([C.shape[0] + 2, C.shape[1] + 2], np.uint8)
    #cv2.floodFill(C, mask, (0, 0), -1)
    Perimeter = cellNumber = 0
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if C[i, j] != 0:
                cellNumber += 1
                if C[i - 1, j] == 0: Perimeter += 1
                if C[i + 1, j] == 0: Perimeter += 1
                if C[i, j + 1] == 0: Perimeter += 1
                if C[i, j - 1] == 0: Perimeter += 1
    score = -0.1 * (cellNumber * 0.3025 - 63) ** 2 + 10 + 200 * cellNumber * 0.3025 / (Perimeter ** 2)
    return cellNumber, Perimeter, score

def check_matrix_old(C):
    C[C == window] = 0
    C[0, 0] = -1
    n, m = C.shape[0] - 1, C.shape[1] - 1
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
                    new_indices.append((new_i, new_j))
                    C[new_i, new_j] = -1
        old_indices = new_indices.copy()
    Perimeter = cellNumber = 0
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if C[i, j] != -1:
                cellNumber += 1
                if C[i - 1, j] == -1: Perimeter += 1
                if C[i + 1, j] == -1: Perimeter += 1
                if C[i, j + 1] == -1: Perimeter += 1
                if C[i, j - 1] == -1: Perimeter += 1
    score = -0.1 * (cellNumber * 0.3025 - 63) ** 2 + 10 + 200 * cellNumber * 0.3025 / (Perimeter ** 2)
    return cellNumber, Perimeter, score

def combine_kernel(pkl, mother_B, cells, door_pos, i, j, k, l, n):
	pos1, pos2, pos3, pos4 = door_pos[i - 1], door_pos[j - 1], door_pos[k - 1], door_pos[l - 1]
	cell1, cell2, cell3, cell4 = cells[2 * i - 2], cells[2 * j - 2], cells[2 * k - 2], cells[2 * l - 2]
	room_pkls = glob(os.path.join(room_dst_path, '*.pickle'))
	toilet_pkls = glob(os.path.join(toilet_dst_path, '*.pickle'))
	kitchen_pkls = glob(os.path.join(kitchen_dst_path, '*.pickle'))
	for pkl1 in room_pkls:
		Bs1, door_indices1, door_poses1, bl_dicts1 = load_pickle(pkl1)[pos1[0][-1]]
		for B1, door_index1, door_pos1, bl_dict1 in zip(Bs1, door_indices1, door_poses1, bl_dicts1):
			mother_B1 = mother_B.copy()
			if not_fit_put(mother_B1, pos1, B1, door_index1): continue
			for pkl2 in room_pkls:
				Bs2, door_indices2, door_poses2, bl_dicts2 = load_pickle(pkl2)[pos2[0][-1]]
				for B2, door_index2, door_pos2, bl_dict2 in zip(Bs2, door_indices2, door_poses2, bl_dicts2):
					mother_B2 = mother_B1.copy()
					if not_fit_put(mother_B2, pos2, B2, door_index2): continue
					for pkl3 in toilet_pkls:
						Bs3, door_indices3, door_poses3, bl_dicts3 = load_pickle(pkl3)[pos3[0][-1]]
						for B3, door_index3, door_pos3, bl_dict3 in zip(Bs3, door_indices3, door_poses3, bl_dicts3):
							mother_B3 = mother_B2.copy()
							if not_fit_put(mother_B3, pos3, B3, door_index3): continue
							for pkl4 in kitchen_pkls:
								Bs4, door_indices4, door_poses4, bl_dicts4 = load_pickle(pkl4)[pos4[0][-1]]
								for B4, door_index4, door_pos4, bl_dict4 in zip(Bs4, door_indices4, door_poses4, bl_dicts4):
									mother_B4 = mother_B3.copy()
									if not_fit_put(mother_B4, pos4, B4, door_index4): continue
									cellNumber, Perimeter, score = check_matrix(mother_B4.copy())
									if score < 10: continue
									draw_data((pkl, pkl1, pkl2, pkl3, pkl4), (cell1, cell2, cell3, cell4), (cellNumber, Perimeter, score), 
                                               ((door_pos1, bl_dict1), (door_pos2, bl_dict2), (door_pos3, bl_dict3), (door_pos4, bl_dict4)), n, mother_B4)
									print('\t', cellNumber, Perimeter, score)
					for pkl3 in kitchen_pkls:
						Bs3, door_indices3, door_poses3, bl_dicts3 = load_pickle(pkl3)[pos3[0][-1]]
						for B3, door_index3, door_pos3, bl_dict3 in zip(Bs3, door_indices3, door_poses3, bl_dicts3):
							mother_B3 = mother_B2.copy()
							if not_fit_put(mother_B3, pos3, B3, door_index3): continue
							for pkl4 in toilet_pkls:
								Bs4, door_indices4, door_poses4, bl_dicts4 = load_pickle(pkl4)[pos4[0][-1]]
								for B4, door_index4, door_pos4, bl_dict4 in zip(Bs4, door_indices4, door_poses4, bl_dicts4):
									mother_B4 = mother_B3.copy()
									if not_fit_put(mother_B4, pos4, B4, door_index4): continue
									cellNumber, Perimeter, score = check_matrix(mother_B4.copy())
									if score < 10: continue
									draw_data((pkl, pkl1, pkl2, pkl3, pkl4), (cell1, cell2, cell3, cell4), (cellNumber, Perimeter, score), 
                                               ((door_pos1, bl_dict1), (door_pos2, bl_dict2), (door_pos3, bl_dict3), (door_pos4, bl_dict4)), n, mother_B4)
									print('\t', cellNumber, Perimeter, score)

def draw_data(pkls, cells, score_info, bl_info, n, mother_B4):
	pkl, pkl1, pkl2, pkl3, pkl4 = pkls
	cellNumber, Perimeter, score = score_info
	global cur_number, folder_name
	filename = os.path.join(folder_name, '%06d(%d-%d-%.2f).pkl' % (cur_number, cellNumber, Perimeter, score))
	save_pickle((pkls, cells, score_info, bl_info, n, mother_B4.copy()), filename)
	cur_number += 1

def draw_final(pkl_file):
	pkls, cells, score_info, bl_info, n, mother_B4 = load_pickle(pkl_file)
	pkl, pkl1, pkl2, pkl3, pkl4 = pkls
	cell1, cell2, cell3, cell4 = cells
	cellNumber, Perimeter, score = score_info
	((door_pos1, bl_dict1), (door_pos2, bl_dict2), (door_pos3, bl_dict3), (door_pos4, bl_dict4)) = bl_info
	filename = pkl_file.replace('.pkl', '.dxf')

	dxf = pkl.replace('.pkl', '.dxf')
	dxf1 = pkl1.replace('.pkl.pickle', '.dxf')
	dxf2 = pkl2.replace('.pkl.pickle', '.dxf')
	dxf3 = pkl3.replace('.pkl.pickle', '.dxf')
	dxf4 = pkl4.replace('.pkl.pickle', '.dxf')
	doc = ezdxf.readfile(dxf)
	msp = doc.modelspace()
	remove_all_entities(msp)
	doc.saveas(filename)
	#
	doc = ezdxf.readfile(filename)
	doc_sub = ezdxf.readfile(dxf1)
	importer = Importer(doc_sub, doc)
	bk_name1 = importer.import_block('AllBlock')
	importer.finalize()
	doc_sub = ezdxf.readfile(dxf2)
	importer = Importer(doc_sub, doc)
	bk_name2 = importer.import_block('AllBlock')
	importer.finalize()
	doc_sub = ezdxf.readfile(dxf3)
	importer = Importer(doc_sub, doc)
	bk_name3 = importer.import_block('AllBlock')
	importer.finalize()
	doc_sub = ezdxf.readfile(dxf4)
	importer = Importer(doc_sub, doc)
	bk_name4 = importer.import_block('AllBlock')
	importer.finalize()
	msp = doc.modelspace()
	msp.add_blockref('AllBlock', (0, 0))
	msp.add_blockref(bk_name1, (door_pos1[0] + cell1[1] * cell_size, door_pos1[1] + (n - cell1[0]) * cell_size), dxfattribs=bl_dict1)
	msp.add_blockref(bk_name2, (door_pos2[0] + cell2[1] * cell_size, door_pos2[1] + (n - cell2[0]) * cell_size), dxfattribs=bl_dict2)
	msp.add_blockref(bk_name3, (door_pos3[0] + cell3[1] * cell_size, door_pos3[1] + (n - cell3[0]) * cell_size), dxfattribs=bl_dict3)
	msp.add_blockref(bk_name4, (door_pos4[0] + cell4[1] * cell_size, door_pos4[1] + (n - cell4[0]) * cell_size), dxfattribs=bl_dict4)
	doc.saveas(filename)
	dxf2img(filename, filename.replace('.dxf', '.png'))

def combine(src_path):
    csv_path = glob(os.path.join(src_path, '*', 'all.csv'))[-1]
    data_path = os.path.join(os.path.dirname(csv_path), 'dxf_png')
    df = pd.read_csv(csv_path)
    with open(csv_path, 'rt') as f:
        cc = f.read().split('\n')
    head, body = cc[0], cc[1:]
    groups = (
        (1, 2, 3, 4),
        (1, 3, 2, 4),
        (1, 4, 2, 3),
        (2, 3, 1, 4),
        (2, 4, 1, 3),
        (3, 4, 1, 2),
    )
    pkls = glob(os.path.join(data_path, '*.pkl'))
    for pkl in pkls:
        print('-' * 50)
        print(pkl)
        print('-' * 50)
        global cur_number, folder_name
        cur_number = 1
        folder_name = os.path.join(hall_dst_path, os.path.basename(pkl).replace('.pkl', ''))
        os.makedirs(folder_name)
        B, access_outline_cells = load_pickle(pkl)
        door_pos = []
        for cell in access_outline_cells:
            if cell[-1] == 0: door_pos.append((cell[0] - 1, cell[1], 0))
            elif cell[-1] == 1: door_pos.append((cell[0] + 1, cell[1], 1))
            elif cell[-1] == 2: door_pos.append((cell[0], cell[1] + 1, 2))
            else: door_pos.append((cell[0], cell[1] - 1, 3))
        total_B = np.zeros((50, 50), dtype=np.int)
        margin_t = (total_B.shape[0] - B.shape[0]) // 2
        margin_l = (total_B.shape[1] - B.shape[1]) // 2
        total_B[margin_t:margin_t + B.shape[0], margin_l:margin_l + B.shape[1]] = B.copy()
        door_pos = [(i + margin_t, j + margin_l, k) for i, j, k in door_pos]
        door_pos = [door_pos[i:i + 2] for i in range(0, 8, 2)]
        for i, j, k, l in groups:
            print(i, j, k, l)
            combine_kernel(pkl, total_B.copy(), access_outline_cells, door_pos, i, j, k, l, B.shape[0])
            print('\n')


if __name__ == '__main__':
	cur_number = 1
	folder_name = ''
	#basic_working()
	combine(hall_src_path)
