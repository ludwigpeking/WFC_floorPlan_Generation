import ezdxf
from commons import *
import numpy as np
import pickle, os

import matplotlib.pyplot as plt
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend


def add_robe_block(doc):
    width = cell_size
    height = cell_size * 0.82
    out_margin = (cell_size - height) / 2
    in_margin = out_margin / 2
    xs = [(in_margin + (width - in_margin * 2) / 5 * i) for i in range(1, 5)]
    y1 = out_margin + in_margin + (height - in_margin * 2) * 0.1
    y2 = out_margin + in_margin + (height - in_margin * 2) * 0.9
    delta_x = (width - in_margin * 2) / 5 / 6
    
    robe = doc.blocks.new(name='robe')
    add_rectangle_outline(robe, (0, out_margin), (width, out_margin + height), 0x00FFFFFF)
    add_rectangle_outline(robe, (in_margin, out_margin + in_margin), (width - in_margin, out_margin + height - in_margin), 0x00BEBEBE)
    robe.add_line((in_margin, cell_size / 2), (width - in_margin, cell_size / 2), dxfattribs={"true_color": 0x00BEBEBE})
    robe.add_line((xs[0] + delta_x, y1), (xs[0] - delta_x, y2), dxfattribs={"true_color": 0x00BEBEBE})
    robe.add_line((xs[1], y1), (xs[1], y2), dxfattribs={"true_color": 0x00BEBEBE})
    robe.add_line((xs[2] - delta_x, y1), (xs[2] + delta_x, y2), dxfattribs={"true_color": 0x00BEBEBE})
    robe.add_line((xs[3], y1), (xs[3], y2), dxfattribs={"true_color": 0x00BEBEBE})

def add_bed_block(doc):
    height = cell_size * 0.82
    out_margin = (cell_size - height) / 2
    
    bed = doc.blocks.new(name='bed')
    add_rectangle_outline(bed, (out_margin, 0), (4 * cell_size, 3 * cell_size), 0x00FFFFFF)
    add_rectangle_outline(bed, (out_margin * 2, out_margin), (cell_size - out_margin, 3 * cell_size / 2 - out_margin), 0x00BEBEBE)
    add_rectangle_outline(bed, (out_margin * 2, 3 * cell_size / 2 + out_margin), (cell_size - out_margin, 3 * cell_size - out_margin), 0x00BEBEBE)
    bed.add_line((out_margin * 2, 0), (out_margin * 2, 3 * cell_size), dxfattribs={"true_color": 0x00FFFFFF})

def add_door_block(doc):
    height = cell_size * 0.82
    out_margin = (cell_size - height) / 2
    door_length = 2 * cell_size - 4.5 * out_margin
    
    door = doc.blocks.new(name='door')
    add_rectangle_filled(door, (0, 2 * cell_size - out_margin), (out_margin, 2 * cell_size + out_margin), 256)
    add_rectangle_filled(door, (2 * cell_size, 2 * cell_size - out_margin), (2 * cell_size - out_margin, 2 * cell_size + out_margin), 256)
    door.add_polyline2d([(out_margin, 2 * cell_size + out_margin), (out_margin, 2 * cell_size - out_margin), (2 * out_margin, 2 * cell_size - out_margin), (2 * out_margin, 2 * cell_size + out_margin / 2), (2.5 * out_margin, 2 * cell_size + out_margin / 2), (2.5 * out_margin, 2 * cell_size + out_margin)], close=True, dxfattribs={"color": 256})
    door.add_polyline2d([(2 * cell_size - out_margin, 2 * cell_size + out_margin), (2 * cell_size - out_margin, 2 * cell_size - out_margin), (2 * cell_size - 2 * out_margin, 2 * cell_size - out_margin), (2 * cell_size - 2 * out_margin, 2 * cell_size + out_margin / 2), (2 * cell_size - 2.5 * out_margin, 2 * cell_size + out_margin / 2), (2 * cell_size - 2.5 * out_margin, 2 * cell_size + out_margin)], close=True, dxfattribs={"color": 256})
    add_rectangle_outline(door, (2 * cell_size - out_margin * 2, 2 * cell_size), (2 * cell_size - out_margin * 3, 2 * cell_size - door_length), 0x00FFFFFF)
    door.add_arc((2 * cell_size - out_margin * 2.5, 2 * cell_size), door_length, 180, 270, dxfattribs={"true_color": 0x00BEBEBE})
    door.add_line((2 * out_margin, 2 * cell_size + out_margin), (2 * cell_size - 2 * out_margin, 2 * cell_size + out_margin), dxfattribs={"true_color": 0x00BEBEBE})
    
    mirrdoor = doc.blocks.new(name='mirrdoor')
    add_rectangle_filled(mirrdoor, (0, 2 * cell_size - out_margin), (out_margin, 2 * cell_size + out_margin), 256)
    add_rectangle_filled(mirrdoor, (2 * cell_size, 2 * cell_size - out_margin), (2 * cell_size - out_margin, 2 * cell_size + out_margin), 256)
    mirrdoor.add_polyline2d([(out_margin, 2 * cell_size + out_margin), (out_margin, 2 * cell_size - out_margin), (2 * out_margin, 2 * cell_size - out_margin), (2 * out_margin, 2 * cell_size + out_margin / 2), (2.5 * out_margin, 2 * cell_size + out_margin / 2), (2.5 * out_margin, 2 * cell_size + out_margin)], close=True, dxfattribs={"color": 256})
    mirrdoor.add_polyline2d([(2 * cell_size - out_margin, 2 * cell_size + out_margin), (2 * cell_size - out_margin, 2 * cell_size - out_margin), (2 * cell_size - 2 * out_margin, 2 * cell_size - out_margin), (2 * cell_size - 2 * out_margin, 2 * cell_size + out_margin / 2), (2 * cell_size - 2.5 * out_margin, 2 * cell_size + out_margin / 2), (2 * cell_size - 2.5 * out_margin, 2 * cell_size + out_margin)], close=True, dxfattribs={"color": 256})
    add_rectangle_outline(mirrdoor, (out_margin * 2, 2 * cell_size), (out_margin * 3, 2 * cell_size - door_length), 0x00FFFFFF)
    mirrdoor.add_arc((out_margin * 2.5, 2 * cell_size), door_length, 270, 359.9, dxfattribs={"true_color": 0x00BEBEBE})
    mirrdoor.add_line((2 * out_margin, 2 * cell_size + out_margin), (2 * cell_size - 2 * out_margin, 2 * cell_size + out_margin), dxfattribs={"true_color": 0x00BEBEBE})

def set_modules():
    doc = ezdxf.new('R2010')
    add_robe_block(doc)
    add_bed_block(doc)
    add_door_block(doc)
    doc.saveas("input_data/models.dxf")

    
#######################################################
cell_size = 550

def add_rectangle_filled(msp, pt1, pt2, color):
    msp.add_solid([pt1, (pt1[0], pt2[1]), (pt2[0], pt1[1]), pt2], dxfattribs={"color": color})

def add_rectangle_outline(msp, pt1, pt2, color):
    msp.add_polyline2d([pt1, (pt1[0], pt2[1]), pt2, (pt2[0], pt1[1])], close=True, dxfattribs={"true_color": color})

def Cells():
    doc = ezdxf.readfile('Blocks.dxf')
    msp = doc.modelspace()
    for i in range(n):
        msp.add_line((0, i * cell_size), ((m - 1) * cell_size, i * cell_size), dxfattribs={"color": 3})
    for i in range(m):
        msp.add_line((i * cell_size, 0), (i * cell_size, (n - 1) * cell_size), dxfattribs={"color": 3})

def get_pos_clean(np_pos):
    return np_pos[0][0], np_pos[1][0]

def put_door(msp, pos_9390, pos_9392, pos_9090, C):
    y_9390, x_9390 = pos_9390
    y_9392, x_9392 = pos_9392
    y_9090, x_9090 = pos_9090
    x_min, y_min = min(x_9390, x_9392, x_9090), min(y_9390, y_9392, y_9090)
    left, top = x_min * cell_size, (n - 1 - y_min) * cell_size
    right, bottom = left + 2 * cell_size, top - 2 * cell_size
    if y_9390 == y_9392:
        if x_9390 > x_9392: rotation, mirror = 0, y_9090 < y_9390
        else: rotation, mirror = 180, y_9090 > y_9390
    elif y_9390 > y_9392: rotation, mirror = 90, x_9090 > x_9390
    else: rotation, mirror = -90, x_9090 < x_9390
    if y_9390 == y_9090: direction_index = 2 if C[y_9390 - 1, x_9390] == 1 else 4
    else: direction_index = 1 if C[y_9390, x_9390 - 1] == 1 else 3
    door_name = 'Door_Narrow_Mirror' if mirror else 'Door_Narrow'
    if rotation == 0: msp.add_blockref(door_name, (right, top))
    elif rotation == 180: msp.add_blockref(door_name, (left, bottom), dxfattribs={'rotation': 180})
    elif rotation == 90: msp.add_blockref(door_name, (right, bottom), dxfattribs={'rotation': -90})
    else: msp.add_blockref(door_name, (left, top), dxfattribs={'rotation': 90})
    return [(y_9390, x_9390, direction_index), (y_9090, x_9090, direction_index)]

def put_shower(msp, pos_2020, pos_2021, pos_2120):
    y_2020, x_2020 = pos_2020
    y_2021, x_2021 = pos_2021
    y_2120, x_2120 = pos_2120
    x_min, y_min = min(x_2020, x_2021, x_2120), min(y_2020, y_2021, y_2120)
    left, top = x_min * cell_size, (n - 1 - y_min) * cell_size
    right, bottom = left + 2 * cell_size, top - 2 * cell_size
    if x_2020 == x_2021:
        if y_2021 > y_2020: rotation, mirror = 0, x_2120 < x_2020
        else: rotation, mirror = 180, x_2120 > x_2020
    elif x_2020 > x_2021: rotation, mirror = 90, y_2120 < y_2020
    else: rotation, mirror = -90, y_2120 > y_2020
    block_name = 'Shower_Mirror' if mirror else 'Shower'
    if mirror:
        if rotation == 0: msp.add_blockref(block_name, (right, top))
        elif rotation == 90: msp.add_blockref(block_name, (right, bottom), dxfattribs={'rotation': -90})
        elif rotation == 180: msp.add_blockref(block_name, (left, bottom), dxfattribs={'rotation': 180})
        else: msp.add_blockref(block_name, (left, top), dxfattribs={'rotation': 90})
    else:
        if rotation == 0: msp.add_blockref(block_name, (left, top))
        elif rotation == 90: msp.add_blockref(block_name, (right, top), dxfattribs={'rotation': -90})
        elif rotation == 180: msp.add_blockref(block_name, (right, bottom), dxfattribs={'rotation': 180})
        else: msp.add_blockref(block_name, (left, bottom), dxfattribs={'rotation': 90})

def put_single_cell(msp, pos, block_name):
    y_min, x_min = pos
    left, top = x_min * cell_size, (n - 1 - y_min) * cell_size
    right, bottom = left + cell_size, top - cell_size
    msp.add_blockref(block_name, (left, bottom))

def put_lavatory(msp, ys_unique, xs_unique, lavatorys, B):
    block_name = 'Lavatory' if lavatorys == 1 else 'Lavatory_' + str(lavatorys)
    if len(ys_unique) == 1:
        left, top = min(xs_unique) * cell_size, (n - 1 - ys_unique[0]) * cell_size
        right, bottom = left + lavatorys * cell_size, top - cell_size
        rotation = 0 if B[ys_unique[0] - 1, xs_unique[0]] == 0 else 180
    else:
        left, top = xs_unique[0] * cell_size, (n - 1 - min(ys_unique)) * cell_size
        right, bottom = left + cell_size, top - lavatorys * cell_size
        rotation = 90 if B[ys_unique[0], xs_unique[0] + 1] == 0 else -90
    if rotation == 0: msp.add_blockref(block_name, (left, top))
    elif rotation == 180: msp.add_blockref(block_name, (right, bottom), dxfattribs={'rotation': 180})
    elif rotation == 90: msp.add_blockref(block_name, (right, top), dxfattribs={'rotation': -90})
    else: msp.add_blockref(block_name, (left, bottom), dxfattribs={'rotation': 90})

def put_toilet(msp, pos_1030, pos_1031, pos_1130):
    y_1030, x_1030 = pos_1030
    y_1031, x_1031 = pos_1031
    y_1130, x_1130 = pos_1130
    x_min, y_min = min(x_1030, x_1031, x_1130), min(y_1030, y_1031, y_1130)
    left, top = x_min * cell_size, (n - 1 - y_min) * cell_size
    right, bottom = left + 2 * cell_size, top - 2 * cell_size
    if x_1030 == x_1031: rotation = 0 if y_1030 < y_1031 else 180
    else: rotation = 90 if x_1030 > x_1031 else -90
    block_name = 'Toilet'
    if rotation == 0: msp.add_blockref(block_name, (left, top))
    elif rotation == 90: msp.add_blockref(block_name, (right, top), dxfattribs={'rotation': -90})
    elif rotation == 180: msp.add_blockref(block_name, (right, bottom), dxfattribs={'rotation': 180})
    else: msp.add_blockref(block_name, (left, bottom), dxfattribs={'rotation': 90})

def add_window(msp, A, B):
    ww1, ww2 = 50, 200
    specials_w = []
    rows, cols, indices = np.where(A == window)
    for ind in range(len(rows)):
        row, col, index = rows[ind], cols[ind], indices[ind]
        if A[row, col, index] != window: continue
        if index == 0: i, j = row - 1, col - 1
        elif index == 1: i, j = row - 1, col
        elif index == 2: i, j = row, col - 1
        else: i, j = row, col
        for i1, j1, k1 in ((i, j, 3), (i, j + 1, 2), (i + 1, j, 1), (i + 1, j + 1, 0)):
            try: A[i1, j1, k1] = wall
            except: pass
        i1, j1 = i + 1, j + 1
        found = False
        if j1 + 1 < B.shape[1]:
            if B[i1, j1 + 1] == 1: specials_w.append((i1, j1 + 1, 1)); found = True
        if not found:
            if B[i1, j1 - 1] == 1: specials_w.append((i1, j1 - 1, 3))
            elif B[i1 - 1, j1] == 1: specials_w.append((i1 - 1, j1, 4))
            else: specials_w.append((i1 + 1, j1, 2))
    # left-side
    indices = [item[:2] for item in specials_w if item[-1] == 1]
    if len(indices) > 0:
        ys = [index[0] for index in indices]
        pt1 = indices[0][1] * cell_size, (n - 1 - min(ys)) * cell_size - ww1
        pt2 = pt1[0] - ww2, pt1[1] - len(ys) * cell_size + 2 * ww1
        pt3 = pt1[0] - ww1, pt1[1]
        pt4 = pt1[0] - ww1, pt2[1]
        add_rectangle_outline(msp, pt1, pt2, 0x00FFFFFF)
        msp.add_line(pt3, pt4, dxfattribs={"true_color": 0x00FFFFFF})
    # down-side
    indices = [item[:2] for item in specials_w if item[-1] == 4]
    if len(indices) > 0:
        xs = [index[1] for index in indices]
        pt1 = min(xs) * cell_size + ww1, (n - 2 - indices[0][0]) * cell_size
        pt2 = pt1[0] + len(xs) * cell_size - 2 * ww1, pt1[1] - ww2
        pt3 = pt1[0], pt1[1] - ww1
        pt4 = pt2[0], pt1[1] - ww1
        add_rectangle_outline(msp, pt1, pt2, 0x00FFFFFF)
        msp.add_line(pt3, pt4, dxfattribs={"true_color": 0x00FFFFFF})
    # right-side
    indices = [item[:2] for item in specials_w if item[-1] == 3]
    if len(indices) > 0:
        ys = [index[0] for index in indices]
        pt1 = (1 + indices[0][1]) * cell_size, (n - 1 - min(ys)) * cell_size - ww1
        pt2 = pt1[0] + ww2, pt1[1] - len(ys) * cell_size + 2 * ww1
        pt3 = pt1[0] + ww1, pt1[1]
        pt4 = pt1[0] + ww1, pt2[1]
        add_rectangle_outline(msp, pt1, pt2, 0x00FFFFFF)
        msp.add_line(pt3, pt4, dxfattribs={"true_color": 0x00FFFFFF})
    return specials_w

def draw_dxf(A, B, C, ys_unique, xs_unique, lavatorys, dxf_name):
    doc = ezdxf.readfile('Blocks.dxf')
    msp = doc.blocks.new(name='AllBlock')
    pos_9390 = get_pos_clean(np.where(C == 9390))
    pos_9392 = get_pos_clean(np.where(C == 9392))
    pos_9090 = get_pos_clean(np.where(C == 9090))
    specials = put_door(msp, pos_9390, pos_9392, pos_9090, C)
    specials += add_window(msp, A, B)
    #print(specials)
    pos_2020 = get_pos_clean(np.where(C == 2020))
    pos_2021 = get_pos_clean(np.where(C == 2021))
    pos_2120 = get_pos_clean(np.where(C == 2120))
    put_shower(msp, pos_2020, pos_2021, pos_2120)
    put_single_cell(msp, get_pos_clean(np.where(C == washmach)), 'Washmach')
    put_single_cell(msp, get_pos_clean(np.where(C == shaft)), 'Shaft')
    put_lavatory(msp, ys_unique, xs_unique, lavatorys, B)
    pos_1030 = get_pos_clean(np.where(C == 1030))
    pos_1031 = get_pos_clean(np.where(C == 1031))
    pos_1130 = get_pos_clean(np.where(C == 1130))
    put_toilet(msp, pos_1030, pos_1031, pos_1130)
    # perimeter
    out_margin = 50
    for i in range(1, B.shape[0] - 1):
        for j in range(1, B.shape[1] - 1):
            if B[i, j] == 0: continue
            left, top = j * cell_size, (n - 1 - i) * cell_size
            right, bottom = left + cell_size, top - cell_size
            if B[i, j - 1] == 0 and (i, j, 1) not in specials:
                x1, x2 = left - out_margin, left + out_margin
                y1, y2 = top + out_margin, bottom - out_margin
                add_rectangle_filled(msp, (x1, y1), (x2, y2), 256)
            if B[i, j + 1] == 0 and (i, j, 3) not in specials:
                x1, x2 = right - out_margin, right + out_margin
                y1, y2 = top + out_margin, bottom - out_margin
                add_rectangle_filled(msp, (x1, y1), (x2, y2), 256)
            if B[i - 1, j] == 0 and (i, j, 2) not in specials:
                x1, x2 = left - out_margin, right + out_margin
                y1, y2 = top + out_margin, top - out_margin
                add_rectangle_filled(msp, (x1, y1), (x2, y2), 256)
            if B[i + 1, j] == 0 and (i, j, 4) not in specials:
                x1, x2 = left - out_margin, right + out_margin
                y1, y2 = bottom + out_margin, bottom - out_margin
                add_rectangle_filled(msp, (x1, y1), (x2, y2), 256)
    msp = doc.modelspace()
    msp.add_blockref('AllBlock', (0, 0))
    doc.saveas(dxf_name)
    dxf2img(dxf_name, os.path.splitext(dxf_name)[0] + '-1.png')

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
    plt.close()
