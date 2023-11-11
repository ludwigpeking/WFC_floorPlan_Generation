import ezdxf
from commons import *
import numpy as np
import pickle, os

import matplotlib.pyplot as plt
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend


cell_size = 550
out_margin = 50

def add_rectangle_filled(msp, pt1, pt2, color):
    msp.add_solid([pt1, (pt1[0], pt2[1]), (pt2[0], pt1[1]), pt2], dxfattribs={"color": color})

def add_rectangle_outline(msp, pt1, pt2, color):
    msp.add_polyline2d([pt1, (pt1[0], pt2[1]), pt2, (pt2[0], pt1[1])], close=True, dxfattribs={"true_color": color})

def Cells(msp):
    for i in range(-5, 6):
        msp.add_line((-5 * cell_size, i * cell_size), (5 * cell_size, i * cell_size), dxfattribs={"color": 3})
        msp.add_line((i * cell_size, -5 * cell_size), (i * cell_size, 5 * cell_size), dxfattribs={"color": 3})

def create_bed_block(msp):
    doc = ezdxf.readfile('Blocks.dxf')
    msp = doc.blocks.new(name='Bed')
    add_rectangle_outline(msp, (0, -50), (3 * cell_size, -4 * cell_size), 0x00FFFFFF)
    msp.add_line((0, -100), (3 * cell_size, -100), dxfattribs={"true_color": 0x00BEBEBE})
    add_rectangle_outline(msp, (50, -100), (775, -500), 0x00FFFFFF)
    add_rectangle_outline(msp, (875, -100), (1600, -500), 0x00FFFFFF)
    doc.saveas('Blocks.dxf')
    # usage
    '''
    doc = ezdxf.readfile('Blocks.dxf')
    msp = doc.modelspace()
    Cells(msp)
    # msp.add_blockref('Bed', (0, 0))
    # msp.add_blockref('Bed', (0, 0), dxfattribs={'rotation': 90})
    # msp.add_blockref('Bed', (0, 0), dxfattribs={'rotation': 180})
    msp.add_blockref('Bed', (0, 0), dxfattribs={'rotation': 270})
    doc.saveas('11.dxf')
    dxf2img('11.dxf', '11-270.png')
    '''

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

def draw_dxf_old(A, B, C, dxf_name):
    doc = ezdxf.readfile('input_data/models.dxf')
    msp = doc.modelspace()
    # params
    height = cell_size * 0.82
    out_margin = (cell_size - height) / 2
    # windows
    for i in range(A.shape[0] - 1, -1, -1):
        if window in A[i]: break
    for x1 in range(A.shape[1]):
        if window in A[i, x1]: break
    for x2 in range(A.shape[1] - 1, -1, -1):
        if window in A[i, x2]: break
    x1 += 1
    x2 += 1
    specials = []
    for j in range(x1, x2): specials.append((i, j, 4))
    add_window(msp, B.shape[0] - 1 - i, x1, x2)
    # door
    pos = np.where(C == 9390)
    i1, j1 = pos[0][0], pos[1][0]
    pos = np.where(C == 9090)
    i2, j2 = pos[0][0], pos[1][0]
    pos = np.where(C == 9392)
    i3, j3 = pos[0][0], pos[1][0]
    i_min, j_min = min(i1, i2, i3), min(j1, j2, j3)
    i_max, j_max = max(i1, i2, i3), max(j1, j2, j3)
    left, bottom = j_min * cell_size, (B.shape[0] - 1 - i_max) * cell_size
    right, top = left + 2 * cell_size, bottom + 2 * cell_size
    direction_index = key = rotation = None
    key = 'door'
    if i1 == i2:
        direction_index = 2 if C[i1 - 1, j1] == 1 else 4
        if i1 < i3:
            rotation = 0
            key = 'mirrdoor' if j1 < j2 else 'door'
        else:
            rotation = 180
            key = 'mirrdoor' if j1 > j2 else 'door'
    else: # j1 = j2
        direction_index = 1 if C[i_min, j1 - 1] == 1 else 3
        if j1 < j3:
            rotation = 90
            key = 'door' if i1 < i2 else 'mirrdoor'
        else:
            rotation = -90
            key = 'door' if i1 > i2 else 'mirrdoor'
    specials.append((i1, j1, direction_index)); specials.append((i2, j2, direction_index))
    #print(i1, j1, i2, j2, i3, j3, key, direction_index, rotation)
    if rotation == 0: msp.add_blockref(key, (left, bottom))
    elif rotation == 90: msp.add_blockref(key, (right, bottom), dxfattribs={'rotation': 90})
    elif rotation == -90: msp.add_blockref(key, (left, top), dxfattribs={'rotation': -90})
    else: msp.add_blockref(key, (right, top), dxfattribs={'rotation': 180})
    for i in range(1, B.shape[0] - 1):
        for j in range(1, B.shape[1] - 1):
            if B[i, j] == 0: continue
            left, bottom = j * cell_size, (B.shape[0] - 1 - i) * cell_size
            right, top = left + cell_size, bottom + cell_size
            if B[i, j - 1] == 0 and (i, j, 1) not in specials: add_rectangle_filled(msp, (left - out_margin, top + out_margin), (left + out_margin, bottom - out_margin), 256)
            if B[i - 1, j] == 0 and (i, j, 2) not in specials: add_rectangle_filled(msp, (left, top - out_margin), (right, top + out_margin), 256)
            if B[i, j + 1] == 0 and (i, j, 3) not in specials: add_rectangle_filled(msp, (right - out_margin, top + out_margin), (right + out_margin, bottom - out_margin), 256)
            if B[i + 1, j] == 0 and (i, j, 4) not in specials: add_rectangle_filled(msp, (left, bottom - out_margin), (right, bottom + out_margin), 256)
            if C[i, j] == robe:
                if C[i - 1, j] == 1:
                    if C[i + 1, j] in robe_in_passages: msp.add_blockref('robe', (left, bottom))
                    elif C[i, j + 1] in robe_in_passages: msp.add_blockref('robe', (right, bottom), dxfattribs={'rotation': 90})
                    elif C[i, j - 1] in robe_in_passages: msp.add_blockref('robe', (left, top), dxfattribs={'rotation': -90})
                elif C[i, j - 1] == 1:
                    if C[i, j + 1] in robe_in_passages: msp.add_blockref('robe', (right, bottom), dxfattribs={'rotation': 90})
                    elif C[i - 1, j] in robe_in_passages: msp.add_blockref('robe', (right, top), dxfattribs={'rotation': 180})
                elif C[i, j + 1] == 1:
                    if C[i, j - 1] in robe_in_passages: msp.add_blockref('robe', (left, top), dxfattribs={'rotation': -90})
                    elif C[i - 1, j] in robe_in_passages: msp.add_blockref('robe', (right, top), dxfattribs={'rotation': 180})
                elif C[i + 1, j] == 1: msp.add_blockref('robe', (right, top), dxfattribs={'rotation': 180})
                elif C[i + 1, j] in robe_in_passages: msp.add_blockref('robe', (left, bottom))
                elif C[i, j + 1] in robe_in_passages: msp.add_blockref('robe', (right, bottom), dxfattribs={'rotation': 90})
                elif C[i, j - 1] in robe_in_passages: msp.add_blockref('robe', (left, top), dxfattribs={'rotation': -90})
                elif C[i - 1, j] in robe_in_passages: msp.add_blockref('robe', (right, top), dxfattribs={'rotation': 180})
    pos = np.where(C == 8000)
    i1, j1 = pos[0][0], pos[1][0]
    pos = np.where(C == 8500)
    i2, j2 = pos[0][0], pos[1][0]
    pos = np.where(C == 8006)
    i3, j3 = pos[0][0], pos[1][0]
    if i1 == i2:
        if i1 > i3: msp.add_blockref('bed', (max(j1, j2) * cell_size + cell_size, (B.shape[0] - 1 - i1) * cell_size), dxfattribs={'rotation': 90})
        else: msp.add_blockref('bed', (min(j1, j2) * cell_size, (B.shape[0] - i1) * cell_size), dxfattribs={'rotation': -90})
    else: # j1 = j2
        if j3 > j1: msp.add_blockref('bed', (j1 * cell_size, (B.shape[0] - 1 - max(i1, i2)) * cell_size))
        else: msp.add_blockref('bed', (j1 * cell_size + cell_size, (B.shape[0] - min(i1, i2)) * cell_size), dxfattribs={'rotation': 180})
    # save
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

robe_in = [passage] + list(doors)

def add_robe(msp, B_cl, no_access_outline_cells):
    nn = B_cl.shape[0]
    yy, xx = np.where(B_cl == robe)
    for i, j in zip(yy, xx):
        left, bottom = j * cell_size, (nn - 1 - i) * cell_size - cell_size
        right, top = left + cell_size, bottom + cell_size
        wall_cons = []
        wall_cons_pure = []
        for k in range(4):
            if (i, j, k) in no_access_outline_cells: 
                cur_val = 1
                wall_cons_pure.append(1)
            else:
                cur_val = 1
                wall_cons_pure.append(0)
                if k == 0:
                    if B_cl[i - 1, j] in robe_in: cur_val = 0
                elif k == 1:
                    if B_cl[i + 1, j] in robe_in: cur_val = 0
                elif k == 2:
                    if B_cl[i, j + 1] in robe_in: cur_val = 0
                elif B_cl[i, j - 1] in robe_in: cur_val = 0
            wall_cons.append(cur_val)
        sum_walls = sum(wall_cons)
        if sum_walls == 3:
            if wall_cons[0] == 0:
                msp.add_blockref('robe', (left, top), dxfattribs={'rotation': 270})
            elif wall_cons[1] == 0:
                msp.add_blockref('robe', (right, bottom), dxfattribs={'rotation': 90})
            elif wall_cons[2] == 0:
                msp.add_blockref('robe', (right, top), dxfattribs={'rotation': 180})
            elif wall_cons[3] == 0:
                msp.add_blockref('robe', (left, bottom))
        elif sum(wall_cons_pure) == 1:
            if wall_cons_pure[1] == 1:
                msp.add_blockref('robe', (left, top), dxfattribs={'rotation': 270})
            elif wall_cons_pure[0] == 1:
                msp.add_blockref('robe', (right, bottom), dxfattribs={'rotation': 90})
            elif wall_cons_pure[3] == 1:
                msp.add_blockref('robe', (right, top), dxfattribs={'rotation': 180})
            elif wall_cons_pure[2] == 1:
                msp.add_blockref('robe', (left, bottom))
        elif sum_walls == 2:
            if wall_cons[3] == 0:
                msp.add_blockref('robe', (left, bottom))
            else:
                msp.add_blockref('robe', (left, top), dxfattribs={'rotation': 270})
        else:
            msp.add_blockref('robe', (left, bottom))

def get_pos_clean(np_pos):
    return np_pos[0][0], np_pos[1][0]

def add_window(msp, i, x1, x2):
    height = cell_size * 0.82
    out_margin = (cell_size - height) / 2
    add_rectangle_outline(msp, (x1 * cell_size + out_margin, i * cell_size + out_margin), (x2 * cell_size - out_margin, i * cell_size), 0x00FFFFFF)
    add_rectangle_outline(msp, (x1 * cell_size + out_margin, i * cell_size - out_margin), (x2 * cell_size - out_margin, i * cell_size - 3 * out_margin), 0x00FFFFFF)

def draw_dxf(b, dxf_name):
    doc = ezdxf.readfile('blocks.dxf')
    msp = doc.blocks.new(name='AllBlock')
    nn = b.shape[0]
    b_c = b.copy()
    b[b == window] = wall
    no_access_outline_cells = []
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            if b[i, j] == wall: continue
            if i > 0:
                if b[i - 1, j] == wall: no_access_outline_cells.append((i, j, 0))
            else: no_access_outline_cells.append((i, j, 0))
            if i < b.shape[0] - 1:
                if b[i + 1, j] == wall: no_access_outline_cells.append((i, j, 1))
            else: no_access_outline_cells.append((i, j, 1))
            if j > 0:
                if b[i, j - 1] == wall: no_access_outline_cells.append((i, j, 3))
            else: no_access_outline_cells.append((i, j, 3))
            if j < b.shape[1] - 1:
                if b[i, j + 1] == wall: no_access_outline_cells.append((i, j, 2))
            else: no_access_outline_cells.append((i, j, 2))
    b[b_c == window] = window
    # door
    pos_9390 = get_pos_clean(np.where(b == 9390))
    pos_9090 = get_pos_clean(np.where(b == 9090))
    pos_9092 = get_pos_clean(np.where(b == 9092))
    specials = []
    if pos_9390[0] == pos_9090[0]:
        if pos_9092[0] > pos_9090[0]:
            specials.append((pos_9390[0], pos_9390[1], 0))
            specials.append((pos_9090[0], pos_9090[1], 0))
            block_name = 'Door_Wide' if pos_9390[1] > pos_9090[1] else 'Door_Wide_Mirror'
            left, top = min(pos_9390[1], pos_9090[1]) * cell_size, (nn - 1 - pos_9390[0]) * cell_size
            msp.add_blockref(block_name, (left, top))
        else:
            specials.append((pos_9390[0], pos_9390[1], 1))
            specials.append((pos_9090[0], pos_9090[1], 1))
    else:
        if pos_9092[1] > pos_9090[1]:
            specials.append((pos_9390[0], pos_9390[1], 3))
            specials.append((pos_9090[0], pos_9090[1], 3))
            block_name = 'Door_Wide' if pos_9390[0] < pos_9090[0] else 'Door_Wide_Mirror'
            left, top = pos_9090[1] * cell_size, (nn - 1 - max(pos_9390[0], pos_9090[0]) - 1) * cell_size
            msp.add_blockref(block_name, (left, top), dxfattribs={'rotation': 90})
        else:
            specials.append((pos_9390[0], pos_9390[1], 2))
            specials.append((pos_9090[0], pos_9090[1], 2))
            block_name = 'Door_Wide' if pos_9390[0] > pos_9090[0] else 'Door_Wide_Mirror'
            left, top = (pos_9090[1] + 1) * cell_size, (nn - 1 - min(pos_9390[0], pos_9090[0])) * cell_size
            msp.add_blockref(block_name, (left, top), dxfattribs={'rotation': 270})
    # window
    yy, xx = np.where(b_c == window)
    add_window(msp, nn - 1 - yy[0], xx[0], xx[-1] + 1)
    specials += [(yy[0] - 1, x, 1) for x in xx]
    # wall
    for i, j, k in no_access_outline_cells:
        if (i, j, k) in specials: continue
        left, top = j * cell_size, (nn - 1 - i) * cell_size
        right, bottom = left + cell_size, top - cell_size
        if k == 3:
            x1, x2 = left - out_margin, left + out_margin
            y1, y2 = top + out_margin, bottom - out_margin
        elif k == 2:
            x1, x2 = right - out_margin, right + out_margin
            y1, y2 = top + out_margin, bottom - out_margin
        elif k == 0:
            x1, x2 = left - out_margin, right + out_margin
            y1, y2 = top + out_margin, top - out_margin
        else:
            x1, x2 = left - out_margin, right + out_margin
            y1, y2 = bottom + out_margin, bottom - out_margin
        add_rectangle_filled(msp, (x1, y1), (x2, y2), 256)
    # wall
    add_robe(msp, b, no_access_outline_cells)
    # bed
    i, j = get_pos_clean(np.where(b == 8000))
    i1, j1 = get_pos_clean(np.where(b == 8500))
    for index in range(4):
        if (i, j, index) in no_access_outline_cells and (i1, j1, index) in no_access_outline_cells: break
    if index == 0:
        msp.add_blockref('Bed', (min(j, j1) * cell_size, (nn - 1 - i) * cell_size))
    elif index == 1:
        msp.add_blockref('Bed', (max(j, j1) * cell_size, (nn - 1 - i) * cell_size), dxfattribs={'rotation': 180})
    elif index == 3:
        msp.add_blockref('Bed', (j * cell_size, (nn - 1 - max(i, i1) - 1) * cell_size), dxfattribs={'rotation': 90})
    elif index == 2:
        msp.add_blockref('Bed', ((j + 1) * cell_size, (nn - 1 - min(i, i1)) * cell_size), dxfattribs={'rotation': 270})
    msp = doc.modelspace()
    msp.add_blockref('AllBlock', (0, 0))
    doc.saveas(dxf_name)
    dxf2img(dxf_name, os.path.splitext(dxf_name)[0] + '-1.png')
        

'''
doc.saveas('11.dxf')
dxf2img('11.dxf', '11.png')

with open('results(2022-11-12 02_49_05)/dxf_png/000004.pkl', 'rb') as f:
  a, b, c = pickle.load(f)
'''