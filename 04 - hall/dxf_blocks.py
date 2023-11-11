import ezdxf
from commons import *
import numpy as np
import pickle, os

import matplotlib.pyplot as plt
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend


cell_size = 550
out_margin = 50
margin, margin1 = 25, 40

def add_rectangle_filled(msp, pt1, pt2, color):
    msp.add_solid([pt1, (pt1[0], pt2[1]), (pt2[0], pt1[1]), pt2], dxfattribs={"color": color})

def add_rectangle_outline(msp, pt1, pt2, color):
    msp.add_polyline2d([pt1, (pt1[0], pt2[1]), pt2, (pt2[0], pt1[1])], close=True, dxfattribs={"true_color": color})

def get_pos_clean(np_pos):
    return np_pos[0][0], np_pos[1][0]

def Cells(msp):
    for i in range(n):
        msp.add_line((0, i * cell_size), ((n - 1) * cell_size, i * cell_size), dxfattribs={"color": 3})
    for i in range(n):
        msp.add_line((i * cell_size, 0), (i * cell_size, (n - 1) * cell_size), dxfattribs={"color": 3})

def add_window(msp, window_pos, windows_len):
    ww1, ww2 = 50, 200
    _, y, xs = window_pos
    specials_w = []
    for ind in range(xs[0], xs[1] + 1):
        specials_w.append((y + 1, ind, 2))
    # draw
    pt1 = xs[0] * cell_size + ww1, (n - 2 - y) * cell_size
    pt2 = pt1[0] + windows_len * cell_size - 2 * ww1, pt1[1] + ww2
    pt3 = pt1[0], pt1[1] + ww1
    pt4 = pt2[0], pt1[1] + ww1
    add_rectangle_outline(msp, pt1, pt2, 0x00FFFFFF)
    msp.add_line(pt3, pt4, dxfattribs={"true_color": 0x00FFFFFF})
    return specials_w

def draw_dxf(B_cl, no_access_outline_cells, access_outline_cells, additional_cells, dxf_name):
    doc = ezdxf.readfile('blocks.dxf')
    msp = doc.blocks.new(name='AllBlock')
    pos_9090 = get_pos_clean(np.where(B_cl == 9090))
    pos_9092 = get_pos_clean(np.where(B_cl == 9092))
    pos_9390 = get_pos_clean(np.where(B_cl == 9390))
    specials = []
    # door
    left, top = pos_9090[1] * cell_size, (n - 1 - pos_9090[0]) * cell_size
    if pos_9090[0] == pos_9092[0]:
        dd_index = 0 if pos_9090[0] < pos_9390[0] else 1
        specials.append((pos_9090[0], pos_9090[1], dd_index))
        specials.append((pos_9092[0], pos_9092[1], dd_index))
        msp.add_blockref('Door_Wide', (left, top))
    else:
        dd_index = 2 if pos_9090[1] > pos_9390[1] else 3
        specials.append((pos_9090[0], pos_9090[1], dd_index))
        specials.append((pos_9092[0], pos_9092[1], dd_index))
        msp.add_blockref('Door_Wide_Mirror', (left + cell_size, top + cell_size), dxfattribs={'rotation': 270})
    # wall
    for i, j, k in no_access_outline_cells:
        if (i, j, k) in specials: continue
        left, top = j * cell_size, (n - 1 - i) * cell_size
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
    # sitting
    pos_6020 = get_pos_clean(np.where(B_cl == 6020))
    left, bottom = pos_6020[1] * cell_size, (n - 1 - pos_6020[0]) * cell_size - cell_size
    msp.add_blockref('Sitting', (left, bottom))
    # dinning
    pos_8016 = get_pos_clean(np.where(B_cl == 8016))
    left, top = pos_8016[1] * cell_size, (n - 1 - pos_8016[0]) * cell_size
    msp.add_blockref('Dining', (left, top))
    # robe
    add_robe(msp, B_cl, no_access_outline_cells)
    # write
    msp = doc.modelspace()
    msp.add_blockref('AllBlock', (0, 0))
    # access
    for ii in range(0, 8, 2):
        i, j, k = access_outline_cells[ii]
        left, top = j * cell_size, (n - 1 - i) * cell_size
        if k == 0: msp.add_blockref('Door_Narrow', (left + 2 * cell_size, top), dxfattribs={'rotation': 270})
        elif k == 1: msp.add_blockref('Door_Narrow', (left, top - cell_size), dxfattribs={'rotation': 90})
        elif k == 2: msp.add_blockref('Door_Narrow', (left + cell_size, top - 2 * cell_size), dxfattribs={'rotation': 180})
        elif k == 3: msp.add_blockref('Door_Narrow', (left, top))
    doc.saveas(dxf_name)
    dxf2img(dxf_name, os.path.splitext(dxf_name)[0] + '-1.png')

def add_robe(msp, B_cl, no_access_outline_cells):
    yy, xx = np.where(B_cl == robe)
    for i, j in zip(yy, xx):
        left, bottom = j * cell_size, (n - 1 - i) * cell_size - cell_size
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
