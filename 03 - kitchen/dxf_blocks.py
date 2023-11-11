import ezdxf
from commons import *
import numpy as np
import pickle, os

import matplotlib.pyplot as plt
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend


cell_size = 550

def add_rectangle_filled(msp, pt1, pt2, color):
    msp.add_solid([pt1, (pt1[0], pt2[1]), (pt2[0], pt1[1]), pt2], dxfattribs={"color": color})

def add_rectangle_outline(msp, pt1, pt2, color):
    msp.add_polyline2d([pt1, (pt1[0], pt2[1]), pt2, (pt2[0], pt1[1])], close=True, dxfattribs={"true_color": color})

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

def put_doubles(msp, pos, bl_code, C):
    ww1, ww2 = 50, 200
    axis, pt, pts = pos
    if axis == 'x':
        rotation = 0 if C[pts[0], pt - 1] == 1 else 180
        left, top = pt * cell_size, (n - 1 - min(pts)) * cell_size
        right, bottom = left + cell_size, top - 2 * cell_size
        if bl_code == frig:
            block_name = 'Frig'
            pt = (left, top) if rotation == 0 else (right, bottom)
        elif bl_code == sink:
            block_name = 'Sink'
            pt = (left, top - cell_size) if rotation == 0 else (right, bottom + cell_size)
        elif bl_code == stove:
            block_name = 'Stove'
            pt = (left + cell_size, top - cell_size) if rotation == 0 else (right - cell_size, bottom + cell_size)
    else:
        rotation = 90 if C[pt - 1, pts[0]] == 1 else -90
        left, top = min(pts) * cell_size, (n - 1 - pt) * cell_size
        right, bottom = left + 2 * cell_size, top - cell_size
        if bl_code == frig:
            block_name = 'Frig'
            pt = (right, top) if rotation == 90 else (left, bottom)
        elif bl_code == sink:
            block_name = 'Sink'
            pt = (right - cell_size, top) if rotation == 90 else (left + cell_size, bottom)
        elif bl_code == stove:
            block_name = 'Stove'
            pt = (right - cell_size, top - cell_size) if rotation == 90 else (left + cell_size, bottom + cell_size)
    if rotation == 0: msp.add_blockref(block_name, pt)
    elif rotation == 90: msp.add_blockref(block_name, pt, dxfattribs={'rotation': -90})
    elif rotation == 180: msp.add_blockref(block_name, pt, dxfattribs={'rotation': 180})
    else: msp.add_blockref(block_name, pt, dxfattribs={'rotation': 90})

def put_Shaft(msp, pos, block_name='Shaft'):
    y_min, x_min = pos
    left, top = x_min * cell_size, (n - 1 - y_min) * cell_size
    right, bottom = left + cell_size, top - cell_size
    msp.add_blockref(block_name, (left, bottom))

def put_counter(msp, pos, block_name='counter'):
    ys, xs = pos
    for x, y in zip(xs, ys):
        left, top = x * cell_size, (n - 1 - y) * cell_size
        msp.add_blockref(block_name, (left, top))

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

def draw_dxf(A, B, C, pins, windows_len, dxf_name):
    doc = ezdxf.readfile('Blocks.dxf')
    msp = doc.blocks.new(name='AllBlock')
    pos_9390 = get_pos_clean(np.where(C == 9390))
    pos_9392 = get_pos_clean(np.where(C == 9392))
    pos_9090 = get_pos_clean(np.where(C == 9090))
    specials = put_door(msp, pos_9390, pos_9392, pos_9090, C)
    specials += add_window(msp, pins[window], windows_len)
    put_Shaft(msp, get_pos_clean(np.where(C == shaft)))
    put_counter(msp, np.where(C == counter))
    put_doubles(msp, pins[frig], frig, C)
    put_doubles(msp, pins[sink], sink, C)
    put_doubles(msp, pins[stove], stove, C)
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
