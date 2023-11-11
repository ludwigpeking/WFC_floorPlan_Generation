import tkinter as tk
from tkinter import ttk
from PIL import ImageGrab, Image, ImageDraw
import numpy as np
from datetime import datetime
import os

cells = {
    # Bedroom
    'passage': [(1, 1), (90, 90), (50, 50)],
    'bed': [(3, 4), (80, 85), (0, 6)],
    'robe': [(1, 1), (0, 0), (90, 91)],
    'door': [(2, 2), (90, 93), (90, 92)],
    'window': [(1, 1), (50, 50), (90, 90)],
    'wall': [(1, 1), (0, 0), (0, 0)],
    # Bath
    # 'tub': [(2, 3), (30, 31), (50, 52)],
    'toilet': [(2, 2), (10, 11), (30, 31)],
    'shower': [(2, 2), (20, 21), (20, 21)],
    'lavatory': [(1, 1), (70, 70), (30, 30)],
    'shaft': [(1, 1), (10, 10), (10, 10)],
    'washmach': [(1, 1), (30, 30), (20, 20)],
	# Kitchen
	'stove':[(1, 1), (70, 70), (60, 60)],
	'frig':[(1, 1), (0, 0), (60, 60)],
	'counter':[(1, 1), (30, 30), (40, 40)],
	'sink':[(1, 1), (75, 75), (75, 75)],
	#'shaft' is used in kitch
	# Living
	'sitting':[(5, 6), (60, 64), (20, 25)],
	'dining':[(3, 3), (80, 85), (10, 16)],
	'access':[(1, 2), (90, 95), (80, 86)],
	'passage_wide': [(2, 1), (80, 80), (50, 50)]
}
U_grad = 10
V_grad = 10

class ExampleApp(tk.Tk):
	def __init__(self):
		tk.Tk.__init__(self)
		
		self.title('Design')
		screenHeight = self.winfo_screenheight()
		screenWidth = self.winfo_screenwidth()		
		self.left = 200
		self.top = 100
		self.width = screenWidth - 2 * self.left
		self.height = screenHeight - 2 * self.top
		self.geometry(f'{self.width}x{self.height}+{self.left}+{self.top}')
		self.resizable(width=False, height=False)
		
		label_tiles = tk.Label(self, text='tiles', font=('Times New Roman', 15), justify='left')
		label_tiles.place(x=0, y=10, width=50, height=20)
		content1 = tk.StringVar()
		self.cell_rows = tk.Entry(self, textvariable=content1, font=('Times New Roman', 12))
		self.cell_rows.place(x=80, y=10, width=30, height=20)
		self.cell_rows.insert(0, '16')
		label_cross = tk.Label(self, text='x', font=('Curier', 15))
		label_cross.place(x=115, y=8, width=10, height=20)		
		content2 = tk.StringVar()
		self.cell_cols = tk.Entry(self, textvariable=content2, font=('Times New Roman', 12))
		self.cell_cols.place(x=130, y=10, width=30, height=20)
		self.cell_cols.insert(0, '12')
		new_button = tk.Button(self, text='New', font=('Times New Roman', 15), command=self.init_cells)
		new_button.place(x=180, y=5, width=80, height=30)
		save_button = tk.Button(self, text='Save', font=('Times New Roman', 15), command=self.save_cells)
		save_button.place(x=300, y=5, width=80, height=30)
		
		style = ttk.Style(self)
		style.configure('flat.TButton', relief=tk.RAISED, font=('Times New Roman', 10, 'italic'))
		style.configure('sunken.TButton', relief=tk.SUNKEN, font=('Times New Roman', 12, 'italic'), foreground="red", background="blue")

		# Bedroom
		self.wall_button = ttk.Button(self, text='wall', command=lambda:self.select_objects(0), style='flat.TButton')
		self.wall_button.place(x=10, y=50, width=50, height=30)
		self.passage_button = ttk.Button(self, text='passage', command=lambda:self.select_objects(1), style='flat.TButton')
		self.passage_button.place(x=65, y=50, width=80, height=30)
		self.bed_button = ttk.Button(self, text='bed', command=lambda:self.select_objects(2), style='flat.TButton')
		self.bed_button.place(x=150, y=50, width=50, height=30)
		self.robe_button = ttk.Button(self, text='robe', command=lambda:self.select_objects(3), style='flat.TButton')
		self.robe_button.place(x=205, y=50, width=50, height=30)
		self.window_button = ttk.Button(self, text='window', command=lambda:self.select_objects(4), style='flat.TButton')
		self.window_button.place(x=260, y=50, width=80, height=30)
		self.door_button = ttk.Button(self, text='door', command=lambda:self.select_objects(5), style='flat.TButton')
		self.door_button.place(x=345, y=50, width=50, height=30)
		# Bath
		self.toilet_button = ttk.Button(self, text='toilet', command=lambda: self.select_objects(6), style='flat.TButton')
		self.toilet_button.place(x=400, y=50, width=80, height=30)
		self.shower_button = ttk.Button(self, text='shower', command=lambda: self.select_objects(7), style='flat.TButton')
		self.shower_button.place(x=485, y=50, width=70, height=30)
		self.lavatory_button = ttk.Button(self, text='lavatory', command=lambda: self.select_objects(8), style='flat.TButton')
		self.lavatory_button.place(x=560, y=50, width=50, height=30)
		self.shaft_button = ttk.Button(self, text='shaft', command=lambda: self.select_objects(9), style='flat.TButton')
		self.shaft_button.place(x=615, y=50, width=80, height=30)
		self.washmach_button = ttk.Button(self, text='washmach', command=lambda: self.select_objects(10), style='flat.TButton')
		self.washmach_button.place(x=700, y=50, width=80, height=30)
		self.stove_button = ttk.Button(self, text='stove', command=lambda: self.select_objects(11),
										  style='flat.TButton')
		self.stove_button.place(x=10, y=85, width=75, height=30)
		self.frig_button = ttk.Button(self, text='frig', command=lambda: self.select_objects(12),
									   style='flat.TButton')
		self.frig_button.place(x=90, y=85, width=80, height=30)
		self.counter_button = ttk.Button(self, text='counter', command=lambda: self.select_objects(13),
									  style='flat.TButton')
		self.counter_button.place(x=175, y=85, width=80, height=30)
		self.sink_button = ttk.Button(self, text='sink', command=lambda: self.select_objects(14),
									  style='flat.TButton')
		self.sink_button.place(x=260, y=85, width=80, height=30)
		self.sitting_button = ttk.Button(self, text='sitting', command=lambda: self.select_objects(15),
									  style='flat.TButton')
		self.sitting_button.place(x=10, y=120, width=80, height=30)
		self.dining_button = ttk.Button(self, text='dining', command=lambda: self.select_objects(16),
									  style='flat.TButton')
		self.dining_button.place(x=95, y=120, width=80, height=30)
		self.access_button = ttk.Button(self, text='access', command=lambda: self.select_objects(17),
									  style='flat.TButton')
		self.access_button.place(x=180, y=120, width=80, height=30)
		self.passage_wide_button = ttk.Button(self, text='passage_wide', command=lambda: self.select_objects(18),
									  style='flat.TButton')
		self.passage_wide_button.place(x=265, y=120, width=100, height=30)
		# 'stove': [(1, 1), (30, 30), (60, 60)],
		# 'frig': [(1, 1), (0, 0), (60, 60)],
		# 'counter': [(1, 1), (30, 30), (40, 40)],
		# 'sink': [(1, 1), (50, 50), (40, 40)]
		# collects
		# 'sitting': [(5, 6), (80, 85), (20, 26)],
		# 'dining': [(3, 3), (90, 95), (10, 16)],
		# 'acess': [(2, 2), (90, 95), (80, 86)],
		# 'passage_wide': [(2, 1), (85, 85), (50, 50)]
		self.input_buttons = [self.wall_button, self.passage_button, self.bed_button, self.robe_button,
							  self.window_button, self.door_button, self.toilet_button, self.shower_button,
							  self.lavatory_button, self.shaft_button, self.washmach_button,self.stove_button,
							  self.frig_button, self.counter_button, self.sink_button,
							  self.sitting_button, self.dining_button, self.access_button, self.passage_wide_button ]
		self.input_types = ['wall', 'passage', 'bed', 'robe', 'window', 'door',
							'toilet', 'shower', 'lavatory', 'shaft', 'washmach',
							'stove', 'frig', 'counter', 'sink', 'sitting', 'dining', 'access', 'passage_wide']
		self.button_index = None
		self.input_type = None
		
		self.canvas = tk.Canvas(self)
		self.canvas_left = 20
		self.canvas_top = 80 + 20
		self.canvas_height = self.height - self.canvas_top - 50
		self.canvas_width = self.width - 2 * self.canvas_left
		self.canvas.place(x=20, y=150, width=self.canvas_width, height=self.canvas_height)
		# events
		self.canvas.bind("<Motion>", self.check_area)
		self.canvas.bind("<ButtonPress-1>", self.on_button_press)
		self.canvas.bind("<B1-Motion>", self.on_move_press)
		self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
		self.LButtonClicked = False
		
		self.init_cells()

		self.result_folder = 'results(' + str(datetime.now()).replace(':', '_')[:19] + ')'
		self.success_number = 1

	def init_cells(self):
		self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), (255, 255, 255))
		self.draw = ImageDraw.Draw(self.image)

		self.canvas.delete("all")
		self.cols = int(self.cell_cols.get()) - 2
		self.rows = int(self.cell_rows.get()) - 2
		cols_ext, rows_ext = self.cols + 2, self.rows + 2
		self.cell_size = min(self.canvas_height // rows_ext, self.canvas_width // cols_ext)
		self.init_circle_radius = max(self.cell_size // 10, 3)
		self.cell_top0 = (self.canvas_height - self.cell_size * rows_ext) // 2
		self.cell_left0 = (self.canvas_width - self.cell_size * cols_ext) // 2
		self.cell_right0 = self.cell_left0 + self.cell_size * cols_ext
		self.cell_bottom0 = self.cell_top0 + self.cell_size * rows_ext
		self.cell_area_left, self.cell_area_top = self.cell_left0 + self.cell_size, self.cell_top0 + self.cell_size
		self.cell_area_right, self.cell_area_bottom = self.cell_area_left + self.cell_size * self.cols, self.cell_area_top + self.cell_size * self.rows
		for i in range(1, self.cols):
			x = self.cell_area_left + i * self.cell_size
			self.canvas.create_line(x, self.cell_area_top, x, self.cell_area_bottom)
			self.draw.line([(x, self.cell_area_top), (x, self.cell_area_bottom)], fill='black', width=0)
		for i in range(1, self.rows):
			y = self.cell_area_top + i * self.cell_size
			self.canvas.create_line(self.cell_area_left, y, self.cell_area_right, y)
			self.draw.line([(self.cell_area_left, y), (self.cell_area_right, y)], fill='black', width=0)
		self.A = np.zeros((rows_ext, cols_ext), dtype='<U4')
		self.A[1:-1, 1:-1] = '9999'
		self.A[0, :] = '0000'
		self.A[-1, :] = '0000'
		self.A[:, 0] = '0000'
		self.A[:, -1] = '0000'
		temp = [None] * cols_ext
		self.pics = [temp.copy() for _ in range(rows_ext)]
		for i in range(cols_ext):
			cell_left = self.cell_left0 + i * self.cell_size
			cell_top = self.cell_top0
			cell_right = cell_left + self.cell_size
			cell_bottom = cell_top + self.cell_size
			self.pics[0][i] = self.canvas.create_rectangle(cell_left, cell_top, cell_right, cell_bottom, fill='black')
			self.draw.rectangle([cell_left, cell_top, cell_right, cell_bottom], fill='black')
			cell_top = self.cell_area_bottom
			cell_bottom = cell_top + self.cell_size
			self.pics[-1][i] = self.canvas.create_rectangle(cell_left, cell_top, cell_right, cell_bottom, fill='black')
			self.draw.rectangle([cell_left, cell_top, cell_right, cell_bottom], fill='black')
		for i in range(1, rows_ext - 1):
			cell_left = self.cell_left0
			cell_top = self.cell_top0 + i * self.cell_size
			cell_right = cell_left + self.cell_size
			cell_bottom = cell_top + self.cell_size
			self.pics[i][0] = self.canvas.create_rectangle(cell_left, cell_top, cell_right, cell_bottom, fill='black')
			self.draw.rectangle([cell_left, cell_top, cell_right, cell_bottom], fill='black')
			cell_left = self.cell_area_right
			cell_right = cell_left + self.cell_size
			self.pics[i][-1] = self.canvas.create_rectangle(cell_left, cell_top, cell_right, cell_bottom, fill='black')
			self.draw.rectangle([cell_left, cell_top, cell_right, cell_bottom], fill='black')

	def save_cells(self):
		if self.success_number == 1: os.mkdir(self.result_folder)
		imgName = self.result_folder + '/%06d.png' % self.success_number
		x = self.winfo_rootx() + self.canvas_left + self.cell_left0 - 10
		y = self.winfo_rooty() + self.canvas_top + self.cell_top0 - 10
		x1 = x + self.cell_size * (self.cols + 2) + 20
		y1 = y + self.cell_size * (self.rows + 2) + 20
		#ImageGrab.grab().crop((x, y, x1, y1)).save(imgName)
		x = max(self.cell_left0 - 10, 0)
		y = max(self.cell_top0 - 10, 0)
		x1 = min(x + self.cell_size * (self.cols + 2) + 20, self.canvas_width)
		y1 = min(y + self.cell_size * (self.rows + 2) + 20, self.canvas_height)
		self.image.crop((x, y, x1, y1)).save(imgName)
		#self.image.save(imgName)
		csvName = self.result_folder + '/%06d.csv' % self.success_number
		with open(csvName, 'wt') as f:
			for i in range(self.A.shape[0]):
				f.write(','.join(self.A[i]) + '\n')
		self.success_number += 1

	def select_objects(self, index):
		if self.button_index is not None: self.input_buttons[self.button_index].configure(style='flat.TButton')
		self.button_index = index
		self.input_type = self.input_types[index]
		self.input_buttons[self.button_index].configure(style='sunken.TButton')
	
	def check_inside(self, x, y):
		if self.input_type is None: return False
		if self.input_type in ['window', 'access', 'wall']:
			return self.cell_left0 <= x <= self.cell_right0 and self.cell_top0 <= y <= self.cell_bottom0
		return self.cell_area_left <= x <= self.cell_area_right and self.cell_area_top <= y <= self.cell_area_bottom

	def check_area(self, event):
		if self.check_inside(event.x, event.y): self.canvas.config(cursor="tcross")
		else: self.canvas.config(cursor="arrow")
	
	def get_cell_pos(self, x, y):
		cell_x = (x - self.cell_left0) // self.cell_size
		cell_y = (y - self.cell_top0) // self.cell_size
		return cell_x, cell_y

	def on_button_press(self, event):
		if not self.check_inside(event.x, event.y): return
		if cells[self.input_type]['uv'] == 1:
			cell_x, cell_y = self.get_cell_pos(event.x, event.y)
			cell_left = self.cell_left0 + cell_x * self.cell_size
			cell_top = self.cell_top0 + cell_y * self.cell_size
			cell_right = cell_left + self.cell_size
			cell_bottom = cell_top + self.cell_size
			color = self.color_by_uv(cells[self.input_type]['u_r'][0], cells[self.input_type]['v_r'][0])
			if self.pics[cell_y][cell_x] is not None: self.canvas.delete(self.pics[cell_y][cell_x])
			self.pics[cell_y][cell_x] = self.canvas.create_rectangle(cell_left, cell_top, cell_right, cell_bottom, fill=color)
			self.draw.rectangle([cell_left, cell_top, cell_right, cell_bottom], fill=color)
			self.A[cell_y, cell_x] = self.value_by_uv(cells[self.input_type]['u_r'][0], cells[self.input_type]['v_r'][0])
		else:
			self.LButtonClicked = True
			self.start_x, self.start_y = event.x, event.y
			self.start_circle = self.canvas.create_oval(self.start_x - self.init_circle_radius, self.start_y - self.init_circle_radius, self.start_x + self.init_circle_radius, self.start_y + self.init_circle_radius)
			self.cur_line = self.canvas.create_line(self.start_x, self.start_y, self.start_x + 1, self.start_y + 1, arrow=tk.LAST)
		
	def on_move_press(self, event):
		if not self.LButtonClicked: return
		if not self.check_inside(event.x, event.y): return
		self.cur_x, self.cur_y = event.x, event.y
		self.canvas.coords(self.cur_line, self.start_x, self.start_y, self.cur_x, self.cur_y)
		
	def on_button_release(self, event):
		if not self.LButtonClicked: return
		if self.check_inside(event.x, event.y): self.cur_x, self.cur_y = event.x, event.y
		self.canvas.delete(self.start_circle)
		self.canvas.delete(self.cur_line)
		# draw & set
		delta_x, delta_y = self.cur_x - self.start_x, self.cur_y - self.start_y
		if max(abs(delta_x), abs(delta_y)) < self.init_circle_radius * 3: return
		cell_step_x = -1 if delta_x < 0 else 1
		cell_step_y = -1 if delta_y < 0 else 1
		if abs(delta_x) < abs(delta_y):
			x_info = cells[self.input_type]['u_a'], cells[self.input_type]['u_r'][0], cells[self.input_type]['u_s']
			y_info = cells[self.input_type]['v_a'], cells[self.input_type]['v_r'][0], cells[self.input_type]['v_s']
			inverted = False
		else:
			x_info = cells[self.input_type]['v_a'], cells[self.input_type]['v_r'][0], cells[self.input_type]['v_s']
			y_info = cells[self.input_type]['u_a'], cells[self.input_type]['u_r'][0], cells[self.input_type]['u_s']
			inverted = True
		cell_x, cell_y = self.get_cell_pos(self.start_x, self.start_y)
		for i in range(y_info[0]):
			cell_pos_y = cell_y + i * cell_step_y
			if cell_pos_y < 0 or cell_pos_y >= self.A.shape[0]: continue
			cell_val_y = int(y_info[1] + i * y_info[2])
			for j in range(x_info[0]):
				cell_pos_x = cell_x + j * cell_step_x
				if cell_pos_x < 0 or cell_pos_x >= self.A.shape[1]: continue
				cell_val_x = int(x_info[1] + j * x_info[2])
				cell_left = self.cell_left0 + cell_pos_x * self.cell_size
				cell_top = self.cell_top0 + cell_pos_y * self.cell_size
				cell_right = cell_left + self.cell_size
				cell_bottom = cell_top + self.cell_size
				if inverted: cell_val_x_c, cell_val_y_c = cell_val_y, cell_val_x
				else: cell_val_x_c, cell_val_y_c = cell_val_x, cell_val_y
				color = self.color_by_uv(cell_val_x_c, cell_val_y_c)
				if self.pics[cell_pos_y][cell_pos_x] is not None: self.canvas.delete(self.pics[cell_pos_y][cell_pos_x])
				self.pics[cell_pos_y][cell_pos_x] = self.canvas.create_rectangle(cell_left, cell_top, cell_right, cell_bottom, fill=color, outline=color)
				self.draw.rectangle([cell_left, cell_top, cell_right, cell_bottom], fill=color)
				self.A[cell_pos_y, cell_pos_x] = self.value_by_uv(cell_val_x_c, cell_val_y_c)
		self.LButtonClicked = False

	def value_by_uv(self, u, v):
		return '%02d%02d' % (u, v)
	
	def color_by_uv(self, u, v):
		red, green, blue = int(u * U_grad), int(v * V_grad), 0
		color = '#'
		color += str(hex(red))[-2:].replace('x', '0')
		color += str(hex(green))[-2:].replace('x', '0')
		color += str(hex(blue))[-2:].replace('x', '0')
		return color


if __name__ == "__main__":
	for key in cells:
		cur_list = cells[key]
		cells[key] = {
			'u_a': cur_list[0][0],
			'v_a': cur_list[0][1],
			'u_r': cur_list[1],
			'v_r': cur_list[2],
			'uv': cur_list[0][0] * cur_list[0][1],
			'u_s': ((cur_list[1][1] - cur_list[1][0]) / (cur_list[0][0] - 1)) if cur_list[0][0] > 1 else 0,
			'v_s': ((cur_list[2][1] - cur_list[2][0]) / (cur_list[0][1] - 1)) if cur_list[0][1] > 1 else 0,
			'uv_same': cur_list[0][0] == cur_list[0][1],
		}
	app = ExampleApp()
	app.mainloop()
