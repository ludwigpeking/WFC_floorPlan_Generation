from functions import *
from datetime import datetime

output_num = 100
output_folder = os.path.join(hall_dst_path, 'final', 'results(' + str(datetime.now()).replace(':', '_')[:19] + ')')
os.makedirs(output_folder)

pkls = glob(os.path.join(hall_dst_path, '*', '*.pkl'))
score = [float(os.path.basename(pkl).split('(')[1].split(')')[0].split('-')[-1]) for pkl in pkls]
orders = np.argsort(score)[::-1]

for i in range(output_num):
	index = orders[i]
	pkl_file = pkls[index]
	draw_final(pkl_file)
	src_dxf = pkl_file.replace('.pkl', '.dxf')
	src_png = pkl_file.replace('.pkl', '.png')
	dst_dxf = os.path.join(output_folder, '%03d-' % i + os.path.basename(src_dxf))
	dst_png = dst_dxf.replace('.dxf', '.png')
	shutil.move(src_dxf, dst_dxf)
	shutil.move(src_png, dst_png)
