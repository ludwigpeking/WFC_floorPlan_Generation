# block codes
doors = (9090, 9392, 9390, 9092)
dining = (8010, 8013, 8016, 8510, 8513, 8516)
dining_size = (3, 2)
sitting_s = (6419, 6426)
sitting_all = (6019, 6426)
sitting_size = (5, 6)
access = (9080, 9086)
robe = 90  # 0090  # passage, doors -> robe
passage_w = 9050  # wide cell passage, doors -> passage
passage_s = 3020  # single cell passage, doors -> passage
robe_in = tuple(list(doors) + [passage_w, passage_s])
window = 5090
wall = 0 # 0000
# others
n = 20  # params
loops = 100000  # loops
TotalScoreThreshold = 11  # scoring

# showing
cell_size = 28
showing = False
duration = 1
