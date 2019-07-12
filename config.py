num_classes = 2
num_crops = 0
batch_size = 44
input_image_size = 512

checkpoint = './'
result_dir = './result'
test_dir = './img'

nms_threshold = 0.5
cls_threshold = 0.5

anchor_areas = [16*16., 64*64.]
aspect_ratios = [0.25,0.5,1.,2.,3.,5.,7.]
scale_ratios = [1., 2., 4.]

num_head_layers = 4