
# create binary images from a file
from load_config import config
import create_binary_images
# if desired could modify create_binary_images.cfg
create_binary_images.cfg['max_frames'] = 1000

imgs_dir = config['bsds_dir'] 
extension = '.jpg'

saved_images = create_binary_images.main(figure_dir=imgs_dir, 
img_file_range=(0,50),
extension=extension) #TODO: return a list of these generated Bernoulli frames

# TODO: add flag to save mask movie; save the average mask; set it up so it only uses pre-generated bernoulli frames
# import run_images
# run_images.main() # run images is in effect the entire example (loads images)

# evaluates an objective function 
