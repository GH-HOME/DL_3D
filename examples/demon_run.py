import tensorflow as tf
import numpy as np
from PIL import Image
from scipy.misc import imresize
import cv2
from matplotlib import pyplot as plt
import os
import sys

examples_dir = os.path.dirname(__file__)
weights_dir = os.path.join(examples_dir,'..','weights')
sys.path.insert(0, os.path.join(examples_dir, '..', 'python'))
sys.path.insert(0, os.path.join(examples_dir, '..', 'lmbspecialops/python'))
from depthmotionnet.networks_original import *



def prepare_input_data(img1, img2, data_format):
    """Creates the arrays used as input from the two images."""
    # scale images if necessary
    if img1.size[0] != 256 or img1.size[1] != 192:
        img1 = img1.resize((256,192))
    if img2.size[0] != 256 or img2.size[1] != 192:
        img2 = img2.resize((256,192))
    img2_2 = img2.resize((64,48))

    # transform range from [0,255] to [-0.5,0.5]
    img1_arr = np.array(img1).astype(np.float32)/255 -0.5
    img2_arr = np.array(img2).astype(np.float32)/255 -0.5
    img2_2_arr = np.array(img2_2).astype(np.float32)/255 -0.5

    if data_format == 'channels_first':
        img1_arr = img1_arr.transpose([2,0,1])
        img2_arr = img2_arr.transpose([2,0,1])
        img2_2_arr = img2_2_arr.transpose([2,0,1])
        image_pair = np.concatenate((img1_arr,img2_arr), axis=0)
    else:
        image_pair = np.concatenate((img1_arr,img2_arr),axis=-1)

    result = {
        'image_pair': image_pair[np.newaxis,:],
        'image1': img1_arr[np.newaxis,:], # first image
        'image2_2': img2_2_arr[np.newaxis,:], # second image with (w=64,h=48)
    }
    return result


def demon_net_build(data_format):

    # DeMoN has been trained for specific internal camera parameters.
    #
    # If you use your own images try to adapt the intrinsics by cropping
    # to match the following normalized intrinsics:
    #
    #  K = (0.89115971  0           0.5)
    #      (0           1.18821287  0.5)
    #      (0           0           1  ),
    #  where K(1,1), K(2,2) are the focal lengths for x and y direction.
    #  and (K(1,3), K(2,3)) is the principal point.
    #  The parameters are normalized such that the image height and width is 1.
    #

    gpu_options = tf.GPUOptions()
    gpu_options.per_process_gpu_memory_fraction=0.8
    session = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))

    # init networks
    bootstrap_net = BootstrapNet(session, data_format)
    iterative_net = IterativeNet(session, data_format)
    refine_net = RefinementNet(session, data_format)

    session.run(tf.global_variables_initializer())

    # load weights
    saver = tf.train.Saver()
    saver.restore(session,os.path.join(weights_dir,'demon_original'))

    return bootstrap_net,iterative_net,refine_net


def run_two_image(img1,img2):
    org_width,org_height=img1.size
    input_data = prepare_input_data(img1,img2,data_format)
    result = bootstrap_net.eval(input_data['image_pair'], input_data['image2_2'])
    # run the network
    for i in range(3):
        result = iterative_net.eval(
            input_data['image_pair'],
            input_data['image2_2'],
            result['predict_depth2'],
            result['predict_normal2'],
            result['predict_rotation'],
            result['predict_translation']
        )
    rotation = result['predict_rotation']
    translation = result['predict_translation']
    result = refine_net.eval(input_data['image1'],result['predict_depth2'])
    result_depth=result['predict_depth0'].squeeze().copy()
    output_depth=imresize(result_depth,(org_height,org_width))
    return output_depth


#init net scripts
data_format=[]
if tf.test.is_gpu_available(True):
    data_format='channels_first'
else: # running on cpu requires channels_last data format
    data_format='channels_last'

bootstrap_net,iterative_net,refine_net=demon_net_build(data_format)


# read data and run scripts
# img1 = Image.open(os.path.join(examples_dir,'0000.png'))
# img2 = Image.open(os.path.join(examples_dir,'0019.png'))
# output_depth=run_two_image(img1,img2)
# plt.imshow(output_depth, cmap='Greys')
# plt.show()
# cv2.imwrite('test.png',output_depth)




# try to visualize the point cloud
# try:
#     from depthmotionnet.vis import *
#     visualize_prediction(
#         inverse_depth=result['predict_depth0'],
#         image=input_data['image_pair'][0,0:3] if data_format=='channels_first' else input_data['image_pair'].transpose([0,3,1,2])[0,0:3],
#         rotation=rotation,
#         translation=translation)
# except ImportError as err:
#     print("Cannot visualize as pointcloud.", err)
