# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Copyright 2017 Modifications Clement Godard.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np

def get_path(split_line,data_path):
    data_path = data_path.decode("utf-8")
    split_line = [split_line[i].decode("utf-8") for i in range(len(split_line))]

    im_path = data_path + split_line[0]


    print(im_path)
    return im_path
    # side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
    #
    # data_path = data_path.decode("utf-8")
    # split_line = [split_line[i].decode("utf-8") for i in range(len(split_line))]
    #
    # frame_id = int(split_line[1])
    # reference_path = data_path+split_line[0]+'/image_0{}/data'.format(side_map[split_line[2]])
    #
    # reference_image_path = reference_path+'/{:010d}.jpg'.format(frame_id)
    #
    # return reference_image_path#,reference_param_path,side_param_path
def get_path_kitti(split_line,data_path):
    side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
    data_path = data_path.decode("utf-8")

    split_line = [split_line[i].decode("utf-8") for i in range(len(split_line))]
    date = split_line[0].split('/')[0]
    date = data_path + date

    frame_id = int(split_line[1])
    reference_path = data_path+split_line[0]+'/image_0{}/data'.format(side_map[split_line[2]])

    reference_image_path = reference_path+'/{:010d}.jpg'.format(frame_id)
    reference_param_path = date + '/param0{}.txt'.format(side_map[split_line[2]])
    reference_rect_path = date+'/R_rect.txt'
    print(reference_image_path)
    return reference_image_path,reference_param_path,reference_rect_path

def get_path_idol(split_line,data_path):
    data_path = data_path.decode("utf-8")
    split_line = [split_line[i].decode("utf-8") for i in range(len(split_line))]

    left_im_path = data_path + split_line[0]
    reference_im_path = data_path + split_line[1]
    right_im_path = data_path + split_line[2]

    left_param_path = data_path + split_line[3]
    reference_param_path = data_path + split_line[4]
    right_param_path = data_path + split_line[5]

    return left_im_path,reference_im_path,right_im_path,left_param_path,reference_param_path,right_param_path

def get_path_train(split_line,data_path):
    side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
    other_side = {"r": "l", "l": "r"}

    data_path = data_path.decode("utf-8")
    split_line = [split_line[i].decode("utf-8") for i in range(len(split_line))]

    date = split_line[0].split('/')[0]
    date = data_path+date
    frame_id = int(split_line[1])
    reference_path = data_path+split_line[0]+'/image_0{}/data'.format(side_map[split_line[2]])
    side_path = data_path+split_line[0]+'/image_0{}/data'.format(side_map[other_side[split_line[2]]])

    reference_image_path = reference_path+'/{:010d}.jpg'.format(frame_id)
    side_image_path = side_path+'/{:010d}.jpg'.format(frame_id)
    reference_param_path = date + '/param0{}.txt'.format(side_map[split_line[2]])
    side_param_path = date + '/param0{}.txt'.format(side_map[other_side[split_line[2]]])
    pre_frame_path = reference_path+'/{:010d}.jpg'.format(frame_id-1)
    post_frame_path = reference_path + '/{:010d}.jpg'.format(frame_id + 1)

    #date = tf.string_split([root_path], "/")
    # reference_path = tf.string_join([root_path,'/image0{}/data'.format(self.side_map[side])])
    # side_path = tf.string_join([root_path,'/image0{}/data'.format(self.other_side[side])])
    #
    #
    # reference_image_path = tf.string_join([reference_path,'/{:010d{}.jpg'.format(frame_id)])
    # side_image_path = tf.string_join([side_path,'/{:010d{}.jpg'.format(frame_id)])
    # reference_param_path = tf.string_join([date,'/param0{}.txt'.format(self.side_map[side])])
    # side_param_path = tf.string_join([date, '/param0{}.txt'.format(self.other_side[side])])
    # pre_frame_path = tf.string_join([reference_path,'/{:010d{}.jpg'.format(int(frame_id-1))])
    # post_frame_path = tf.string_join([reference_path, '/{:010d{}.jpg'.format(int(frame_id + 1))])

    # reference_image_path=0
    # side_image_path=0
    # pre_frame_path=0
    # post_frame_path=0
    # reference_param_path=0
    # side_param_path=0

    return reference_image_path, side_image_path, pre_frame_path, post_frame_path, reference_param_path, side_param_path
#ver2
def load_kitti(path):
    print(path)
    num, side = path.shape

    l_k = np.zeros((num, 3 * 3), dtype=np.float32)
    l_r = np.zeros((num, 3 * 4), dtype=np.float32)
    rect = np.zeros((num,4*4),dtype =np.float32)
    for i in range(num):
        f_l = open(path[i][0], 'r')
        line_l = f_l.readlines()
        l_k[i]=(np.array(line_l[0].replace('\n','').split(' '))[:3*3]).astype(np.float32)
        l_r[i]=(np.array(line_l[1].replace('\n', '').split(' '))[:3 * 4]).astype(np.float32)

        f_rect = open(path[i][1],'r')
        line_rect = f_rect.readlines()
        rect[i]=(np.array(line_rect[0].replace('\n','').split(' '))[:4*4]).astype(np.float32)

        f_l.close()
        f_rect.close()
    print(l_k)
    print(l_r)
    print(rect)
    return l_k,l_r,rect

def load_param(path):
    num, side = path.shape
    l_k = np.zeros((num,3*3),dtype=np.float32)
    l_r = np.zeros((num,3*4),dtype=np.float32)
    r_k = np.zeros((num,3*3),dtype=np.float32)
    r_r = np.zeros((num,3*4),dtype=np.float32)
    # print('--------------------------------------------------------')
    # print('path={}'.format(path))
    for i in range(num):
        f_l = open(path[i][0], 'r')
        line_l = f_l.readlines()
        l_k[i]=(np.array(line_l[0].replace('\n','').split(' '))[:3*3]).astype(np.float32)
        l_r[i]=(np.array(line_l[1].replace('\n', '').split(' '))[:3 * 4]).astype(np.float32)

        f_r = open(path[i][1], 'r')
        line_r = f_r.readlines()
        r_k[i]=(np.array(line_r[0].replace('\n', '').split(' '))[:3 * 3]).astype(np.float32)
        r_r[i]=(np.array(line_r[1].replace('\n', '').split(' '))[:3 * 4]).astype(np.float32)

        f_l.close()
        f_r.close()
    return l_k,l_r,r_k,r_r

def load_kocca_param(path):
    num, side = path.shape
    l_k = np.zeros((num, 3 * 3), dtype=np.float32)
    l_r = np.zeros((num, 3 * 4), dtype=np.float32)
    m_k = np.zeros((num, 3 * 3), dtype=np.float32)
    m_r = np.zeros((num, 3 * 4), dtype=np.float32)
    r_k = np.zeros((num, 3 * 3), dtype=np.float32)
    r_r = np.zeros((num, 3 * 4), dtype=np.float32)
    for i in range(num):
        #print('path{} ={} '.format(i,path[i]))
        f_l = open(path[i][0], 'r')
        line_l = f_l.readlines()
        l_k[i] = (np.array(line_l[0].replace('\n', '').split(' '))[:3 * 3]).astype(np.float32)
        l_r[i] = (np.array(line_l[1].replace('\n', '').split(' '))[:3 * 4]).astype(np.float32)
        #print('f_l={}'.format(f_l))
        f_m = open(path[i][1], 'r')
        line_m = f_m.readlines()
        m_k[i] = (np.array(line_m[0].replace('\n', '').split(' '))[:3 * 3]).astype(np.float32)
        m_r[i] = (np.array(line_m[1].replace('\n', '').split(' '))[:3 * 4]).astype(np.float32)
        #print('f_m={}'.format(f_m))

        f_r = open(path[i][2], 'r')
        #print('f_r={}'.format(f_r))
        line_r = f_r.readlines()
        r_k[i] = (np.array(line_r[0].replace('\n', '').split(' '))[:3 * 3]).astype(np.float32)
        r_r[i] = (np.array(line_r[1].replace('\n', '').split(' '))[:3 * 4]).astype(np.float32)

        f_l.close()
        f_m.close()
        f_r.close()

    return l_k, l_r, m_k, m_r, r_k, r_r

def meshgrid(num, height, width):
  """Construct a 2D meshgrid.

  Args:
    batch: batch size
    height: height of the grid
    width: width of the grid
    is_homogeneous: whether to return in homogeneous coordinates
  Returns:
    x,y grid coordinates [batch, 2 (3 if homogeneous), height, width]
  """
  x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                  tf.transpose(tf.expand_dims(
                      tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
  y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                  tf.ones(shape=tf.stack([1, width])))
  x_t = (x_t + 1.0) * 0.5 * tf.cast(width - 1, tf.float32)
  y_t = (y_t + 1.0) * 0.5 * tf.cast(height - 1, tf.float32)
  ones = tf.ones_like(x_t)
  coords = tf.stack([x_t, y_t, ones], axis=0)
  coords = tf.tile(tf.expand_dims(coords, 0), [num, 1, 1, 1])
  return coords

def pixel_diff(src_pixel_coords,height, width):
    src_pixel_coords=tf.cast(src_pixel_coords, 'float32')
    coords_x, coords_y = tf.split(src_pixel_coords, [1, 1], axis=3)
    safe_x = tf.clip_by_value(coords_x, 0, width - 1)
    safe_y = tf.clip_by_value(coords_y, 0, height - 1)
    safe_pixel = tf.concat([safe_x, safe_y], 3)
    coord_diff = tf.abs(src_pixel_coords - safe_pixel)

    print("coord={}".format(coord_diff))

    return coord_diff

def pixel2cam(depth, pixel_coords, target_intrinsics):
    """Transforms coordinates in the pixel frame to the camera frame.
    Args:
      depth: [flip, height, width,c]
      pixel_coords: homogeneous pixel coordinates [3, height, width]
      intrinsics: camera intrinsics [num,3, 3] side[0]=target, side[1]=src
    Returns:
      Coords in the camera frame [3,height, width]
    """
    num, height, width,_ = depth.get_shape().as_list()
    ## test mode
    # flip_depth = tf.reverse(depth[1],[1])
    # depth = tf.concat([depth[0],flip_depth],0)
    depth = tf.reshape(depth,[num,1, -1])
    pixel_coords = tf.reshape(pixel_coords, [num,3, -1])  ## pixel_coord[0][0] = [X,Y,1]
    #target_intrinsic = tf.tile(tf.expand_dims(intrinsics[0], 0), [num, 1, 1])
    cam_coords = tf.matmul(tf.matrix_inverse(target_intrinsics),
                           pixel_coords * depth)  ## camera_coordin = k^-1(3x3) * p(3x1) * z(1x1) = (3x1)
    cam_coords = tf.reshape(cam_coords, [num,-1, height, width])
    return cam_coords

def rot_from_axisangle(vec):
    angle = tf.norm(vec,axis=2,keepdims=True)
    axis = vec / (angle + 1e-7)
    ca = tf.cos(angle)
    sa = tf.sin(angle)
    C = 1-ca

    x = tf.expand_dims(axis[..., 0], 1)
    y = tf.expand_dims(axis[..., 1], 1)
    z = tf.expand_dims(axis[..., 2], 1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot_0 = tf.concat([x * xC + ca,xyC-zs,zxC+ys],2)
    rot_1 = tf.concat([xyC+zs,y*yC+ca,yzC-xs],2)
    rot_2 = tf.concat([zxC - ys,yzC + xs,z * zC + ca],2)
    rot = tf.concat([rot_0,rot_1,rot_2],1)

    return rot

def cam2world(cam_coords, target_pose):
  """Transforms coordinates in the pixel frame to the camera frame.

  Args:
    cam_coords: homogeneous pixel coordinates [3, height, width]
    pose: camera extrinsic [side, 3, 4]
  Returns:
    Coords in the world [4, height, width]
  """
  num,axis,height, width = cam_coords.get_shape().as_list()
  rotation = tf.slice(target_pose, [0, 0, 0], [num, 3, 3])
  translation = tf.slice(target_pose, [0, 0, 3], [num, 3, 1])
  #translation=tf.expand_dims(translation, 1)

  #target_rotation=tf.concat([rotation,rotation],0)
  #target_rotation = tf.tile(tf.expand_dims(rotation, 0), [num, 1, 1])
  #targer_translation = tf.tile(tf.expand_dims(translation, 0), [num, 1, 1])

  cam_coords = tf.reshape(cam_coords, [num,axis, -1])
  cam_m = cam_coords - translation
  world_coords = tf.matmul(tf.matrix_inverse(rotation), cam_m) ## camera_coordin = k^-1(3x3) * p(3x1) * z(1x1) = (3x1)

  ones = tf.ones([num,1, height*width])
  world_coords = tf.concat([world_coords, ones], axis=1) # 3x1 -> 4x1
  world_coords = tf.reshape(world_coords, [num,-1, height, width])

  return world_coords

def bilinear_sampler(imgs, coords):
  """Construct a new image by bilinear sampling from the input image.

  Points falling outside the source image boundary have value 0.

  Args:
    imgs: source image to be sampled from [batch, height_s, width_s, channels]
    coords: coordinates of source pixels to sample from [batch, height_t,
      width_t, 2]. height_t/width_t correspond to the dimensions of the output
      image (don't need to be the same as height_s/width_s). The two channels
      correspond to x and y coordinates respectively.
  Returns:
    A new sampled image [batch, height_t, width_t, channels]
  """
  def _repeat(x, n_repeats):
    rep = tf.transpose(
        tf.expand_dims(tf.ones(shape=tf.stack([
            n_repeats,
        ])), 1), [1, 0])
    rep = tf.cast(rep, 'float32')                 ## rep = [1,1,1,1,1....] n_repeats 만큼, shape = [1,n_repeats]
    x = tf.matmul(tf.reshape(x, (-1, 1)), rep)    ## x= [[0,0,0,0,0....],[h*w,h*w,h*w,h*w,h*w,h*w],[...],[(b-1)*h*w,...]] shape = [(b-1),n_repeats]
    return tf.reshape(x, [-1])                    ## x= [0,0,0,0,h*w,h*w,h*w,h*w,...,(b-1)*h*w,(b-1)*h*w,(b-1)*h*w,(b-1)*h*w]

  with tf.name_scope('image_sampling'):
    coords_x, coords_y = tf.split(coords, [1, 1], axis=3)  ## coords_x = [b,h,w,1] , coords_y = [b,h,w,1]
    inp_size = imgs.get_shape()                            ## imgs = [b,h,w,3]
    coord_size = coords.get_shape()                        ## coords_ = [b,h,w,2]
    out_size = coords.get_shape().as_list()
    out_size[3] = imgs.get_shape().as_list()[3]            ## out = [b,h,w,3]  / 굳이 이짓거리 왜하는지 모르겠음

    coords_x = tf.cast(coords_x, 'float32')
    coords_y = tf.cast(coords_y, 'float32')

    x0 = tf.floor(coords_x)                                ## shape = [b,h,w,1]
    x1 = x0 + 1
    y0 = tf.floor(coords_y)
    y1 = y0 + 1

    y_max = tf.cast(tf.shape(imgs)[1] - 1, 'float32') ## y_max = h-1
    x_max = tf.cast(tf.shape(imgs)[2] - 1, 'float32') ## x_max = w-1
    zero = tf.zeros([1], dtype='float32')

    x0_safe = tf.clip_by_value(x0, zero, x_max)     ## 0보다 작은 값은 0으로, width 값보다 큰건 width로
    y0_safe = tf.clip_by_value(y0, zero, y_max)     ## shape =[b,h,w,1]
    x1_safe = tf.clip_by_value(x1, zero, x_max)
    y1_safe = tf.clip_by_value(y1, zero, y_max)

    ## bilinear interp weights, with points outside the grid having weight 0
    # wt_x0 = (x1 - coords_x) * tf.cast(tf.equal(x0, x0_safe), 'float32')
    # wt_x1 = (coords_x - x0) * tf.cast(tf.equal(x1, x1_safe), 'float32')
    # wt_y0 = (y1 - coords_y) * tf.cast(tf.equal(y0, y0_safe), 'float32')
    # wt_y1 = (coords_y - y0) * tf.cast(tf.equal(y1, y1_safe), 'float32')

    wt_x0 = x1_safe - coords_x    # 오른쪽 픽셀과 왼쪽 픽셀 중 어느거에 더 가까운지
    wt_x1 = coords_x - x0_safe    #
    wt_y0 = y1_safe - coords_y    # 위쪽 픽셀과 아래쪽 픽셀 중 어느거에 더 가까운지
    wt_y1 = coords_y - y0_safe    #

    ## indices in the flat image to sample from
    dim2 = tf.cast(inp_size[2], 'float32')                  ## dim2 = w
    dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32')    ## dim1 = h*w
    base = tf.reshape(
        _repeat(                                                ## tf.range(x) = [0,1,2,...,x-1]
            tf.cast(tf.range(coord_size[0]), 'float32') * dim1, ## range(b) = [0,1,2,...,b-1] range*dim1 = [0,h*w,2*h*w,...(b-1)*h*w]
            coord_size[1] * coord_size[2]),                     ## x,n_repeats // ,n_repeats = h*w
        [out_size[0], out_size[1], out_size[2], 1])         ## shape = [b,h,w,1]

    base_y0 = base + y0_safe * dim2                         ## shape = [b,h,w,1]
    base_y1 = base + y1_safe * dim2                         ## shape = [b,h,w,1]
    idx00 = tf.reshape(x0_safe + base_y0, [-1])           ## [b*h*w]
    idx01 = x0_safe + base_y1                             ## shape = [b,h,w,1] opencv 처럼 index tum 알려준거    idx00 = x0_safe + base_y0
    idx10 = x1_safe + base_y0                             ## shape = [b,h,w,1]
    idx11 = x1_safe + base_y1                             ## shape = [b,h,w,1]

    ## sample from imgs
    imgs_flat = tf.reshape(imgs, tf.stack([-1, inp_size[3]]))                     ## [b*h*w,3]
    imgs_flat = tf.cast(imgs_flat, 'float32')                                     ##
    im00 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx00, 'int32')), out_size)    ## gather의 outshape은 [b*w*h,3], [idx.shape,3]
    im01 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx01, 'int32')), out_size)
    im10 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx10, 'int32')), out_size)
    im11 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx11, 'int32')), out_size)

    w00 = wt_x0 * wt_y0
    w01 = wt_x0 * wt_y1
    w10 = wt_x1 * wt_y0
    w11 = wt_x1 * wt_y1

    output = tf.add_n([                         ## 위에 식을 보면 w00은 x_1,y_1과 기준의 차이임 0,1이 im reconstruction과 반대임
        w00 * im00, w01 * im01,
        w10 * im10, w11 * im11
    ])
    return output


def world2pixel(world_coords,proj_src):
    """
    :param world_coords: Coords in the world [4, height, width]
    :param proj_src:projection matrix of src image [3, 4]
    :return: Pixel coordinates projected from the camera frame [flip, height, width, 2]
    """
    num,axis,height,width = world_coords.get_shape().as_list()
    world_coords = tf.reshape(world_coords,[num,4,-1])
    #proj_src=tf.tile(tf.expand_dims(proj_src, 0), [num, 1, 1])

    unnormalized_pixel_coords = tf.matmul(proj_src,world_coords)
    x_u = tf.slice(unnormalized_pixel_coords, [0,0, 0],
                   [-1,1, -1])  ## x_u = unormalized pixel[b][0][w*h], shape = [1][w*h]
    y_u = tf.slice(unnormalized_pixel_coords, [0,1, 0], [-1,1, -1])
    z_u = tf.slice(unnormalized_pixel_coords, [0,2, 0], [-1,1, -1])

    x_n = (x_u / (z_u + 1e-10))
    y_n = (y_u / (z_u + 1e-10)) # shape = [b][1][w*h]

    src_pixel_coords = tf.concat([x_n, y_n], axis=1)  # shape = [b][2][w*h]
    src_pixel_coords = tf.reshape(src_pixel_coords, [num,2, height, width])
    return tf.transpose(src_pixel_coords, perm=[0, 2, 3, 1]) # 전치 행렬




def projective_inverse_warp(img,depth,target_pose,target_intrinsics,src_pose,src_intrinsics):
    """
    :param img: [fliped, height_s, width_s, c=3] src image
    :param depth: [fliped, height_s, width_s, c=1] target depth
    :param pose: [side,3,4] [0] = target pose, src pose
    :param intrinsics: [side, 3, 3] = target intrinsic, src intrinsic
    :return: predict_image: [2,height_s,width_s,c]
    """
    num, height, width, _ = img.get_shape().as_list()



    target_intrinsics=tf.concat(
        [[tf.expand_dims(tf.concat([target_intrinsics[i][0]*width, target_intrinsics[i][1]*height, target_intrinsics[i][2]], 0),0)] for i in range(num)],
        0)
    target_intrinsics=tf.reshape(target_intrinsics,[num,3,3])
    src_intrinsics =tf.concat(
        [[tf.expand_dims(tf.concat([src_intrinsics[i][0]*width, src_intrinsics[i][1]*height, src_intrinsics[i][2]], 0),0)] for i in range(num)],
        0)
    src_intrinsics=tf.reshape(src_intrinsics,[num,3,3])

    # Construct pixel grid coordinates
    pixel_coords = meshgrid(num,height,width)


    cam_coords = pixel2cam(depth, pixel_coords, target_intrinsics)  ## pixel_coord to camera_coordin

    world_coords = cam2world(cam_coords, target_pose)

    proj_src = tf.matmul(src_intrinsics, src_pose)

    src_pixel_coords = world2pixel(world_coords, proj_src)

    coord_diff = pixel_diff(src_pixel_coords, height, width)
    output_img = bilinear_sampler(img, src_pixel_coords)

    return output_img,coord_diff

def generate_mask(depth_src,depth_warping):
    #return tf.greater_equal(depth_src,depth_warping) #sgreater
    return tf.greater_equal(depth_warping, depth_src-0.05)
    #return tf.greater_equal(depth_warping,depth_src-0.05) # r->m >= m-0.05 (tgreater)

def mask_loss(mean):

    b = tf.cast(tf.reduce_sum(tf.exp(np.linspace(0,5,100))),dtype=tf.float32)
    return tf.where(tf.greater_equal(mean,tf.ones_like(mean)*0.5),(tf.exp(tf.clip_by_value(mean, 0, 1) * 5) / b) * 20,tf.zeros_like(mean))


def trans_kitti(world_coords,proj):
    num, axis, height, width = world_coords.get_shape().as_list()
    world_coords = tf.reshape(world_coords, [num, 4, -1])

    unnormalized_pixel_coords = tf.matmul(proj, world_coords)

    z_u = tf.slice(unnormalized_pixel_coords, [0, 2, 0], [-1, 1, -1])

    n_depth = tf.reshape(z_u, [num, height, width, 1])

    return n_depth

def projective_inverse_warp_depth(depth1,depth2,pose_1,pose_2,intrinsic_1,intrinsic_2):
    """
    :param depth1:
    :param depth2:
    :param pose_1:
    :param pose_2:
    :param intrinsic_1:
    :param intrinsic_2:
    :return:
    """
    num, height, width, _ = depth1.get_shape().as_list()
    intrinsic_1 = tf.concat(
        [[tf.expand_dims(
            tf.concat([intrinsic_1[i][0] * width, intrinsic_1[i][1] * height, intrinsic_1[i][2]], 0),
            0)] for i in range(num)],
        0)
    intrinsic_1 = tf.reshape(intrinsic_1, [num, 3, 3])
    intrinsic_2 = tf.concat(
        [[tf.expand_dims(
            tf.concat([intrinsic_2[i][0] * width, intrinsic_2[i][1] * height, intrinsic_2[i][2]], 0), 0)] for i
         in range(num)],
        0)
    intrinsic_2 = tf.reshape(intrinsic_2, [num, 3, 3])

    proj_1 = tf.matmul(intrinsic_1,pose_1)
    proj_2 = tf.matmul(intrinsic_2,pose_2)

    # 1_to_2
    pixel_coords_1 = meshgrid(num,height,width)
    cam_coords_1 = pixel2cam(depth1,pixel_coords_1, intrinsic_1)
    world_coords_1 = cam2world(cam_coords_1, pose_1)

    pixel_coords_2 = meshgrid(num, height, width)
    cam_coords_2 = pixel2cam(depth2, pixel_coords_2, intrinsic_2)
    world_coords_2 = cam2world(cam_coords_2, pose_2)

    depth_1to2, src_pixel_coords_2to1 = depth_warping_and_world2pixel(world_coords_1, proj_2)
    depth_2to1, src_pixel_coords_1to2 = depth_warping_and_world2pixel(world_coords_2, proj_1)

    depth_est1 = bilinear_sampler(depth_2to1, src_pixel_coords_2to1)
    depth_est2 = bilinear_sampler(depth_1to2, src_pixel_coords_1to2)

    mask2_to_1 = generate_mask(depth1,depth_est1)
    mask1_to_2 = generate_mask(depth2,depth_est2)

    return depth_est1,depth_est2,mask2_to_1,mask1_to_2

def depth_warping_and_world2pixel(world_coords,proj):
    num, axis, height, width = world_coords.get_shape().as_list()
    world_coords = tf.reshape(world_coords, [num, 4, -1])

    unnormalized_pixel_coords = tf.matmul(proj, world_coords)
    x_u = tf.slice(unnormalized_pixel_coords, [0, 0, 0],
                   [-1, 1, -1])  ## x_u = unormalized pixel[b][0][w*h], shape = [1][w*h]
    y_u = tf.slice(unnormalized_pixel_coords, [0, 1, 0], [-1, 1, -1])
    z_u = tf.slice(unnormalized_pixel_coords, [0, 2, 0], [-1, 1, -1])

    n_depth = tf.reshape(z_u,[num,height,width,1])
    x_n = (x_u / (z_u + 1e-10))
    y_n = (y_u / (z_u + 1e-10))  # shape = [b][1][w*h]

    src_pixel_coords = tf.concat([x_n, y_n], axis=1)  # shape = [b][2][w*h]
    src_pixel_coords = tf.reshape(src_pixel_coords, [num, 2, height, width])
    src_pixel_coords = tf.transpose(src_pixel_coords, perm=[0, 2, 3, 1])

    return n_depth,src_pixel_coords

