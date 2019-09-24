from __future__ import absolute_import, division, print_function
from collections import namedtuple

#TODO: disparity loss, 갯수, 상수값, 구현방식
#TODO: ssim+recon * rand* 0.00001
#TODO: total_loss / num_Scale
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib as tfc
import tensorflow.contrib.layers as layers
from util import *
arg_scope = tf.contrib.framework.arg_scope
monodepth_parameters = namedtuple('parameters',
                        'encoder, '
                        'height, width, '
                        'batch_size, '
                        'num_threads, '
                        'num_epochs, '
                        'do_stereo, '
                        'wrap_mode, '
                        'use_deconv, '
                        'alpha_image_loss, '
                        'disp_gradient_loss_weight, '
                        'lr_loss_weight, '
                        'full_summary, '
                        'parameter_path, '
                        'do_change, '
                        'input_type, '
                        'mask_loss_weight ')

class MonodepthModel(object):
    """monodepth model"""

    def __init__(self,params,mode,reference,left=None,right=None,param_path=None,reuse_variables=None,model_index=0):
        self.params = params

        self.mode = mode
        if mode == 'train':
            self.param_mask_loss_weight = tf.reshape(self.params.mask_loss_weight, [1])
            self.param_lr_loss_weight = tf.reshape(self.params.lr_loss_weight, [1])
            self.is_training =True
        elif mode == 'test':
            self.is_training =False
        elif mode == 'eval':
            self.param_mask_loss_weight = tf.reshape(self.params.mask_loss_weight, [1])
            self.param_lr_loss_weight = tf.reshape(self.params.lr_loss_weight, [1])
            self.is_training = False
        elif mode == 'kitti_eval':
            self.param_mask_loss_weight = tf.reshape(self.params.mask_loss_weight, [1])
            self.param_lr_loss_weight = tf.reshape(self.params.lr_loss_weight, [1])
            self.is_training = True
        else:
            print('mode is undecleared')
            return

        if param_path==None:
            print("None 뜬다")

        ## image
        if mode == 'train':
            self.left = left
            self.right = right
            self.reference = reference
        else:
            self.left = reference
            self.right = reference
            self.reference = reference

        ## cam_param
        self.param_path = param_path

        if self.param_path is not None:
            self.left_intrinsic, self.left_pose, self.reference_intrinsic, self.reference_pose, self.right_intrinsic, self.right_pose = self.load_kocca_param(self.param_path)

        self.model_collection = ['model_'+str(model_index)]

        self.reuse_variables = reuse_variables
        self.MAX_SCALE = 50
        self.MIN_DISP = 0.1

        self.epoch=False

        if self.mode:
            self.get_weight()

            self.left_to_reference_est=[]
            self.right_to_reference_est = []

            self.pose_left_to_reference_est=[]
            self.pose_right_to_reference_est = []



        #self.build_model()
        self.build_outputs()
        if self.mode is not 'train':
            print("return")
            return

        self.build_losses()
    ################################################################################################
    ## param function
    def load_kitti_param(self,path):
        print(path)
        l_k,l_r,rect = tf.py_func(load_kitti,[path],[tf.float32, tf.float32, tf.float32])
        num, side = path.get_shape().as_list()
        l_k = tf.reshape(l_k, [num, 3, 3])
        l_r = tf.reshape(l_r, [num, 3, 4])
        rect = tf.reshape(rect,[num,4,4])

        return l_k,l_r,rect

    def load_kocca_param(self, path):
        l_k, l_r, m_k, m_r,  r_k, r_r = tf.py_func(load_kocca_param, [path],[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])
        num, side = path.get_shape().as_list()

        l_k = tf.reshape(l_k, [num, 3, 3])
        r_k = tf.reshape(r_k, [num, 3, 3])
        m_k = tf.reshape(m_k, [num, 3, 3])

        l_r = tf.reshape(l_r, [num, 3, 4])
        r_r = tf.reshape(r_r, [num, 3, 4])
        m_r = tf.reshape(m_r, [num, 3, 4])

        return l_k, l_r, m_k, m_r, r_k, r_r

    #################################################################################################
    ## train function
    def gradient_x(self, img): #TODO
        gx = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gx

    def gradient_y(self, img): #TODO
        gy = img[:, :-1, :, :] - img[:, 1:, :, :]
        return gy

    def generate_image(self,img, depth, target_pose, target_intrinsics, src_pose, src_intrinsics):
        img,coord = projective_inverse_warp(img,depth,target_pose,target_intrinsics,src_pose,src_intrinsics)
        return img,coord

    def generate_depth_and_mask(self,depth_1,pose_1,intrinsics_1,depth_2,pose_2,intrinsics_2):
        depth_est1,depth_est2,mask_est1,mask_est2 = projective_inverse_warp_depth(depth_1,depth_2,pose_1,pose_2,intrinsics_1,intrinsics_2)
        return depth_est1,depth_est2,mask_est1,mask_est2

    ####################################################################################
    # up resize function
    def upsample_nn(self, x, ratio):
        s = tf.shape(x)
        h = s[1]
        w = s[2]
        return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])

    def get_disp(self, x,scope):
        disp = self.conv(x, 2, 3, 1,scope,activation_fn= tf.nn.sigmoid) ## 0~0.3 값을 가짐 , shape 은 input=output, but channel = 2
        return disp

    def get_disp_stereo(self, x,scope):
        disp = self.conv(x, 2, 3, 1,scope, activation_fn=tf.nn.sigmoid) ## 0~0.3 값을 가짐 , shape 은 input=output, but channel = 2
        return disp

    def get_disp_mono(self, x,scope,reuse):
        disp = self.conv(x, 1, 3, 1,scope,reuse,activation_fn= tf.nn.sigmoid) ## 0~0.3 값을 가짐 , shape 은 input=output, but channel = 2
        return disp

    def disp_to_depth(self, disp):
        """Convert network's sigmoid output into depth prediction
        The formula for this conversion is given in the 'additional considerations'
        section of the paper.
        """
        min_disp = 1 / self.MAX_SCALE
        max_disp = 1 / self.MIN_DISP
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1 / scaled_disp
        return scaled_disp, depth

    ###############################################################################################
    # layer convenience function
    def conv(self, x, num_out_layers, kernel_size, stride,scope,reuse,activation_fn=tf.nn.elu):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]]) ## shape = (2, height,width,3)로 들어옴, height와 width에 양쪽으로 p만큼 padding을 하는 line
        re = slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=scope,reuse=reuse)
        print(re)
        return re

    def conv2d(self, x, num_out_layers, kernel_size, stride,scope,  reuse,activation_fn=None):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p],
                         [0, 0]])  ## shape = (2, height,width,3)로 들어옴, height와 width에 양쪽으로 p만큼 padding을 하는 line
        re = slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn,
                           weights_initializer=tf.contrib.layers.xavier_initializer(),scope=scope,reuse=reuse)
        print(re)
        return re

    def maxpool(self, x, kernel_size):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]]) ## 여기서도 마찬가지로 height,width에만 padding이 들어가게끔 함
        return slim.max_pool2d(p_x, kernel_size)

    def resblock_basic(self,x,num_layer,num_blocks,scope,reuse,down_sample = True,is_training=False):
        for i in range(num_blocks):
            identity = x
            if down_sample == True:
                conv1 = self.conv2d(x,num_layer,3,2,scope+'/conv_{}_1'.format(i),reuse)
                identity = self.conv2d(identity,num_layer,1,2,scope+'/identity_{}'.format(i),reuse)
                identity = self.batch_norm(identity,scope+'/identity_batch_{}'.format(i),reuse,is_training)
                down_sample = False
            else:
                conv1 = self.conv2d(x,num_layer,3,1,scope+'/conv_{}_1'.format(i),reuse)
            bn1 = self.batch_norm(conv1,scope+'/batch_{}_1'.format(i),reuse,is_training)
            relu1 = tf.nn.relu(bn1)

            conv2 = self.conv2d(relu1,num_layer,3,1,scope+'/conv_{}_2'.format(i),reuse)
            bn2 = self.batch_norm(conv2,scope+'/batch_{}_2'.format(i),reuse,is_training)

            x = tf.nn.relu(bn2+identity)
        return x

    def resblock_no_bn(self, x, num_layer, num_blocks,scope,reuse, down_sample=True, is_training=False ):
        for i in range(num_blocks):
            identity = x
            if down_sample == True:
                conv1 = self.conv2d(x, num_layer, 3, 2,scope+'/conv_{}_1'.format(i),reuse)
                identity = self.conv2d(identity, num_layer, 1, 2,scope+'/identity_{}'.format(i),reuse)
                down_sample = False
            else:
                conv1 = self.conv2d(x, num_layer, 3, 1,scope+'/conv_{}_1'.format(i),reuse)
            relu1 = tf.nn.relu(conv1)

            conv2 = self.conv2d(relu1, num_layer, 3, 1,scope+'/conv_{}_2'.format(i),reuse)

            x = tf.nn.relu(conv2 + identity)
        return x

    def batch_norm(self,x,scope,reuse,is_training=False):
        return tfc.layers.batch_norm(x,decay=0.1,epsilon=1e-05,scale=True,scope=scope,reuse=reuse,is_training=is_training)

    def upconv_mono2(self, x, num_out_layers, kernel_size, scale,scope,reuse):
        conv = self.conv(x, num_out_layers, kernel_size, 1,scope,reuse)
        upsample = self.upsample_nn(conv, scale)
        return upsample

    # def upconv(self, x, num_out_layers, kernel_size, scale):
    #     upsample = self.upsample_nn(x, scale)
    #     conv = self.conv(upsample, num_out_layers, kernel_size, 1)
    #     return conv
    #
    # def deconv(self, x, num_out_layers, kernel_size, scale):
    #     p_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
    #     conv = slim.conv2d_transpose(p_x, num_out_layers, kernel_size, scale, 'SAME',weights_initializer=tf.contrib.layers.xavier_initializer())
    #     return conv[:,3:-1,3:-1,:]

    #########################################################################################
    # backbone
    def cat_pose_input(self,left_feature,right_feature,reuse):
        with tf.variable_scope('pose_cat'):
            left = self.conv2d(left_feature,256,1,1,'conv',reuse,activation_fn=tf.nn.relu)
            right = self.conv2d(right_feature, 256, 1, 1, 'conv', True, activation_fn=tf.nn.relu)
            cat_feature = tf.concat([left,right],3)

        return cat_feature

    def pose_from_out(self, angle, translation,inv_angle,inv_translation):
        num, _, _ = angle.get_shape().as_list()
        # 1 pose
        rot = rot_from_axisangle(angle)
        pose = tf.concat([rot, translation], 2)

        inv_rot = rot_from_axisangle(inv_angle)
        pose_diff = tf.reduce_mean(tf.abs(rot-inv_rot))+tf.reduce_mean(tf.abs(translation-inv_translation))

        return pose,pose_diff

    def get_reference_train_pose(self,angle1,translation1):
        num, _, _ = angle1.get_shape().as_list()

        iden_R = tf.tile(tf.expand_dims(tf.eye(3, 3), 0), [num, 1, 1])
        zero_t = tf.zeros_like(translation1)
        refer_pose = tf.concat([iden_R,zero_t],2)
        return refer_pose

    # def pose_from_out(self,angle1,translation1,angle2,translation2):
    #     num, _, _ = angle1.get_shape().as_list()
    #
    #     # 1 pose
    #     rot_1 = rot_from_axisangle(angle1)
    #     pose_1 = tf.concat([rot_1, translation1], 2)
    #     rot_2 = rot_from_axisangle(angle2)
    #     pose_2 = tf.concat([rot_2, translation2], 2)
    #
    #     # reference pose(identity)
    #     iden_R = tf.tile(tf.expand_dims(tf.eye(3,3),0),[num,1,1])
    #     zero_t = tf.zeros_like(translation1)
    #     refer_pose = tf.concat([iden_R,zero_t],2)
    #
    #     return refer_pose,pose_1,pose_2

    def res_encoder(self,color,reuse):
        #block = self.resblock_no_bn
        block = self.resblock_basic
        conv = self.conv2d
        print(self.is_training)
        with tf.variable_scope('encoder',reuse=reuse):
            conv1 = conv(color, 64, 7, 2,'conv1',reuse)  # H/2  -   64D
            relu = tf.nn.relu(conv1)
            pool1 = self.maxpool(relu, 3)  # H/4  -   64D
            res_block1 = block(pool1, 64, 2,'res_block1', reuse,down_sample=False, is_training=self.is_training)  # H/8  -  256D
            res_block2 = block(res_block1, 128, 2,'res_block2',reuse, down_sample=True, is_training=self.is_training)
            res_block3 = block(res_block2, 256, 2, 'res_block3',reuse,down_sample=True, is_training=self.is_training)
            res_block4 = block(res_block3, 512, 2,'res_block4',reuse, down_sample=True, is_training=self.is_training)

        print('relu={}'.format(relu))
        print('res_block1={}'.format(res_block1))
        print('res_block2={}'.format(res_block2))
        print('res_block3={}'.format(res_block3))
        print('res_block4={}'.format(res_block4))

        return [relu,res_block1,res_block2,res_block3,res_block4]

    def create_scales(self,constraint_minimum,reuse):
        """Creates variables representing rotation and translation scaling factors.

        Args:
          constraint_minimum: A scalar, the variables will be constrained to not fall
            below it.

        Returns:
          Two scalar variables, rotation and translation scale.
        """

        def constraint(x):
            return tf.nn.relu(x - constraint_minimum) + constraint_minimum

        with tf.variable_scope('Scales', initializer=0.01, constraint=constraint,reuse=reuse):
            rot_scale = tf.get_variable('rotation')
            trans_scale = tf.get_variable('translation')

        return rot_scale, trans_scale

    def add_intrinsics_head(self,bottleneck, image_height, image_width,reuse):
        """Adds a head the preficts camera intrinsics.

        Args:
          bottleneck: A tf.Tensor of shape [B, 1, 1, C], typically the bottlenech
            features of a netrowk.
          image_height: A scalar tf.Tensor or an python scalar, the image height in
            pixels.
          image_width: A scalar tf.Tensor or an python scalar, the image width in
            pixels.

        image_height and image_width are used to provide the right scale for the focal
        length and the offest parameters.

        Returns:
          a tf.Tensor of shape [B, 3, 3], and type float32, where the 3x3 part is the
          intrinsic matrix: (fx, 0, x0), (0, fy, y0), (0, 0, 1).
        """
        with tf.variable_scope('CameraIntrinsics'):
            # Since the focal lengths in pixels tend to be in the order of magnitude of
            # the image width and height, we multiply the network prediction by them.

            focal_lengths = tf.squeeze(
                layers.conv2d(
                    bottleneck,
                    2, [1, 1],
                    stride=1,
                    activation_fn=tf.nn.softplus,
                    weights_regularizer=None,
                    scope='foci',
                    reuse=reuse),
                axis=(1, 2))# * tf.to_float(tf.convert_to_tensor([[image_width, image_height]]))

            # The pixel offsets tend to be around the center of the image, and they
            # are typically a fraction the image width and height in pixels. We thus
            # multiply the network prediction by the width and height, and the
            # additional 0.5 them by default at the center of the image.
            offsets = (tf.squeeze(
                layers.conv2d(
                    bottleneck,
                    2, [1, 1],
                    stride=1,
                    activation_fn=None,
                    weights_regularizer=None,
                    biases_initializer=None,
                    scope='offsets',
                    reuse=reuse),
                axis=(1, 2)) + 0.5) #* tf.to_float(tf.convert_to_tensor([[image_width, image_height]]))

            foci = tf.linalg.diag(focal_lengths)

            intrinsic_mat = tf.concat([foci, tf.expand_dims(offsets, -1)], axis=2)
            batch_size = tf.shape(bottleneck)[0]
            last_row = tf.tile([[[0.0, 0.0, 1.0]]], [batch_size, 1, 1])
            intrinsic_mat = tf.concat([intrinsic_mat, last_row], axis=1)

            return intrinsic_mat



    def pose_motion_net(self,images, reuse,weight_reg=0.0):
            """Predict object-motion vectors from a stack of frames or embeddings.

            Args:
              images: Input tensor with shape [B, h, w, 2c], containing two
                depth-concatenated images.
              weight_reg: A float scalar, the amount of weight regularization.

            Returns:
              A tuple of 3 tf.Tensors:
              rotation: [B, 3], global rotation angles (due to camera rotation).
              translation: [B, 1, 1, 3], global translation vectors (due to camera
                translation).
              residual_translation: [B, h, w, 3], residual translation vector field, due
                to motion of objects relatively to the scene. The overall translation
                field is translation + residual_translation.
            """

            with tf.variable_scope('MotionFieldNet'):
                with arg_scope([layers.conv2d],
                               weights_regularizer=layers.l2_regularizer(weight_reg),
                               activation_fn=tf.nn.relu):
                    print('images={}'.format(images))
                    conv1 = layers.conv2d(images, 16, [3, 3], stride=2, scope='Conv1',reuse=reuse)
                    print('conv1={}'.format(conv1))
                    conv2 = layers.conv2d(conv1, 32, [3, 3], stride=2, scope='Conv2',reuse=reuse)
                    print('conv2={}'.format(conv2))
                    conv3 = layers.conv2d(conv2, 64, [3, 3], stride=2, scope='Conv3',reuse=reuse)
                    print('conv3={}'.format(conv3))
                    conv4 = layers.conv2d(conv3, 128, [3, 3], stride=2, scope='Conv4',reuse=reuse)
                    print('conv4={}'.format(conv4))
                    conv5 = layers.conv2d(conv4, 256, [3, 3], stride=2, scope='Conv5',reuse=reuse)
                    print('conv5={}'.format(conv5))
                    conv6 = layers.conv2d(conv5, 512, [3, 3], stride=2, scope='Conv6',reuse=reuse)
                    print('conv6={}'.format(conv6))
                    conv7 = layers.conv2d(conv6, 1024, [3, 3], stride=2, scope='Conv7',reuse=reuse)
                    print('conv7={}'.format(conv7))
                    bottleneck = tf.reduce_mean(conv7, axis=[1, 2], keepdims=True)
                    print('bottleneck={}'.format(bottleneck))
                    background_motion = layers.conv2d(
                        bottleneck,
                        6, [1, 1],
                        stride=1,
                        activation_fn=None,
                        biases_initializer=None,
                        scope='background_motion',
                        reuse = reuse)

                rotation = background_motion[:, 0, 0, :3]
                translation = background_motion[:, :, :, 3:]

                rot_scale, trans_scale = self.create_scales(0.001,reuse)
                translation *= trans_scale
                rotation *= rot_scale

                image_height, image_width = tf.unstack(tf.shape(images)[1:3])
                intrinsic_mat = self.add_intrinsics_head(bottleneck, image_height, image_width,reuse)
            rotation = tf.expand_dims(rotation,1)
            translation = tf.expand_dims(tf.squeeze(translation,axis=[1,2]),2)
            return (rotation, translation, intrinsic_mat)

    def depth_decoder(self,feature,reuse):
        skip1 = feature[0]
        skip2 = feature[1]
        skip3 = feature[2]
        skip4 = feature[3]
        skip5 = feature[4]

        upconv = self.upconv_mono2
        conv_elu = self.conv

        with tf.variable_scope('depth_decoder',reuse=reuse):
            upconv5 = upconv(skip5, 256, 3, 2,'upconv5',reuse)
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5 = conv_elu(concat5, 256, 3, 1,'iconv5',reuse)

            upconv4 = upconv(iconv5, 128, 3, 2,'upconv4',reuse)
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4 = conv_elu(concat4, 128, 3, 1,'iconv4',reuse)
            disp4 = self.get_disp_mono(iconv4,'disp4',reuse)

            upconv3 = upconv(iconv4, 64, 3, 2,'upconv3',reuse)
            concat3 = tf.concat([upconv3, skip2], 3)
            iconv3 = conv_elu(concat3, 64, 3, 1,'iconv3',reuse)
            disp3 = self.get_disp_mono(iconv3,'disp3',reuse)

            upconv2 = upconv(iconv3, 32, 3, 2,'upconv2',reuse)
            concat2 = tf.concat([upconv2, skip1], 3)
            iconv2 = conv_elu(concat2, 32, 3, 1,'iconv2',reuse)
            disp2 = self.get_disp_mono(iconv2,'disp2',reuse)

            upconv1 = upconv(iconv2, 16, 3, 2,'upconv1',reuse)
            iconv1 = conv_elu(upconv1, 16, 3, 1,'iconv1',reuse)
            disp1 = self.get_disp_mono(iconv1,'disp1',reuse)
            # print("disp=", self.disp1)

            _, depth4 = self.disp_to_depth(self.upsample_nn(disp4, 8))
            _, depth3 = self.disp_to_depth(self.upsample_nn(disp3, 4))
            _, depth2 = self.disp_to_depth(self.upsample_nn(disp2, 2))
            _, depth1 = self.disp_to_depth(disp1)

        print('disp1={}'.format(disp1))
        print('disp2={}'.format(disp2))
        print('disp3={}'.format(disp3))
        print('disp4={}'.format(disp4))

        return [disp1,disp2,disp3,disp4],[depth1,depth2,depth3,depth4]

    def build_outputs(self):
        with tf.variable_scope('out_put'):
            self.reference_feature = self.res_encoder(self.reference,self.reuse_variables)
            if self.mode == 'train':
                self.left_feature = self.res_encoder(self.left,True)
                self.right_feature = self.res_encoder(self.right, True)

            self.disp_reference_est, self.depth_reference_est = self.depth_decoder(self.reference_feature,self.reuse_variables)
            if self.mode =='test':
                return
            if self.mode is not 'train':
                print("return")
                return

            self.disp_left_est, self.depth_left_est = self.depth_decoder(self.left_feature,True)
            self.disp_right_est, self.depth_right_est = self.depth_decoder(self.right_feature, True)

            # angle,trans,intrin = self.pose_motion_net(tf.concat([self.reference,self.left],-1),self.reuse_variables)
            # inv_angle,inv_trans,intrin_inv = self.pose_motion_net(tf.concat([self.left,self.reference],-1),True)
            # print("angle= ",angle)
            # print("trans= ",trans)
            # self.left_intrinsic_train =  0.5 * (intrin + intrin_inv)
            # self.left_pose_train,self.left_pose_diff = self.pose_from_out(angle,trans,inv_angle,inv_trans)
            #
            # angle, trans, intrin = self.pose_motion_net(tf.concat([self.reference, self.right], -1),
            #                                             True)
            # inv_angle, inv_trans, intrin_inv = self.pose_motion_net(tf.concat([self.right, self.reference], -1),
            #                                                         True)
            #
            # self.right_intrinsic_train = 0.5 * (intrin + intrin_inv)
            # self.right_pose_train, self.right_pose_diff = self.pose_from_out(angle, trans, inv_angle, inv_trans)
            #
            # self.reference_pose_train = self.get_reference_train_pose(angle,trans)

        #######################################################################################################################################
    # GENERATE IMAGES
        with tf.variable_scope('images'):
            for i in range(4):
                Im,_= self.generate_image(self.left, self.depth_reference_est[i], self.reference_pose, self.reference_intrinsic,self.left_pose,self.left_intrinsic)
                self.left_to_reference_est.append(Im)

                Im, _ = self.generate_image(self.right, self.depth_reference_est[i], self.reference_pose, self.reference_intrinsic,
                                            self.right_pose, self.right_intrinsic)
                self.right_to_reference_est.append(Im)

                # Im, _ = self.generate_image(self.left, self.depth_reference_est[i], self.reference_pose_train,
                #                             self.left_intrinsic_train,
                #                             self.left_pose_train, self.left_intrinsic_train)
                # self.pose_left_to_reference_est.append(Im)
                # Im, _ = self.generate_image(self.right, self.depth_reference_est[i], self.reference_pose_train,
                #                             self.right_intrinsic_train,
                #                             self.right_pose_train, self.right_intrinsic_train)
                # self.pose_right_to_reference_est.append(Im)

        if self.mode == 'test':
            return


        # # DISPARITY SMOOTHNESS
        # with tf.variable_scope('smoothness'):
        #
        #     self.disp_reference_smoothness = self.get_disparity_smoothness_1scale(self.disp_reference_est[0], self.reference)
        #     self.disp_left_smoothness = self.get_disparity_smoothness_1scale(self.disp_left_est[0], self.left)
        #     self.disp_right_smoothness = self.get_disparity_smoothness_1scale(self.disp_right_est[0], self.right)

    ###################################################################################################################################
    ## LOSS
    def SSIM(self, x, y): #TODO
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = slim.avg_pool2d(x, 3, 1, 'SAME')
        mu_y = slim.avg_pool2d(y, 3, 1, 'SAME')

        sigma_x = slim.avg_pool2d(x ** 2, 3, 1, 'SAME') - mu_x ** 2
        sigma_y = slim.avg_pool2d(y ** 2, 3, 1, 'SAME') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y, 3, 1, 'SAME') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    def reprojection_loss(self,refer,target):
        SSIM_loss = tf.reduce_mean(self.SSIM(refer, target), 3)
        l1_loss =tf.reduce_mean(tf.abs(refer-target),3)
        reprojection_loss = self.params.alpha_image_loss*SSIM_loss+(1 - self.params.alpha_image_loss)*l1_loss
        return tf.expand_dims(reprojection_loss,1)

    def get_disparity_smoothness(self, disp, pyramid): #TODO
        up_disp = [self.upsample_nn(disp[i],2**i) for i in range(4)]
        mean_disp = [tf.reduce_mean(up_disp[i], axis=[1, 2, 3], keep_dims=True) for i in range(4)]
        disp_in = [up_disp[i] / mean_disp[i] for i in range(4)]

        disp_gradients_x = [self.gradient_x(disp_in[i]) for i in range(4)]
        disp_gradients_y = [self.gradient_y(disp_in[i]) for i in range(4)]


        image_gradients_x = [self.gradient_x(pyramid) for i in range(4)]
        image_gradients_y = [self.gradient_y(pyramid) for i in range(4)]

        weights_x = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_x]
        weights_y = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_y]

        smoothness_x = tf.add_n([tf.reduce_mean(tf.abs(disp_gradients_x[i] * weights_x[i])) for i in range(4)])
        smoothness_y = tf.add_n([tf.reduce_mean(tf.abs(disp_gradients_y[i] * weights_y[i])) for i in range(4)])

        return (smoothness_x+smoothness_y)/4

    def get_weight(self):
        if self.epoch:
            self.mask_loss_weight = self.params.mask_loss_weight
            self.lr_loss_weight = self.param_lr_loss_weight
        else:
            self.mask_loss_weight = 0
            self.lr_loss_weight =0

    def get_disparity_smoothness_1scale(self, disp, pyramid): #TODO

        disp_gradients_x = self.gradient_x(disp)
        disp_gradients_y = self.gradient_y(disp)

        image_gradients_x = self.gradient_x(pyramid)
        image_gradients_y = self.gradient_y(pyramid)

        weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_x), 3, keep_dims=True))
        weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_y), 3, keep_dims=True))

        smoothness_x = disp_gradients_x * weights_x
        smoothness_y = disp_gradients_y * weights_y
        smooth =[]
        smooth.append(smoothness_x)
        smooth.append(smoothness_y)

        return smooth


    def build_losses(self): #TODO
        with tf.variable_scope('losses', reuse=self.reuse_variables):
            self.reproject_loss_left = [self.reprojection_loss(self.reference,self.left_to_reference_est[i]) for i in range(4)]
            self.reproject_loss_right = [self.reprojection_loss(self.reference, self.right_to_reference_est[i]) for i in range(4)]

            #self.reproject_loss_pose_left = [self.reprojection_loss(self.reference, self.pose_left_to_reference_est[i]) for i in range(4)]
            #self.reproject_loss_pose_right = [self.reprojection_loss(self.reference, self.pose_right_to_reference_est[i]) for i in range(4)]

            self.origin_cat = tf.add_n([tf.reduce_mean(tf.reduce_min(tf.concat([self.reproject_loss_left[i],self.reproject_loss_right[i]],1),1,keepdims=True),axis=[1,2,3]) for i in range(4)])/4
            #self.pose_cat = tf.add_n([tf.reduce_mean(tf.reduce_min(tf.concat([self.reproject_loss_pose_left[i], self.reproject_loss_pose_right[i]], 1), 1,keepdims=True),axis=[1,2,3]) for i in range(4)])/4
            #self.origin_image_loss = tf.reduce_mean(self.origin_cat)
            #self.pose_image_loss = tf.reduce_mean(self.pose_cat)

            #print('self.origin_cat =',self.origin_cat)
            #print('self.pose_cat =', self.pose_cat)
            self.image_loss = tf.reduce_mean(self.origin_cat)
            #self.image_loss = tf.reduce_mean(tf.cond(tf.equal(self.epoch,True),lambda:tf.where(tf.greater(self.origin_cat,self.pose_cat),self.pose_cat,self.origin_cat),lambda:self.origin_cat))
            #self.pose_cond = tf.where(tf.greater(self.origin_cat,self.pose_cat),tf.ones_like(self.pose_cat),tf.zeros_like(self.origin_cat))
            print(' self.image_loss=', self.image_loss)
            #print('self.pose_cond = ', self.pose_cond)
            #self.image_loss = tf.reduce_mean(self.min_cat)
            #self.image_loss = self.min_cat

            # DISPARITY SMOOTHNESS
            self.disp_reference_loss = self.get_disparity_smoothness(self.disp_reference_est,self.reference)
            self.disp_left_loss = self.get_disparity_smoothness(self.disp_left_est,self.left)
            self.disp_right_loss = self.get_disparity_smoothness(self.disp_right_est,self.right)

            self.disp_gradient_loss = (self.disp_reference_loss + self.disp_left_loss+self.disp_right_loss)/3
            #self.pose_diff_loss = self.right_pose_diff+ self.left_pose_diff
            # TOTAL LOSS
            self.total_loss = self.image_loss + self.params.disp_gradient_loss_weight * self.disp_gradient_loss #+ self.pose_diff_loss*0.0001# + self.mask_loss*self.mask_loss_weight

    ###################################################################################################################################################