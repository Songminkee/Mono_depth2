from __future__ import absolute_import, division, print_function
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.slim as slim
from monodepth_model import *
from monodepth_dataloader import *
from average_gradients import *
import numpy as np
import glob

## TODO : flip, or multi, or minus


def mask_colored(mask):
    mask = tf.expand_dims(mask,-1)
    return tf.cast(tf.concat([tf.where(mask,tf.zeros_like(mask,dtype='float32'),tf.ones_like(mask,dtype='float32')),tf.zeros_like(mask,dtype='float32'),tf.zeros_like(mask,dtype='float32')],-1),'float32')

def count_text_lines(file_path): ## tensorflow 함수로는 line이 몇개까지 있는지 카운트를 못함
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)

def train(params,dataloader):
    """Training loop."""
    num_gpus = 1
    learning_rate = 1e-4
    log_directory = './log_diretory'
    model_name = '/mono_depth2_mpeg_nointrin'#'/kitti_resnet_MS2_nbn_1epoch_pose_fix'#'/kitti_resnet_MS2_nbn_1epoch'#'/kitti_resnet_MS_cl=0,ml=0,gl=0.1,nbn_1epoch'#'/kitti_eigen_mask_vgg'

    checkpoint_path =''#'./log_diretory/11~32/model-323631'#'./log_diretory/lr_ssim_mask_bn/model-189107'#'./log_diretory/kitti_resnet_MS2_nbn_1epoch/model-8000'#'./trained/model-199060'#'./log_diretory/kitti_resnet_pose_cl=0,ml=1,gl=0.5,lr=0.1,nbn/model-48000'#'./log_diretory/kitti_resnet_pose_cl=0,ml=1,gl=0.1,lr=0.5,no_bn/model-19000'#'./log_diretory/kitti_vgg_mask_deconv/model-2000'#'./log_diretory/kitti_eigen/model-66360'
    path = log_directory + model_name

    dst ='./mono_depth2_mpeg_nointrin'#'./kitti_resnet_MS2_nbn_1epoch_pose_fix'#'./kitti_resnet_MS2_nbn_1epoch' #'./kitti_resnet_MS_cl=0,ml=0,gl=0.1,nbn_1epoch'#./kitti_eigen_mask_vgg'
    if not os.path.exists(dst):
        os.makedirs(dst)

    if not os.path.exists(path):
        os.makedirs(path)
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.Variable(0, trainable=False)

        # OPTIMIZER
        num_training_samples = count_text_lines(dataloader.filenames_file)  # 읽어올 파일 갯수

        steps_per_epoch = np.ceil(num_training_samples / params.batch_size).astype(np.int32)  # iteration return
        num_total_steps = params.num_epochs * steps_per_epoch  # total iteration
        start_learning_rate = learning_rate

        boundaries = [np.int32((3 / 5) * num_total_steps), np.int32((4 / 5) * num_total_steps)]
        values = [learning_rate, learning_rate / 2, learning_rate / 4]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

        opt_step = tf.train.AdamOptimizer(learning_rate)

        print("total number of samples: {}".format(num_training_samples))
        print("total number of steps: {}".format(num_total_steps))

        ## data loader
        dataloader = MonodepthDataloader(dataloader.data_path, dataloader.filenames_file, params, dataloader.dataset, dataloader.mode)
        reference  = dataloader.reference_image_batch
        left = dataloader.left_image_batch
        right = dataloader.right_image_batch
        param_path = dataloader.param_path_batch

        # split for each gpu
        reference_splits  = tf.split(reference,  num_gpus, 0)     # num_gpus에 맞춰 split 되는데 여기서는 num_gpus 가 1로 설정되어 있어서 [ 2,w,h,c] 에서 [1,2,w,h,c]
        left_splits = tf.split(left, num_gpus, 0)
        right_splits = tf.split(right, num_gpus, 0)
        param_path_splits = tf.split(param_path, num_gpus, 0)

        print(steps_per_epoch)
        tower_grads  = [] # 각 step별 graident
        tower_losses = [] # 각 step별 loss
        tower_image_loss = []
        tower_grad_loss = []
        # tower_pose_diff_loss = []
        # tower_origin_image_loss = []
        # tower_pose_image_loss = []
        #tower_coord_loss = []
        reuse_variables = None

        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%d' % i) as scope:
                        print(i)
                        model = MonodepthModel(params, dataloader.mode, reference_splits[i], left_splits[i],right_splits[i],
                                               param_path_splits[i],
                                               reuse_variables, i)

                        loss = model.total_loss
                        # pose_diff =model.pose_diff_loss
                        t_im_loss = model.image_loss
                        t_gr_loss = model.disp_gradient_loss
                        # origin_im_loss = model.origin_image_loss
                        # pose_im_loss = model.pose_image_loss

                        tower_losses.append(loss)
                        tower_image_loss.append(t_im_loss)
                        tower_grad_loss.append(t_gr_loss)
                        # tower_pose_diff_loss.append(pose_diff)
                        # tower_origin_image_loss.append(origin_im_loss)
                        # tower_pose_image_loss.append(pose_im_loss)

                        reuse_variables = True ## ??

                        bn_updates_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope)

                        grads = opt_step.compute_gradients(loss)

                        tower_grads.append(grads)

        grads = average_gradients(tower_grads)

        apply_gradient_op = opt_step.apply_gradients(grads, global_step=global_step) # gradient 적용
        train_op = tf.group(apply_gradient_op, tf.group(*bn_updates_op))

        total_loss = tf.reduce_mean(tower_losses)  # 걍 평균
        image_loss = tf.reduce_mean(tower_image_loss)
        grad_loss = tf.reduce_mean(tower_grad_loss)
        # pose_loss = tf.reduce_mean(tower_pose_diff_loss)
        # ori_im_loss = tf.reduce_mean(tower_origin_image_loss)
        # po_im_loss = tf.reduce_mean(tower_pose_image_loss)

        l2r_out = model.left_to_reference_est[0]
        r2r_out = model.right_to_reference_est[0]
        # l2r_p_out = model.pose_left_to_reference_est[0]
        # r2r_p_out = model.pose_right_to_reference_est[0]


        re_in = model.reference
        l_in = model.left
        r_in = model.right

        depth_left_out = model.depth_left_est[0]
        depth_right_out = model.depth_right_est[0]
        depth_reference_out = model.depth_reference_est[0]

        left_ori_pose = model.left_pose[0]
        left_ori_intrin = model.left_intrinsic[0]
        right_ori_pose = model.right_pose[0]
        right_ori_intrin = model.right_intrinsic[0]

        # left_est_pose = model.left_pose_train[0]
        # left_est_intrin = model.left_intrinsic_train[0]
        # right_est_pose = model.right_pose_train[0]
        # right_est_intrin = model.right_intrinsic_train[0]

        im_loss_left = model.reproject_loss_left[0]
        im_loss_right = model.reproject_loss_right[0]
        # im_loss_pose_left = model.reproject_loss_pose_left[0]
        # im_loss_pose_right = model.reproject_loss_pose_right[0]
        #
        # pose_cond = model.pose_cond

        epoch = 0

        # SESSION
        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)

        # SAVER
        train_saver = tf.train.Saver()

        # COUNT PARAMS
        total_num_parameters = 0
        for variable in tf.trainable_variables():
            total_num_parameters += np.array(variable.get_shape().as_list()).prod()
        print("number of trainable parameters: {}".format(total_num_parameters))

        # INIT
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

        # LOAD CHECKPOINT IF SET
        if checkpoint_path != '':
            print('----------------------------------------------')
            print(checkpoint_path)
            print('\n')
            print(checkpoint_path.split(".")[0])
            print('----------------------------------------------')
            train_saver.restore(sess, checkpoint_path)
            sess.run(global_step.assign(0))

        #num_total_steps-=323631
        # GO!
        start_step = global_step.eval(session=sess)
        start_time = time.time()
        for step in range(start_step,num_total_steps):#num_total_steps):
        #     #step = s+323631
        #
            before_op_time = time.time()
        #     if step==steps_per_epoch:
        #         model.epoch=True
        #         model.reuse_variables = True
        #
        #         tower_grads = []  # 각 step별 graident
        #         tower_losses = []  # 각 step별 loss
        #         tower_image_loss = []
        #         tower_grad_loss = []
        #         tower_pose_diff_loss = []
        #         tower_origin_image_loss = []
        #         tower_pose_image_loss = []
        #
        #         with tf.variable_scope(tf.get_variable_scope()):
        #             for i in range(num_gpus):
        #                 with tf.device('/gpu:%d' % i):
        #                     with tf.name_scope('%d' % i) as scope:
        #                         model.build_losses()
        #                         loss = model.total_loss
        #                         pose_diff = model.pose_diff_loss
        #                         t_im_loss = model.image_loss
        #                         t_gr_loss = model.disp_gradient_loss
        #                         origin_im_loss = model.origin_image_loss
        #                         pose_im_loss = model.pose_image_loss
        #
        #                         tower_losses.append(loss)
        #                         tower_image_loss.append(t_im_loss)
        #                         tower_grad_loss.append(t_gr_loss)
        #                         tower_pose_diff_loss.append(pose_diff)
        #                         tower_origin_image_loss.append(origin_im_loss)
        #                         tower_pose_image_loss.append(pose_im_loss)
        #
        #                         bn_updates_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)
        #
        #                         grads = opt_step.compute_gradients(loss)
        #
        #                         tower_grads.append(grads)
        #
        #         grads = average_gradients(tower_grads)
        #
        #         apply_gradient_op = opt_step.apply_gradients(grads, global_step=global_step)  # gradient 적용
        #         train_op = tf.group(apply_gradient_op, tf.group(*bn_updates_op))
        #
        #         total_loss = tf.reduce_mean(tower_losses)  # 걍 평균
        #         image_loss = tf.reduce_mean(tower_image_loss)
        #         grad_loss = tf.reduce_mean(tower_grad_loss)
        #         pose_loss = tf.reduce_mean(tower_pose_diff_loss)
        #         ori_im_loss =tf.reduce_mean(tower_origin_image_loss)
        #         po_im_loss = tf.reduce_mean(tower_pose_image_loss)
        #
        #
        #         l2r_out = model.left_to_reference_est[0]
        #         r2r_out = model.right_to_reference_est[0]
        #         l2r_p_out = model.pose_left_to_reference_est[0]
        #         r2r_p_out = model.pose_right_to_reference_est[0]
        #
        #         re_in = model.reference
        #         l_in = model.left
        #         r_in = model.right
        #
        #         depth_left_out = model.depth_left_est[0]
        #         depth_right_out = model.depth_right_est[0]
        #         depth_reference_out = model.depth_reference_est[0]
        #
        #         left_ori_pose = model.left_pose[0]
        #         left_ori_intrin = model.left_intrinsic[0]
        #         right_ori_pose = model.right_pose[0]
        #         right_ori_intrin = model.right_intrinsic[0]
        #
        #         left_est_pose = model.left_pose_train[0]
        #         left_est_intrin = model.left_intrinsic_train[0]
        #         right_est_pose = model.right_pose_train[0]
        #         right_est_intrin = model.right_intrinsic_train[0]
        #
        #         im_loss_left = model.reproject_loss_left[0]
        #         im_loss_right = model.reproject_loss_right[0]
        #         im_loss_pose_left = model.reproject_loss_pose_left[0]
        #         im_loss_pose_right = model.reproject_loss_pose_right[0]
        #
        #         pose_cond = model.pose_cond
        #
        #         train_saver = tf.train.Saver()

            _,loss_value,im_value,gr_value,\
            _,_,_, \
            re_im, l_im, r_im, \
            depth_re,depth_l,depth_r,\
            l2r_im,r2r_im, \
            l_ori_k,r_ori_k, \
            l_ori_pose, r_ori_pose,\
            l_loss,r_loss, =\
                sess.run([train_op, total_loss, image_loss, grad_loss, #coord_loss,
                          left_splits,right_splits,reference_splits,
                          re_in,l_in,r_in,
                          depth_reference_out,depth_left_out,depth_right_out,
                          l2r_out,r2r_out,
                          left_ori_intrin,right_ori_intrin,
                          left_ori_pose,right_ori_pose,
                          im_loss_left,im_loss_right])

            duration = time.time() - before_op_time                         # 각 step(iteration) 마다 걸리는 시간 계산
            if step == 0:
                print("Start!!")
            if step and step % 1 == 0:
                print('\n')
                print("mask_weight={}".format(model.mask_loss_weight))
                depth_re = depth_re.squeeze()
                examples_per_sec = params.batch_size / duration
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / step - 1.0) * time_sofar#(num_total_steps / s - 1.0) * time_sofar#(num_total_steps / step - 1.0) * time_sofar
                print_string = 'batch {:>6} | examples/s: {:4.2f} | loss: {:.5f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(print_string.format(step, examples_per_sec, loss_value, time_sofar, training_time_left))
                print("image loss: {:.5f} | grad_loss: {:.5f} ".format(im_value,gr_value))#mask_value))
                #print("coord={}/{}={}".format(coord_value,params.width,coord_value/params.width))

                print("depth min: {} | max: {}\nmin_index={} | max_index={}".format(np.min(depth_re),np.max(depth_re),np.argmin(depth_re),np.argmax(depth_re)))
                print("left_ori_k = ",l_ori_k)
                print("left_ori_pose = ", l_ori_pose)

                print("right_ori_k = ",r_ori_k)
                print("right_ori_pose = ", r_ori_pose)



                row = (np.argmax(depth_re[0]))//(params.width)
                col = ((np.argmax(depth_re[0]))%params.width)
                print("row,col={} {}".format(row,col))
                print("depth .shape={}".format(depth_re.shape))
                print("depth [{},{}] = {}".format(row,col,depth_re[0,row,col]))

            if step and step % 500==0:
                # for src
                plt.imshow(re_im[0].squeeze())
                plt.savefig(dst + '/{}_re_src.png'.format(step))
                plt.clf()
                plt.imshow(l_im[0].squeeze())
                plt.savefig(dst + '/{}_l_src.png'.format(step))
                plt.clf()
                plt.imshow(r_im[0].squeeze())
                plt.savefig(dst + '/{}_r_src.png'.format(step))
                plt.clf()

                print(depth_re.shape)
                # for depth
                plt.imshow(depth_re[0]/100.,cmap='gray')
                plt.savefig(dst + '/{}_re_depth.png'.format(step))
                plt.clf()
                plt.imshow(depth_l[0].squeeze()/100., cmap='gray')
                plt.savefig(dst + '/{}_l_depth.png'.format(step))
                plt.clf()
                plt.imshow(depth_r[0].squeeze() / 100., cmap='gray')
                plt.savefig(dst + '/{}_r_depth.png'.format(step))
                plt.clf()

                # for est
                plt.imshow(l2r_im[0].squeeze())
                plt.savefig(dst + '/{}_re_l2re.png'.format(step))
                plt.clf()
                plt.imshow(r2r_im[0].squeeze())
                plt.savefig(dst + '/{}_re_r2re.png'.format(step))
                plt.clf()

                print("l_loss.shape",l_loss[0][0].squeeze().shape)
                plt.imshow(l_loss[0][0].squeeze(), cmap='gray')
                plt.savefig(dst + '/{}_l_loss.png'.format(step))
                plt.clf()
                plt.imshow(r_loss[0][0].squeeze(), cmap='gray')
                plt.savefig(dst + '/{}_r_loss.png'.format(step))
                plt.clf()



            if step and step % steps_per_epoch ==0:
                train_saver.save(sess, log_directory + '/' + model_name + '/model/'+str(epoch), global_step=step)
                epoch+=1
                print("{}/{}".format(step,num_total_steps))

        train_saver.save(sess, log_directory + '/' + model_name + '/model/'+str(epoch), global_step=num_total_steps)

def test(params,dataloader):
    # TODO:
    print('test')
    num=0
    txts = glob.glob('./utils/filenames/mpeg_list/*.txt')
    for filenames in txts:

        dataloader.filenames_file = filenames.replace('\\', '/')
        dataloader.__init__(dataloader.data_path, dataloader.filenames_file, dataloader.params, dataloader.dataset,
                            dataloader.mode)

        # MonodepthDataloader from monodepth_dataloader
        # MonodepthDataloader(args.data_path,
        #                    args.filenames_file,
        #                    params,
        #                    args.dataset,
        #                    args.mode)
        output_directory=''
        if dataloader.mode == 'test':
            #checkpoint_path='./pretrain/model_kitti/model_kitti'
            #checkpoint_path = './pretrain/model_new_sampler/model-35100'
            checkpoint_path = 'log_diretory/mono_depth2_mpeg_nointrin/model/20-288750'
            name = dataloader.filenames_file.split('/')[-1].replace('.txt','')
            output_directory = './result'+'/'+name
        else:
            checkpoint_path = ''
            log_directory = './log_diretory'
            model_name = '/model_ktii'
            path = log_directory+model_name
            if not os.path.exists(path):
                os.makedirs(path)

        #left = dataloader.left_image_batch # shape = [2,height,width,channel]
        refer = dataloader.reference_image_batch
        #right = dataloader.right_image_batch # params.do_stereo = True일 때만 값 들어옴
        #param_path = dataloader.param_path_batch

        # model load
        if num==0:
            model = MonodepthModel(params, dataloader.mode,refer)
            num+=1
        else:

            model = MonodepthModel(params, dataloader.mode, refer, reuse_variables=True)

        # Session
        config = tf.ConfigProto(allow_soft_placement = True) # allow_soft_placement는 명시된 device없을 때 자동으로 잡아준다.
        sess = tf.Session(config=config)

        # Saver
        train_saver = tf.train.Saver()

        # Init
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coordinator = tf.train.Coordinator() ## coordinator=조정자, threads 관리해주는 함수
        threads = tf.train.start_queue_runners(sess=sess,coord=coordinator)

        # Restore -> 여기는 그냥 모델 불러올지 말지 정하는 곳
        if checkpoint_path == '':
            restore_path = tf.train.latest_checkpoint(path)
        else:
            restore_path = checkpoint_path
            print('-------------------------------restore_path={}'.format(restore_path))
        train_saver.restore(sess, restore_path)

        # disparity matrix init
        num_test_samples = count_text_lines(dataloader.filenames_file) # .txt 내에 line 몇개까지 있는지 return

        #disparities = np.zeros((num_test_samples,params.height,params.width),dtype=np.float32)
        disparities_m = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
        depths_m = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
        #disparities_r = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
        #est_l = np.zeros((num_test_samples, params.height, params.width,3), dtype=np.float32)
        #est_r = np.zeros((num_test_samples, params.height, params.width,3), dtype=np.float32)


        #disparities_pp = np.zeros((num_test_samples,params.height,params.width),dtype=np.float32)

        # disparities_flip = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
        # testing
        print('Start')
        #for step in range(num_test_samples):
        for step in range(num_test_samples):
            #disp_l,disp_r,l_test, r_test = sess.run([model.disp_left_est[0],model.disp_right_est[0],model.left_est[0], model.right_est[0]])
            depth_m,disp_m = sess.run(
                [model.depth_reference_est[0],
            model.disp_reference_est[0]])

            #print(sess.run(tf.shape(l_test[0])))

            disparities_m[step] = disp_m[0].squeeze()

            depths_m[step] = depth_m[0].squeeze()

            print("{}/{}".format(step,num_test_samples))

        print('done.')

        ## save image
        print('writing disparities.')
        if output_directory =='':
            output_directory = os.path.dirname(checkpoint_path)
        else:
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
        coordinator.request_stop()
        coordinator.join(threads)
        np.save(output_directory + '/mpeg_disp.npy', disparities_m)
        np.save(output_directory + '/mpeg_depth.npy', depths_m)
        sess.close()


def main(_):
    params = monodepth_parameters(
        encoder='resnet50',
        height=192,
        width=640,
        num_threads=8,
        num_epochs=21,
        batch_size=6,
        do_stereo=True,  # if set, will train the stereo model
        wrap_mode='border',  # or edge
        use_deconv=True,  # if set, will use transeposed convolutions
        alpha_image_loss=0.85,
        disp_gradient_loss_weight=0.1,
        lr_loss_weight=0.1,
        full_summary=False,  # 이거는 tensorboard 용 인듯
        parameter_path = '',
        do_change=False,
        input_type = 'MS',
        mask_loss_weight = 0.1
    )
    # # kitti 용
    # dataloader = MonodepthDataloader('E:/KITTI_backup/raw/',
    #                                  './utils/filenames/eigen_zhou/Mono2_train_files.txt',
    #                                  params,
    #                                  'kitti',
    #                                  'train')
    dataloader = MonodepthDataloader('D:/mpeg/',#'C:/Users/server/Desktop/recovery/Vol1/Users/Y/Desktop/Kocca/3.실험데이터/8.DataSet/mpeg/',
                                     './utils/filenames/mpeg_list/ballon1.txt',#'./utils/filenames/Mono_mpeg_random.txt',
                                     params,
                                     'kitti',
                                     'test')#'train')

    if dataloader.mode == 'test':
        test(params,dataloader)
    else:
        print('train')
        train(params, dataloader)

if __name__=='__main__':
    tf.app.run()