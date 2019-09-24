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

import cv2

STEREO_SCALE_FACTOR = 5.4
# TODO: R_rect00 반영
def count_text_lines(file_path): ## tensorflow 함수로는 line이 몇개까지 있는지 카운트를 못함
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)

def disp_to_depth(disp):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    MIN_DEPTH = 0.1#0.1#1e-3#0.1
    MAX_DEPTH = 100#100
    min_disp = 1 / MAX_DEPTH
    max_disp = 1 / MIN_DEPTH
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def evaluate(params,dataloader):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    num_gpus = 1
    pred_depth_scale_factor = 1
    checkpoint_path = './log_diretory/mono_depth2-102000/model-97060'#'./log_diretory/kitti_resnet_MS2_nbn_1epoch_pose_fix/model-189107'

    gt_path = './utils/gt/eigen_zhou'
    eval_stereo = False

    with tf.Graph().as_default(), tf.device('/cpu:0'):

        dataloader = MonodepthDataloader(dataloader.data_path, dataloader.filenames_file, params, dataloader.dataset,
                                         dataloader.mode)
        reference = dataloader.reference_image_batch
        param = dataloader.param_path_batch


        # split for each gpu
        reference_splits = tf.split(reference, num_gpus,0)
        param_splits = tf.split(param,num_gpus,0)



        reuse_variables = None

        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%d' % i) as scope:
                        print(i)
                        model = MonodepthModel(params, dataloader.mode, reference_splits[i],None,None,None,param_splits[i],
                                               #param_path=param_path_splits[i],
                                               reuse_variables=reuse_variables, model_index=i)



        config = tf.ConfigProto(allow_soft_placement=True)  # allow_soft_placement는 명시된 device없을 때 자동으로 잡아준다.
        sess = tf.Session(config=config)
        # Saver
        train_saver = tf.train.Saver()

        # Init
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coordinator = tf.train.Coordinator()  ## coordinator=조정자, threads 관리해주는 함수
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

        # Restore
        print("Restore")

        if checkpoint_path != '':
            print('----------------------------------------------')
            print(checkpoint_path)
            print('\n')
            print(checkpoint_path.split(".")[0])
            print('----------------------------------------------')
            train_saver.restore(sess, checkpoint_path)
        print("Restore OK")
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%d' % i) as scope:
                        bn_updates_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)
        num_test_samples = count_text_lines(dataloader.filenames_file)
        pred_disps = []
        print('Start')
        for step in range(num_test_samples):
            pred_disp = sess.run(model.disp_reference_est[0])

            pred_disp = pred_disp.squeeze()
            pred_disp,_ = disp_to_depth(pred_disp)

            # print(pred_disp.shape)
            # plt.imshow(pred_disp)
            # plt.show()
            pred_disp = np.expand_dims(pred_disp,0)

            pred_disps.append(pred_disp)

        pred_disps = np.concatenate(pred_disps)
        print(pred_disps.shape)
        gt_path = gt_path+ '/gt_depths.npz'
        gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]
        print(gt_depths[0].shape)

        print("-> Evaluating")
        disable_median_scaling=False
        if eval_stereo:
            print("   Stereo evaluation - "
                  "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
            disable_median_scaling = True
            pred_depth_scale_factor = STEREO_SCALE_FACTOR
        else:
            print("   Mono evaluation - using median scaling")

        errors = []
        ratios = []

        for i in range(pred_disps.shape[0]):

            gt_depth = gt_depths[i]
            gt_height, gt_width = gt_depth.shape[:2]

            pred_disp = pred_disps[i]
            pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
            pred_depth = 1 / pred_disp
            print(pred_depth[0,0])




            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)

            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

            print(mask)
            #if i ==pred_disps.shape[0]-3:
            # plt.imshow(pred_depth / 100)  # pred_depth[mask]/100)
            # plt.show()
            # plt.imshow(np.where(mask,pred_depth,np.zeros_like(pred_depth))/100)#pred_depth[mask]/100)
            # plt.show()
            # plt.imshow(np.where(mask,gt_depth,np.zeros_like(gt_depth))/100)
            # plt.show()

            print("pred_depth[mask]", pred_depth[mask])
            print("gt_depth[mask]", gt_depth[mask])
            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

            pred_depth *= pred_depth_scale_factor
            if not disable_median_scaling:
                print('?')
                ratio = np.median(gt_depth) / np.median(pred_depth)
                ratios.append(ratio)
                pred_depth *= ratio

            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
            print("pred_depth={}".format(pred_depth))
            print("pred_depth < MIN_DEPTH",pred_depth < MIN_DEPTH)
            print(" pred_depth[pred_depth < MIN_DEPTH] ", pred_depth[pred_depth < MIN_DEPTH] )
            print("pred_depth > MAX_DEPTH",pred_depth > MAX_DEPTH)
            print("pred_depth[pred_depth > MAX_DEPTH]",pred_depth[pred_depth > MAX_DEPTH])
            print("pred_depth_shape={}".format(pred_depth.shape))
            print("gt_depth_shape={}".format(gt_depth.shape))

            errors.append(compute_errors(gt_depth, pred_depth))

        if not disable_median_scaling:
            ratios = np.array(ratios)
            med = np.median(ratios)
            print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

        mean_errors = np.array(errors).mean(0)

        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
        print("\n-> Done!")

def main(_):
    params = monodepth_parameters(
        encoder='resnet50',
        height=192,
        width=640,
        num_threads=8,
        num_epochs=20,
        batch_size=4,
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
    # kitti 용
    dataloader = MonodepthDataloader('E:/KITTI_backup/raw/',
                                     './utils/filenames/eigen_zhou/Mono2_test_files.txt',#'./utils/filenames/eigen_zhou/test.txt',#'./utils/filenames/eigen_zhou/Mono2_test_files.txt',
                                     params,
                                     'kitti',
                                     'kitti_eval')
    # dataloader = MonodepthDataloader('E:/2nd_idol/Monodepth1_4/',
    #                                  './utils/filenames/Mono_1.4_train_random_stereo.txt',
    #                                  params,
    #                                  'kitti',
    #                                  'train')


    if dataloader.mode == 'eval':
        evaluate(params,dataloader)
    elif dataloader.mode == 'kitti_eval':
        evaluate(params, dataloader)
    elif dataloader.mode == 'train':
        evaluate(params,dataloader)


if __name__=='__main__':
    tf.app.run()