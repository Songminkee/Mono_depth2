from __future__ import absolute_import, division, print_function
import tensorflow as tf
from util import *

def string_length_tf(t): # t=image_path, 진짜 경로 string 길이 재는 함수
    return tf.cast(tf.py_func(len ,[t], [tf.int32]),tf.int64)

class MonodepthDataloader(object):
    """monodepth dataloader"""

    def __init__(self, data_path, filenames_file, params, dataset, mode):
        # data_path = 'E:/KITTI_backup/raw/',
        # filenames_file = './utils/filenames/kitti_test_files.txt',
        # prams = params,
        # dataset = 'kitti',
        # mode = 'test'
        self.data_path = data_path # data root_path임, 최종 path는 각각 root_path/splint_line 이 될거임
        self.params = params
        self.dataset = dataset
        self.mode = mode
        self.filenames_file = filenames_file # 이거는 내가 추가한거

        self.param_path_batch =[]
        self.reference_image_batch = None
        self.left_image_batch = None
        self.right_image_batch = None

        input_queue = tf.train.string_input_producer([filenames_file], shuffle=False)
        line_reader = tf.TextLineReader()
        number, line = line_reader.read(input_queue) # _에는 'filenames_file:line number' 식으로 들어감, line에는 열 파일

        split_line = tf.string_split([line]).values # 아마 띄어쓰기만 split

        if mode == 'train':
            # randomly side change
            if params.input_type == 'MS':
                left_image_path,reference_image_path,right_image_path,left_param_path,reference_param_path,right_param_path = self.tf_get_path_train(split_line)

                reference_image_o = self.read_image(reference_image_path)
                left_image_o = self.read_image(left_image_path)
                right_image_o = self.read_image(right_image_path)

                param_path_batch = tf.stack([left_param_path, reference_param_path,right_param_path])

                # randomly augment images
                do_augment = tf.random_uniform([], 0, 1)
                reference_image, left_image ,right_image= tf.cond(do_augment > 0.5,
                                                      lambda: self.augment_image_pair_triple(reference_image_o, left_image_o,right_image_o),
                                                      lambda: (reference_image_o, left_image_o,right_image_o))

                reference_image = (reference_image - 0.45) / 0.225
                left_image = (left_image - 0.45) / 0.225
                right_image = (right_image - 0.45) / 0.225

                reference_image.set_shape([None, params.height, params.width, 3])
                left_image.set_shape([None, params.height, params.width, 3])
                right_image.set_shape([None, params.height, params.width, 3])

                param_path_batch = tf.reshape(param_path_batch, [1, 3])
                param_path_batch.set_shape([None, 3])

                min_after_dequeue = 4096
                capacity = min_after_dequeue + 4 * params.batch_size

                self.reference_image_batch, self.left_image_batch,self.right_image_batch, self.param_path_batch = \
                    tf.train.shuffle_batch(
                        [reference_image, left_image,right_image, param_path_batch], params.batch_size, capacity,
                        min_after_dequeue,
                        params.num_threads, enqueue_many=True)
        elif mode == 'test':
            reference_image_path = self.tf_get_path(split_line)
            self.reference_image_batch = (self.read_image_not_train(reference_image_path)- 0.45) / 0.225
            self.reference_image_batch.set_shape([None, params.height, params.width, 3])

        elif mode == 'eval':
            if params.input_type == 'MS':

                reference_image_path = self.tf_get_path(split_line)
                print(reference_image_path)


                self.pppppath = reference_image_path
                self.reference_image_batch = (self.read_image_not_train(reference_image_path)- 0.45) / 0.225

                self.reference_image_batch.set_shape([None, params.height, params.width, 3])

        elif mode == 'kitti_eval':
            if params.input_type == 'MS':

                reference_image_path,reference_param_path,reference_rect_path = self.tf_get_path_kitti(split_line)
                self.reference_image_batch = self.read_image_not_train(reference_image_path)
                self.param_path_batch = tf.stack([reference_param_path, reference_rect_path])
                self.param_path_batch = tf.reshape(self.param_path_batch, [1, 2])
                self.param_path_batch.set_shape([None,2])
                self.reference_image_batch.set_shape([None, params.height, params.width, 3])


    def augment_image_pair(self, left_image, right_image): ## train시만 들어옴, agumentationㅇ
        # randomly shift gamma
        random_gamma = tf.random_uniform([], 0.8, 1.2)
        left_image_aug = left_image ** random_gamma
        right_image_aug = right_image ** random_gamma

        # randomly shift brightness
        random_brightness = tf.random_uniform([], 0.5, 2.0)
        left_image_aug = left_image_aug * random_brightness
        right_image_aug = right_image_aug * random_brightness

        # randomly shift color
        random_colors = tf.random_uniform([3], 0.8, 1.2)
        # white = tf.ones([tf.shape(left_image)[0], tf.shape(left_image)[1]])
        # color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=2)
        white = tf.ones([tf.shape(left_image)[0], tf.shape(left_image)[1], tf.shape(left_image)[2]])
        color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=3)
        left_image_aug *= color_image
        right_image_aug *= color_image

        # saturate
        left_image_aug = tf.clip_by_value(left_image_aug, 0, 1)
        right_image_aug = tf.clip_by_value(right_image_aug, 0, 1)

        return left_image_aug, right_image_aug

    def augment_image_pair_triple(self,reference_image, left_image, right_image):  ## train시만 들어옴, agumentationㅇ
        # randomly shift gamma
        random_gamma = tf.random_uniform([], 0.8, 1.2)
        left_image_aug = left_image ** random_gamma
        right_image_aug = right_image ** random_gamma
        reference_image_aug = reference_image ** random_gamma

        # randomly shift brightness
        random_brightness = tf.random_uniform([], 0.5, 2.0)
        left_image_aug = left_image_aug * random_brightness
        right_image_aug = right_image_aug * random_brightness
        reference_image_aug = reference_image_aug * random_brightness

        # randomly shift color
        random_colors = tf.random_uniform([3], 0.8, 1.2)
        # white = tf.ones([tf.shape(left_image)[0], tf.shape(left_image)[1]])
        # color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=2)
        white = tf.ones([tf.shape(left_image)[0], tf.shape(left_image)[1], tf.shape(left_image)[2]])
        color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=3)
        left_image_aug *= color_image
        right_image_aug *= color_image
        reference_image_aug *= color_image

        # saturate
        left_image_aug = tf.clip_by_value(left_image_aug, 0, 1)
        right_image_aug = tf.clip_by_value(right_image_aug, 0, 1)
        reference_image_aug = tf.clip_by_value(reference_image_aug,0,1)

        return reference_image_aug,left_image_aug, right_image_aug

    def augment_image_pair_MS(self, left_image, right_image,pre_image,post_image): ## train시만 들어옴, agumentationㅇ
        # randomly shift gamma
        random_gamma = tf.random_uniform([], 0.8, 1.2)
        left_image_aug = left_image ** random_gamma
        right_image_aug = right_image ** random_gamma
        pre_image_aug = pre_image ** random_gamma
        post_image_aug = post_image ** random_gamma

        # randomly shift brightness
        random_brightness = tf.random_uniform([], 0.5, 2.0)
        left_image_aug = left_image_aug * random_brightness
        right_image_aug = right_image_aug * random_brightness
        pre_image_aug = pre_image_aug * random_brightness
        post_image_aug = post_image_aug * random_brightness

        # randomly shift color
        random_colors = tf.random_uniform([3], 0.8, 1.2)
        # white = tf.ones([tf.shape(left_image)[0], tf.shape(left_image)[1]])
        # color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=2)
        white = tf.ones([tf.shape(left_image)[0], tf.shape(left_image)[1], tf.shape(left_image)[2]])
        color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=3)
        left_image_aug *= color_image
        right_image_aug *= color_image
        pre_image_aug *= color_image
        post_image_aug *= color_image

        # saturate
        left_image_aug = tf.clip_by_value(left_image_aug, 0, 1)
        right_image_aug = tf.clip_by_value(right_image_aug, 0, 1)
        pre_image_aug = tf.clip_by_value(pre_image_aug, 0, 1)
        post_image_aug = tf.clip_by_value(post_image_aug, 0, 1)

        return left_image_aug, right_image_aug,pre_image_aug,post_image_aug

    def read_image(self, image_path):
        # tf.decode_image does not return the image size, this is an ugly workaround to handle both jpeg and png
        #print('image_path={}'.format(image_path))
        path_length = string_length_tf(image_path)[0]
        file_extension = tf.substr(image_path, path_length - 3, 3)
        file_cond = tf.equal(file_extension, 'jpg') ## image path 보고 .jpg 인지 .png 인지 확인

        image = tf.cond(file_cond, lambda: tf.image.decode_jpeg(tf.read_file(image_path)), ## file_cond가 참이면, jpg decode 함수 씀
                        lambda: tf.image.decode_png(tf.read_file(image_path))) # 거짓이면, png_decode 함수 씀

        # if the dataset is cityscapes, we crop the last fifth to remove the car hood
        if self.dataset == 'cityscapes':
            o_height = tf.shape(image)[0]
            crop_height = (o_height * 4) // 5
            image = image[:crop_height, :, :]

        image = tf.image.convert_image_dtype(image, tf.float32) # 이미지는 tf.float32인 상태로
        image = tf.image.resize_images(image, [self.params.height, self.params.width], tf.image.ResizeMethod.AREA)   ## 불러온 이미지를 입력했던대로 resize, method에 대해서는 더 찾아봐야 할듯
        image = tf.reshape(image, [1, self.params.height, self.params.width, 3])
        return image

    def read_image_not_train(self, image_path):
        # tf.decode_image does not return the image size, this is an ugly workaround to handle both jpeg and png
        #print('image_path={}'.format(image_path))
        path_length = string_length_tf(image_path)[0]
        file_extension = tf.substr(image_path, path_length - 3, 3)
        file_cond = tf.equal(file_extension, 'jpg') ## image path 보고 .jpg 인지 .png 인지 확인

        image = tf.cond(file_cond, lambda: tf.image.decode_jpeg(tf.read_file(image_path)), ## file_cond가 참이면, jpg decode 함수 씀
                        lambda: tf.image.decode_png(tf.read_file(image_path))) # 거짓이면, png_decode 함수 씀

        # if the dataset is cityscapes, we crop the last fifth to remove the car hood
        if self.dataset == 'cityscapes':
            o_height = tf.shape(image)[0]
            crop_height = (o_height * 4) // 5
            image = image[:crop_height, :, :]

        image = tf.image.convert_image_dtype(image, tf.float32) # 이미지는 tf.float32인 상태로
        image = tf.image.resize_images(image, [self.params.height, self.params.width], tf.image.ResizeMethod.AREA)   ## 불러온 이미지를 입력했던대로 resize, method에 대해서는 더 찾아봐야 할듯
        image = tf.reshape(image, [1, self.params.height, self.params.width, 3])
        return image

    def tf_get_path(self, split_line):

        reference_image_path = \
            tf.py_func(get_path, [split_line, self.data_path],
                       tf.string)

        return reference_image_path

    def tf_get_path_kitti(self, split_line):

        reference_image_path,reference_param_path,reference_rect_path = \
            tf.py_func(get_path_kitti, [split_line, self.data_path],
                       [tf.string,tf.string,tf.string])

        return reference_image_path,reference_param_path,reference_rect_path

    def tf_get_path_train(self,split_line):

        left_image_path, reference_image_path, right_image_path,left_param_path, reference_param_path, right_param_path= \
            tf.py_func(get_path_idol, [split_line,self.data_path], [tf.string, tf.string, tf.string, tf.string,tf.string,tf.string])

        return left_image_path, reference_image_path, right_image_path,left_param_path, reference_param_path, right_param_path
