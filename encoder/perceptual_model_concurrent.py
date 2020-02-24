import os
import bz2
import PIL.Image
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.utils import get_file
from keras.applications.vgg16 import VGG16, preprocess_input
import keras.backend as K
import traceback


# TODO : une fois que ca tournera, faudra tej tout ce qui est en rapport avec les MASKS

def load_images(images_list, image_size=256):
    loaded_images = list()
    for img_path in images_list:
        img = PIL.Image.open(img_path).convert('RGB').resize((image_size,image_size),PIL.Image.LANCZOS)
        img = np.array(img)
        img = np.expand_dims(img, 0)
        loaded_images.append(img)
    loaded_images = np.vstack(loaded_images)
    return loaded_images

def tf_custom_l1_loss(img1,img2):
    return tf.math.reduce_mean(tf.math.abs(img2-img1), axis=None)

def tf_custom_logcosh_loss(img1,img2):
    return tf.math.reduce_mean(tf.keras.losses.logcosh(img1,img2))

def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path

class PerceptualModelConcurrent:
    def __init__(self, args, batch_size=1, perc_model=None, sess=None):
        self.sess = tf.get_default_session() if sess is None else sess
        K.set_session(self.sess)
        self.epsilon = 1e-7
        self.lr = args.lr
        self.decay_rate = args.decay_rate
        self.decay_steps = args.decay_steps
        self.img_size = args.image_size
        self.layer = args.use_vgg_layer
        self.vgg_loss = args.use_vgg_loss
        self.face_mask = args.face_mask
        self.use_grabcut = args.use_grabcut
        self.scale_mask = args.scale_mask
        self.mask_dir = args.mask_dir
        if (self.layer <= 0 or self.vgg_loss <= self.epsilon):
            self.vgg_loss = None
        self.pixel_loss = args.use_pixel_loss
        if (self.pixel_loss <= self.epsilon):
            self.pixel_loss = None
        self.mssim_loss = args.use_mssim_loss
        if (self.mssim_loss <= self.epsilon):
            self.mssim_loss = None
        self.lpips_loss = args.use_lpips_loss
        if (self.lpips_loss <= self.epsilon):
            self.lpips_loss = None
        self.l1_penalty = args.use_l1_penalty
        if (self.l1_penalty <= self.epsilon):
            self.l1_penalty = None
        self.batch_size = batch_size
        if perc_model is not None and self.lpips_loss is not None:
            self.perc_model = perc_model
        else:
            self.perc_model = None
        self.perceptual_model = None
        self.ref_img_1 = None
        self.ref_weight_1 = None
        self.ref_img_features_1 = None
        self.ref_features_weight_1 = None
        self.ref_img_2 = None
        self.ref_weight_2 = None
        self.ref_img_features_2 = None
        self.ref_features_weight_2 = None
        self.loss = None

        if self.face_mask:
            import dlib
            self.detector = dlib.get_frontal_face_detector()
            LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
            landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                                    LANDMARKS_MODEL_URL, cache_subdir='temp'))
            self.predictor = dlib.shape_predictor(landmarks_model_path)

    def compare_images(self,img1,img2):
        if self.perc_model is not None:
            return self.perc_model.get_output_for(tf.transpose(img1, perm=[0,3,2,1]), tf.transpose(img2, perm=[0,3,2,1]))
        return 0

    def add_placeholder(self, var_name):
        var_val = getattr(self, var_name)
        setattr(self, var_name + "_placeholder", tf.placeholder(var_val.dtype, shape=var_val.get_shape()))
        setattr(self, var_name + "_op", var_val.assign(getattr(self, var_name + "_placeholder")))

    def assign_placeholder(self, var_name, var_val):
        self.sess.run(getattr(self, var_name + "_op"), {getattr(self, var_name + "_placeholder"): var_val})

    def build_perceptual_model(self, generator):
        # Learning rate
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
        incremented_global_step = tf.assign_add(global_step, 1)
        self._reset_global_step = tf.assign(global_step, 0)
        self.learning_rate = tf.train.exponential_decay(self.lr, incremented_global_step,
                self.decay_steps, self.decay_rate, staircase=True)
        self.sess.run([self._reset_global_step])

        generated_image_tensor = generator.generated_image
        generated_image = tf.image.resize_nearest_neighbor(generated_image_tensor,
                                                           (self.img_size, self.img_size),
                                                           align_corners=True)

        # reference images 1
        self.ref_img_1 = tf.get_variable('ref_img_1',
                                         shape=generated_image.shape,
                                         dtype='float32',
                                         initializer=tf.initializers.zeros())
        self.ref_weight_1 = tf.get_variable('ref_weight_1',
                                            shape=generated_image.shape,
                                            dtype='float32',
                                            initializer=tf.initializers.zeros())
        self.add_placeholder("ref_img_1")
        self.add_placeholder("ref_weight_1")

        # reference images 2
        self.ref_img_2 = tf.get_variable('ref_img_2',
                                         shape=generated_image.shape,
                                         dtype='float32',
                                         initializer=tf.initializers.zeros())
        self.ref_weight_2 = tf.get_variable('ref_weight_2',
                                            shape=generated_image.shape,
                                            dtype='float32',
                                            initializer=tf.initializers.zeros())
        self.add_placeholder("ref_img_2")
        self.add_placeholder("ref_weight_2")


        if (self.vgg_loss is not None):
            vgg16 = VGG16(include_top=False, input_shape=(self.img_size, self.img_size, 3))
            self.perceptual_model = Model(vgg16.input, vgg16.layers[self.layer].output)
            generated_img_features = self.perceptual_model(preprocess_input(self.ref_weight_1 * generated_image))
            # TODO: understand why self.ref_weight_1 above > voir si on peut pas le tej..

            # reference images 1
            self.ref_img_features_1 = tf.get_variable('ref_img_features_1',
                                                      shape=generated_img_features.shape,
                                                      dtype='float32',
                                                      initializer=tf.initializers.zeros())
            self.ref_features_weight_1 = tf.get_variable('ref_features_weight_1',
                                                         shape=generated_img_features.shape,
                                                         dtype='float32',
                                                         initializer=tf.initializers.zeros())
            self.sess.run([self.ref_features_weight_1.initializer, self.ref_features_weight_1.initializer])
            self.add_placeholder("ref_img_features_1")
            self.add_placeholder("ref_features_weight_1")

            # reference images 2
            self.ref_img_features_2 = tf.get_variable('ref_img_features_2',
                                                      shape=generated_img_features.shape,
                                                      dtype='float32',
                                                      initializer=tf.initializers.zeros())
            self.ref_features_weight_2 = tf.get_variable('ref_features_weight_2',
                                                         shape=generated_img_features.shape,
                                                         dtype='float32',
                                                         initializer=tf.initializers.zeros())
            self.sess.run([self.ref_features_weight_2.initializer, self.ref_features_weight_2.initializer])
            self.add_placeholder("ref_img_features_2")
            self.add_placeholder("ref_features_weight_2")

        self.loss = 0
        # L1 loss on VGG16 features
        if (self.vgg_loss is not None):
            self.loss += 0.5 * self.vgg_loss * tf_custom_l1_loss(self.ref_features_weight_1 * self.ref_img_features_1, self.ref_features_weight_1 * generated_img_features)
            self.loss += 0.5 * self.vgg_loss * tf_custom_l1_loss(self.ref_features_weight_2 * self.ref_img_features_2, self.ref_features_weight_2 * generated_img_features)
        # + logcosh loss on image pixels
        if (self.pixel_loss is not None):
            self.loss += 0.5 * self.pixel_loss * tf_custom_logcosh_loss(self.ref_weight_1 * self.ref_img_1, self.ref_weight_1 * generated_image)
            self.loss += 0.5 * self.pixel_loss * tf_custom_logcosh_loss(self.ref_weight_2 * self.ref_img_2, self.ref_weight_2 * generated_image)
        # + MS-SIM loss on image pixels
        if (self.mssim_loss is not None):
            self.loss += 0.5 * self.mssim_loss * tf.math.reduce_mean(1 - tf.image.ssim_multiscale(self.ref_weight_1 * self.ref_img_1, self.ref_weight_1 * generated_image, 1))
            self.loss += 0.5 * self.mssim_loss * tf.math.reduce_mean(1 - tf.image.ssim_multiscale(self.ref_weight_2 * self.ref_img_2, self.ref_weight_2 * generated_image, 1))
        # + extra perceptual loss on image pixels
        if self.perc_model is not None and self.lpips_loss is not None:
            self.loss += 0.5 * self.lpips_loss * tf.math.reduce_mean(self.compare_images(self.ref_weight_1 * self.ref_img_1, self.ref_weight_1 * generated_image))
            self.loss += 0.5 * self.lpips_loss * tf.math.reduce_mean(self.compare_images(self.ref_weight_2 * self.ref_img_2, self.ref_weight_2 * generated_image))
        # + L1 penalty on dlatent weights
        if self.l1_penalty is not None:
            self.loss += 0.5 * self.l1_penalty * 512 * tf.math.reduce_mean(tf.math.abs(generator.dlatent_variable - generator.get_dlatent_avg()))

    def generate_face_mask(self, im):
        from imutils import face_utils
        import cv2
        rects = self.detector(im, 1)
        # loop over the face detections
        for (j, rect) in enumerate(rects):
            """
            Determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
            """
            shape = self.predictor(im, rect)
            shape = face_utils.shape_to_np(shape)

            # we extract the face
            vertices = cv2.convexHull(shape)
            mask = np.zeros(im.shape[:2],np.uint8)
            cv2.fillConvexPoly(mask, vertices, 1)
            if self.use_grabcut:
                bgdModel = np.zeros((1,65),np.float64)
                fgdModel = np.zeros((1,65),np.float64)
                rect = (0,0,im.shape[1],im.shape[2])
                (x,y),radius = cv2.minEnclosingCircle(vertices)
                center = (int(x),int(y))
                radius = int(radius*self.scale_mask)
                mask = cv2.circle(mask,center,radius,cv2.GC_PR_FGD,-1)
                cv2.fillConvexPoly(mask, vertices, cv2.GC_FGD)
                cv2.grabCut(im,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
                mask = np.where((mask==2)|(mask==0),0,1)
            return mask

    def set_reference_images(self, images_list_1, images_list_2):

        # TODO: simplify that by refactoring ?

        assert(len(images_list_1) != 0 and len(images_list_1) <= self.batch_size)
        assert(len(images_list_1) == len(images_list_2))

        # set reference images 1

        loaded_image_1 = load_images(images_list_1, self.img_size)
        image_features_1 = None
        # Compute reference images features
        if self.perceptual_model is not None:
            image_features_1 = self.perceptual_model.predict_on_batch(preprocess_input(loaded_image_1))
            weight_mask_1 = np.ones(self.ref_features_weight_1.shape)

        if self.face_mask:
            image_mask_1 = np.zeros(self.ref_weight_1.shape)
            for (i, im) in enumerate(loaded_image_1):
                try:
                    _, img_name = os.path.split(images_list_1[i])
                    mask_img_1 = os.path.join(self.mask_dir, f'{img_name}')
                    if (os.path.isfile(mask_img_1)):
                        print("Loading mask " + mask_img_1)
                        imask = PIL.Image.open(mask_img_1).convert('L')
                        mask = np.array(imask)/255
                        mask = np.expand_dims(mask,axis=-1)
                    else:
                        mask = self.generate_face_mask(im)
                        imask = (255*mask).astype('uint8')
                        imask = PIL.Image.fromarray(imask, 'L')
                        print("Saving mask " + mask_img_1)
                        imask.save(mask_img_1, 'PNG')
                        mask = np.expand_dims(mask,axis=-1)
                    mask = np.ones(im.shape,np.float32) * mask
                except Exception as e:
                    print("Exception in mask handling for " + mask_img_1)
                    traceback.print_exc()
                    mask = np.ones(im.shape[:2],np.uint8)
                    mask = np.ones(im.shape,np.float32) * np.expand_dims(mask,axis=-1)
                image_mask_1[i] = mask
            img = None
        else:
            image_mask_1 = np.ones(self.ref_weight_1.shape)

        if len(images_list_1) != self.batch_size:
            if image_features_1 is not None:
                features_space = list(self.ref_features_weight_1.shape[1:])
                existing_features_shape = [len(images_list_1)] + features_space
                empty_features_shape = [self.batch_size - len(images_list_1)] + features_space
                existing_examples = np.ones(shape=existing_features_shape)
                empty_examples = np.zeros(shape=empty_features_shape)
                weight_mask_1 = np.vstack([existing_examples, empty_examples])
                image_features_1 = np.vstack([image_features_1, np.zeros(empty_features_shape)])

            images_space = list(self.ref_weight_1.shape[1:])
            existing_images_space = [len(images_list_1)] + images_space
            empty_images_space = [self.batch_size - len(images_list_1)] + images_space
            existing_images = np.ones(shape=existing_images_space)
            empty_images = np.zeros(shape=empty_images_space)
            image_mask_1 = image_mask_1 * np.vstack([existing_images, empty_images])
            loaded_image_1 = np.vstack([loaded_image_1, np.zeros(empty_images_space)])

        if image_features_1 is not None:
            self.assign_placeholder("ref_features_weight_1", weight_mask_1)
            self.assign_placeholder("ref_img_features_1", image_features_1)
        self.assign_placeholder("ref_weight_1", image_mask_1)
        self.assign_placeholder("ref_img_1", loaded_image_1)

        # set reference images 2

        loaded_image_2 = load_images(images_list_2, self.img_size)
        image_features_2 = None
        # Compute reference images features
        if self.perceptual_model is not None:
            image_features_2 = self.perceptual_model.predict_on_batch(preprocess_input(loaded_image_2))
            weight_mask_2 = np.ones(self.ref_features_weight_2.shape)

        if self.face_mask:
            image_mask_2 = np.zeros(self.ref_weight_2.shape)
            for (i, im) in enumerate(loaded_image_2):
                try:
                    _, img_name = os.path.split(images_list_2[i])
                    mask_img_2 = os.path.join(self.mask_dir, f'{img_name}')
                    if (os.path.isfile(mask_img_2)):
                        print("Loading mask " + mask_img_2)
                        imask = PIL.Image.open(mask_img_2).convert('L')
                        mask = np.array(imask) / 255
                        mask = np.expand_dims(mask, axis=-1)
                    else:
                        mask = self.generate_face_mask(im)
                        imask = (255 * mask).astype('uint8')
                        imask = PIL.Image.fromarray(imask, 'L')
                        print("Saving mask " + mask_img_2)
                        imask.save(mask_img_2, 'PNG')
                        mask = np.expand_dims(mask, axis=-1)
                    mask = np.ones(im.shape, np.float32) * mask
                except Exception as e:
                    print("Exception in mask handling for " + mask_img_2)
                    traceback.print_exc()
                    mask = np.ones(im.shape[:2], np.uint8)
                    mask = np.ones(im.shape, np.float32) * np.expand_dims(mask, axis=-1)
                image_mask_2[i] = mask
            img = None
        else:
            image_mask_2 = np.ones(self.ref_weight_2.shape)

        if len(images_list_2) != self.batch_size:
            if image_features_2 is not None:
                features_space = list(self.ref_features_weight_2.shape[1:])
                existing_features_shape = [len(images_list_2)] + features_space
                empty_features_shape = [self.batch_size - len(images_list_2)] + features_space
                existing_examples = np.ones(shape=existing_features_shape)
                empty_examples = np.zeros(shape=empty_features_shape)
                weight_mask_2 = np.vstack([existing_examples, empty_examples])
                image_features_2 = np.vstack([image_features_2, np.zeros(empty_features_shape)])

            images_space = list(self.ref_weight_2.shape[1:])
            existing_images_space = [len(images_list_2)] + images_space
            empty_images_space = [self.batch_size - len(images_list_2)] + images_space
            existing_images = np.ones(shape=existing_images_space)
            empty_images = np.zeros(shape=empty_images_space)
            image_mask_2 = image_mask_2 * np.vstack([existing_images, empty_images])
            loaded_image_2 = np.vstack([loaded_image_2, np.zeros(empty_images_space)])

        if image_features_2 is not None:
            self.assign_placeholder("ref_features_weight_2", weight_mask_2)
            self.assign_placeholder("ref_img_features_2", image_features_2)
        self.assign_placeholder("ref_weight_2", image_mask_2)
        self.assign_placeholder("ref_img_2", loaded_image_2)

    def optimize(self, vars_to_optimize, iterations=200):
        vars_to_optimize = vars_to_optimize if isinstance(vars_to_optimize, list) else [vars_to_optimize]
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        min_op = optimizer.minimize(self.loss, var_list=[vars_to_optimize])
        self.sess.run(tf.variables_initializer(optimizer.variables()))
        self.sess.run(self._reset_global_step)
        fetch_ops = [min_op, self.loss, self.learning_rate]
        for _ in range(iterations):
            _, loss, lr = self.sess.run(fetch_ops)
            yield {"loss":loss, "lr": lr}
