import dlib
import numpy as np
from keras.utils import get_file
from utils.utils import unpack_bz2

LANDMARKS_MODEL_URL   = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
RECOGNITION_MODEL_URL = 'http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2'


class FaceRecognizer:
    def __init__(self, tolerance=0.6):
        """
        :param tolerance level for deciding if two encoded images represent the same person
        """

        # recognition tolerance level
        self.tolerance = tolerance

        # get models paths
        predictor_model_path   = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                                     LANDMARKS_MODEL_URL,
                                                     cache_subdir='temp'))
        recognition_model_path = unpack_bz2(get_file('dlib_face_recognition_resnet_model_v1.dat.bz2',
                                                     RECOGNITION_MODEL_URL,
                                                     cache_subdir='temp'))

        # init face detectors, shape_predictors and face recognizer models
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(predictor_model_path)
        self.face_encoder = dlib.face_recognition_model_v1(recognition_model_path)

    def get_encoding(self, im_file):
        """
        :param im_file: image file name
        :return: A list of 128-dimensional face encodings (one for each face in the image)
        """
        im = dlib.load_rgb_image(im_file)
        face_locations = self.detector(im, 1)
        raw_landmarks = [self.shape_predictor(im, face_location) for face_location in face_locations]
        output = [np.array(self.face_encoder.compute_face_descriptor(im, landmarks_set, 1)) for landmarks_set in raw_landmarks]
        return output

    def compare_faces(self, reference_encoding, face_encodings_to_check):
        """
        Compare a list of face encodings against a candidate encoding to see if they match.
        """
        face_distances = np.linalg.norm(face_encodings_to_check - reference_encoding, axis=1)
        return list(face_distances <= self.tolerance), list(face_distances)
