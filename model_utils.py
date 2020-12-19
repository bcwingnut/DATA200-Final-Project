import json
import xgboost as xgb
import sklearn
from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.cluster import KMeans
from scipy import ndimage
from scipy.spatial import distance
from skimage import data
from skimage import io
import numpy as np
import pandas as pd
import cv2
import time
from math import sqrt
from collections import defaultdict
import torch
from featexp import get_univariate_plots
import mahotas

import matplotlib.pyplot as plt

class_names=['Airplanes', 'Bear', 'Blimp', 'Comet', 'Crab', 'Dog', 'Dolphin',
             'Giraffe', 'Goat', 'Gorilla', 'Kangaroo', 'Killer-Whale',
             'Leopards', 'Llama', 'Penguin', 'Porcupine', 'Teddy-Bear',
             'Triceratops', 'Unicorn', 'Zebra']

def sample_image(df, show=False):

    classes = [
        'airplanes', 'bear', 'blimp', 'comet', 'crab', 'dog', 'dolphin',
        'giraffe', 'goat', 'gorilla', 'kangaroo', 'killer-whale', 'leopards',
        'llama', 'penguin', 'porcupine', 'teddy-bear', 'triceratops',
        'unicorn', 'zebra'
    ]

    item = df.sample(n=1, random_state=np.random.RandomState()).iloc[0]
    pic = item['Pictures']
    label = item['Encoding']
    print(pic.shape)
    print('Class Label: ' + classes[label])
    if show:
        io.imshow(pic)
    return pic


def normalize_images(images, gpu=False):
    """ Normalize the image using ZCA (zero component analysis) according to this paper:
        https://ieeexplore.ieee.org/document/7808140        
        
        args:
        - images: sequence of images, either ndarray or pandas series each element representing a single image as a ndarray
        - gpu: boolean (default: False), if True uses torch.svd. Cuda and torch capability required.
        do not use gpu=True option until pytorch fixes the SVD slowness issue.
    """
    if isinstance(images, pd.Series):
        images = np.stack(images.values)
    # flatten to 1D (n, K*K*3) with K being the image dimension
    n = images.shape[0]
    img_dims = images.shape[1:]
    X = images.reshape(images.shape[0],
                       images.shape[1] * images.shape[2] * images.shape[3])
    # min max scale to [0, 1]
    X = X / 255
    assert X.min() >= 0.0 and X.max() <= 1.0
    # center the images around per-pixel mean
    X_norm = X - np.mean(X, axis=0)
    # calculate the covariance matrix since each row is an "observation" we do rowvar = False
    cov = np.cov(X_norm, rowvar=False)
    # SVD
    t1 = time.time()
    print("Computing SVD...")
    # covariance matrix is hermitian (symmetric) by construct so use the faster algorithm
    # it doesnt matter if full_matrices is False/True in this case since matrix is square
    # if gpu and torch is available use the torch svd for speed
    if gpu and torch.cuda.is_available():
        cov_tensor = torch.from_numpy(cov)
        u, s, vt = torch.svd(cov_tensor)
        u = u.numpy()
        s = s.numpy()
        vt = vt.numpy()
    # if torch/gpu is not available use the fact that covariance matrix is hermitian
    else:
        u, s, vt = np.linalg.svd(cov, hermitian=True)
    print("SVD Finished: {0}".format(time.time() - t1))
    epsilon = 0.1
    X_zca = u @ np.diag(1.0 / np.sqrt(s + epsilon)) @ vt @ X_norm.T
    X_zca = X_zca.T

    # min max scale the images to [0, 1]
    X_zca = (X_zca - X_zca.min()) / (X_zca.max() - X_zca.min())

    # reshape back to (n, K, K, 3) dimensions
    X_zca = X_zca.reshape(n, img_dims[0], img_dims[1], img_dims[2])
    print("Final Shape: " + str(X_zca.shape))
    # just to get back the original dimensions for pd.Series
    return pd.Series(np.split(X_zca, n)).apply(lambda s: np.squeeze(s))


class BOVWVectorizer(object):
    def __init__(self, encoding='rgb'):
        self.encoding = encoding
        self.visual_words = None  # uninitialized
        self.features = None  # uninitialized
        self.std = None


    def _get_sift_features(self, sift, image):
        """ Get sift keypoints and descriptors for a single RGB image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        kps, des = sift.detectAndCompute(gray, None)
        return des

    def fit(self, images, num_clusters=None, n_init=10):
        """Learn the image feature vocabulary using all the images in the corpus.
        - images: pandas Series shape (n, ) or numpy ndarray shape (n, w, h, c) of RGB images.
        - labels: pandas Series or nump ndarray of image classes corresponding to images
        - n_clusters: number of clusters to form for KMeans component. 
            Default is sqrt of size of SIFT features.
        - n_iter: number of time k-means algorithm runs with different centroid seeds
        """

        if isinstance(images, pd.Series):
            images = np.stack(images.values)

        sift = cv2.SIFT_create()

        n = images.shape[0]
        descriptors = []

        print("Computing SIFT feature descriptors")
        for i in range(n):
            image = images[i]
            des = self._get_sift_features(sift, image)
            # in the instance that there is no SIFT features for an image
            # print the index and pop it from labels
            if isinstance(des, type(None)) or not des.any() or des.size == 0:
                print(
                    "WARN: Image at index: {} does not have SIFT features.\nOutput size will shrink."
                    .format(i))
            else:
                descriptors.append(des)
        stacked_descriptors = np.vstack(descriptors)
        self.std = np.std(stacked_descriptors, axis=0)
        features = stacked_descriptors / self.std

        print("Stacked descriptors shape {}".format(stacked_descriptors.shape))

        # do kmeans for descriptors
        if num_clusters == None:
            num_clusters = int(sqrt(stacked_descriptors.shape[0]))

        print("Computing KMeans with {} clusters".format(num_clusters))
        kmeans = KMeans(num_clusters, n_init=n_init)

        # people recommend standardization (unit variance) before KMeans
        # but Im not too sure about the outcome.
        kmeans.fit(features)
        self.visual_words = kmeans.cluster_centers_

        print("Visual vocabulary is built.")

        return descriptors

    def fit_transform(self, images, num_clusters=None, n_init=10):
        """Learn the visual vocabulary and return the term document matrix.
        In this case the term-document matrix is referring to the image-histogram
        matrix for the given images.

        Returns:
        - numpy ndarray of shape (n_samples, n_clusters)
        """
        from scipy.cluster.vq import whiten, vq
        from numpy import histogram

        descriptors = self.fit(images, num_clusters, n_init)

        histograms = []
        # iterate over the keys (labels) in the dictionary
        # compute the histogram for each label

        print("Creating image histograms...")

        for img_descriptor in descriptors:
            code, dist = vq(img_descriptor/self.std, self.visual_words)
            histogram_of_visual_words, bin_edges = histogram(
                code, bins=range(self.visual_words.shape[0] + 1))
            histograms.append(histogram_of_visual_words)

        print("\nDone.")

        return np.array(histograms)

    def transform(self, images):
        """
        """
        from scipy.cluster.vq import whiten, vq
        from numpy import histogram

        # convert to ndarray of (n, channels)
        if isinstance(images, pd.Series):
            images = np.stack(images.values)

        sift = cv2.SIFT_create()
        
        histograms = []
        for i, image in enumerate(images, 0):
            des = self._get_sift_features(sift,image)
            # in the instance that there is no SIFT features for an image
            # print the index and pop it from labels
            if isinstance(des, type(None)) or not des.any() or des.size == 0:
                print(
                    "Image at index: {} does not have SIFT features.\nOutput size will shrink."
                    .format(i))
            else:
                code, dist = vq(des/self.std, self.visual_words)
                histogram_of_visual_words, bin_edges = histogram(
                    code, bins=range(self.visual_words.shape[0] + 1))
                histograms.append(histogram_of_visual_words)

        return np.array(histograms)


class XGBoostModel(object):
    PARAMS = {
        'max_depth': 5,
        'eta': 0.5,
        'objective': 'multi:softmax',
        'num_class': 20,
        'n_estimators': 1000,
        'tree_method': 'hist'  # gpu_hist for gpu 
    }

    def __init__(self, columns, model_fp=''):
        self.bst = xgb.Booster()
        self.columns = columns

    def predict(self, features, verbose=False, ntree_limit=0):
        dset = xgb.DMatrix(features.to_numpy(), feature_names=self.columns)
        return self.bst.predict(dset, ntree_limit=ntree_limit)

    def train(self,
              train_features,
              train_labels,
              test_features=None,
              test_labels=None,
              model_params=None,
              n_iter=300,
              verbose=False,
              xgb_model=None):

        if not model_params:
            model_params = self.PARAMS

        dtrain = xgb.DMatrix(train_features.to_numpy(),
                             label=train_labels.to_numpy(),
                             feature_names=self.columns)
        evals = [(dtrain, 'train')]

        dtest = None
        if test_features is not None and test_labels is not None:
            if not test_features.empty and not test_labels.empty:
                dtest = xgb.DMatrix(test_features.to_numpy(),
                                    label=test_labels.to_numpy(),
                                    feature_names=self.columns)
                evals.append((dtest, 'test'))

        self.bst = xgb.train(model_params,
                             dtrain,
                             n_iter,
                             evals,
                             verbose_eval=verbose,
                             xgb_model=xgb_model)
        return self.bst

    def cv_error(
        self,
        train_features,
        train_labels,
        model_params=None,
        n_iter=300,
        verbose=False,
        cv=None,
    ):

        nsplits = cv if cv else 5

        if not model_params:
            model_params = self.PARAMS

        kf = KFold(n_splits=nsplits)

        training_errors = []
        validation_errors = []

        first_iter = True

        for train_idx, validation_idx in kf.split(train_features):

            split_train_features, split_validation_features = train_features.iloc[
                train_idx], train_features.iloc[validation_idx]
            split_train_labels, split_validation_labels = train_labels.iloc[
                train_idx], train_labels.iloc[validation_idx]

            if first_iter:
                first_iter = False
                self.train(split_train_features, split_train_labels,
                           split_validation_features, split_validation_labels)

            else:
                self.train(split_train_features,
                           split_train_labels,
                           split_validation_features,
                           split_validation_labels,
                           xgb_model=self.bst)

            # calculate the errors

            training_error = 1 - accuracy_score(
                split_train_labels.to_numpy(),
                self.predict(split_train_features))
            validation_error = 1 - accuracy_score(
                split_validation_labels.to_numpy(),
                self.predict(split_validation_features))

            training_errors.append(training_error)
            validation_errors.append(validation_error)

        return (np.mean(training_errors), np.mean(validation_errors))

    def save_model(self, model_fp, config_fp):
        self.bst.save_model(model_fp)
        config = {"xg_features": self.columns}

        with open(config_fp, 'w') as f:
            json.dump(config, f)

    def plot_importances(self, savefig=False):
        _, ax = plt.subplots(1, 3, figsize=(20, 6))
        xgb.plot_importance(
            self.bst,
            importance_type="weight",
            title="Weight: number of times a feature appears in a tree",
            ax=ax[0])

        xgb.plot_importance(
            self.bst,
            importance_type="gain",
            title="Gain: avg gain of splits which use the feature",
            ax=ax[1])

        xgb.plot_importance(
            self.bst,
            importance_type="cover",
            title="Coverage: avg coverage of splits which use the feature",
            ax=ax[2])

        if savefig:
            plt.savefig('importances.png', dpi=300, bbox_inches='tight')
