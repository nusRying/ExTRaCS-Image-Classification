import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops

def extract_glcm(
    image,
    distances=[1],
    angles=[0],
    levels=256
):
    """
    Extract GLCM texture features from an image.

    Returns:
    - 1D numpy array of GLCM features
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    glcm = graycomatrix(
        gray,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True
    )

    features = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
    ]

    return np.array(features, dtype=np.float32)
