import numpy as np
import cv2
from skimage.feature import local_binary_pattern

def extract_lbp(
    image,
    radius=1,
    n_points=8,
    method="uniform",
    n_bins=10
):
    """
    Extract uniform LBP histogram features from an image.

    Parameters:
    - image: BGR image (OpenCV)
    - radius: LBP radius
    - n_points: number of neighbors
    - method: LBP method
    - n_bins: histogram bins

    Returns:
    - 1D numpy array of LBP features
    """

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute LBP
    lbp = local_binary_pattern(gray, n_points, radius, method)

    # Compute histogram
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=n_bins,
        range=(0, n_bins),
        density=True
    )

    return hist.astype(np.float32)
