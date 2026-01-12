import numpy as np
import pywt

def _entropy(x, eps=1e-10):
    """Shannon entropy of coefficients"""
    x = np.abs(x).ravel()
    p = x / (np.sum(x) + eps)
    return -np.sum(p * np.log2(p + eps))


def _band_stats(coeffs):
    """Return mean, std, energy, entropy"""
    coeffs = np.asarray(coeffs)
    mean = np.mean(coeffs)
    std = np.std(coeffs)
    energy = np.mean(coeffs ** 2)
    entropy = _entropy(coeffs)
    return [mean, std, energy, entropy]


def extract_wavelet_features(
    image,
    wavelet="db2",
    level=2
):
    """
    Extract pyramid wavelet features from RGB image.

    Parameters
    ----------
    image : ndarray (H, W, 3)
        RGB image
    wavelet : str
        Wavelet type (db2 recommended)
    level : int
        Decomposition levels

    Returns
    -------
    features : list
        Wavelet feature vector
    """

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected RGB image (H,W,3)")

    features = []

    # Process each color channel independently
    for ch in range(3):
        channel = image[:, :, ch].astype(np.float32)

        coeffs = pywt.wavedec2(channel, wavelet=wavelet, level=level)

        # coeffs format:
        # [LL, (LH1, HL1, HH1), (LH2, HL2, HH2), ...]
        for lvl in range(1, len(coeffs)):
            LH, HL, HH = coeffs[lvl]

            features.extend(_band_stats(LH))
            features.extend(_band_stats(HL))
            features.extend(_band_stats(HH))

        # Approximation coefficients (LL at last level)
        LL = coeffs[0]
        features.extend(_band_stats(LL))

    return features
