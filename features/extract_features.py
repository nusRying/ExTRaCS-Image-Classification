import numpy as np
from features.lbp import extract_lbp
from features.glcm import extract_glcm
from features.semantic_features import extract_semantic_features

def extract_features(image):
    """
    Extract combined handcrafted features from an image.
    
    Features (27 total):
    - LBP histogram: 10 features
    - GLCM properties: 5 features
    - Semantic/clinical: 12 features
    
    Returns a 1D numpy array.
    """

    lbp_feat = extract_lbp(image)              # 10 features
    glcm_feat = extract_glcm(image)            # 5 features
    semantic_feat = extract_semantic_features(image)  # 12 features

    features = np.concatenate([
        lbp_feat,
        glcm_feat,
        semantic_feat
    ])

    return features.astype(np.float32)