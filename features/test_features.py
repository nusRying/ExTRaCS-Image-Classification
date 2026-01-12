import cv2
from features.extract_features import extract_features

img = cv2.imread("CleanData/HAM10000/images/ISIC_0024306.jpg")
features = extract_features(img)

print("Total feature length:", len(features))
print(features)
