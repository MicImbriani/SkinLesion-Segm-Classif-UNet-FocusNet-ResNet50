import albumentations as A
from albumentations.augmentations import transforms
import random
import cv2
import numpy as np

path = "D:/Users/imbrm/ISIC_2017-2/Train/ISIC_0000000.png"
path2 = "D:/Users/imbrm/ISIC_2017-2/Train_GT_masks/ISIC_0000000_segmentation.png"

image = cv2.imread(path, 0)
image = np.asarray(image)
mask = np.asarray(cv2.imread(path2))



transform = A.Compose([
    A.OneOf([
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
        A.GridDistortion(p=0.5),
        A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)                  
        ], p=1),
    A.CLAHE(p=1),
    A.GaussNoise(p=1),
    A.ToGray(p=1),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomRotate90(p=1),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
])
random.seed(42) 
# augmented_image = transform(image=image)['image']
# visualize(augmented_image)

#transformed = transform(image=image, mask=mask)
tr = A.CLAHE(p=1)
transformed_image = tr(image=image)['image']
#transformed_image = transformed['image']
#transformed_mask = transformed['mask']

#print(type(transformed_image))

img = img.astype(np.float64)
img = Image.fromarray(img)
img = img.convert("L")
img.save(save_path + "/" + image)