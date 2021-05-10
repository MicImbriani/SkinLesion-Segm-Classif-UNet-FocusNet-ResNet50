import albumentations as A
from albumentations.augmentations import transforms
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

path = "D:/Users/imbrm/ISIC_2017-2/Train/ISIC_0000000.png"
path2 = "D:/Users/imbrm/ISIC_2017-2/Train_GT_masks/ISIC_0000000_segmentation.png"

image = cv2.imread(path, 0)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = np.asarray(image)
mask = np.asarray(cv2.imread(path2))


transform = A.Compose([
    A.ElasticTransform(alpha=30, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
    A.CLAHE(p=1),
    A.GaussNoise(p=1),
    A.Resize(256,256),
    A.ToGray(p=1),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomRotate90(p=1),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
])
for i in range(0, 11):
    r = random.randint(0, 1e9)
    random.seed(r) 
    # augmented_image = transform(image=image)['image']

    transformed = transform(image=image, mask=mask)
    transformed_image = transformed['image']
    transformed_mask = transformed['mask']

    #print(type(transformed_image))

    # img = img.astype(np.float64)
    # img = Image.fromarray(img)
    # img = img.convert("L")
    # img.save(save_path + "/" + image)

    plt.imshow(transformed_image)
    
    plt.waitforbuttonpress()
    plt.imshow(transformed_mask)
    plt.waitforbuttonpress()