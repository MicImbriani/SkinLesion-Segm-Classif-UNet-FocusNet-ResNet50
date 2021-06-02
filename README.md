# SkinLesion-Segm-Classif-UNet-FocusNet-ResNet50

In this project, the vanilla U-Net, U-Net with Batch Norm, a variation of U-Net using Residual blocks and Squeeze-and-excitation blocks, and the novel FocusNet developed at the University of York will be used for analysing the influence of image segmentation as a pre-processing step for skin lesion classification.
<br>
<br>
After having trained the four models on an augmented version of the ISIC 2017 dataset, they are used to generated prediction mask on the train, validation and test split. These predictions are then used to crop the initial images, effectively generating 4 new datasets. A visual representation of the process is shown below:
<br>
![Cropping Example](/images/cropping_example.png)
<br><br>
These new datasets are used to train 4 different versions of ResNet50 from Keras Application. By performing quantitative tests on the results, it is possible to discuss whether the employment of image segmentation has a positive, negative, or no effect at all on the classification of skin lesions.
<br> 
An example pipeline of what one of the end-to-end networks would look like is shown in the image below:
<br>
![Cropping Example](/images/pipeline.png)

<br><br>
Performing Student's T-Test on the classification AUC ROC results show that a positive correlation exists between the application of increasingly more advanced segmentation architectures and the classification performance.
<br><br><br><br>
Dataset can be downloaded directly from ISIC 2020 challenge's website.
<br>
https://challenge.isic-archive.com/data
<br><br><br><br>
Data augmentation using the Albumentation library has been applied to the dataset.
<br>
Here are some examples of augmentation performed:<br>

![Augmentation Example](/images/augm_example.png)
