# SkinLesion-Segm-Classif-UNet-FocusNet-ResNet50

The aim of this project would be to extract segmentations, landmarks, and contours from real-life pictures of lesions in order to detect and classify, using advanced algorithms, the stages of the development of skin cancer. The level of accessibility of the results would then be within the scope of the average users: people with no prior medicine/dermatology knowledge will be able to intuitively understand the results; thus, ensuring that the appropriate action (e.g. book appointment with specialist) will be taken if required. To execute this project, I will employ advanced ML techniques: I will create a Convolutional Neural Network (CNN) based on a U-Net architecture. Furthermore, I will explore the different variants of the U-Net architecture currently existing. I will start my work by investigating and assessing the appropriateness of the vanilla U-Net variant, and will then proceed to compare this with different architectures, eventually culminating with a comparison of the various architectures, training regimes and loss functions for the tasks. The ultimate aim is that the architecture, loss functions and training regimes that are most successful on the benchmarks should inform possible improvements in each of these and thereby extent the state-of-the-art in this area.

Dataset can be downloaded directly from ISIC 2020 challenge's website.

https://challenge.isic-archive.com/data

Data augmentation using the Albumentation library has been applied to the dataset.
<br>
Here are some examples of augmentation performed:<br>

![Augmentation Example](/images/augm_example.png)
Format: ![Augmentation Example](https://postimg.cc/Z0Z035Ln)

