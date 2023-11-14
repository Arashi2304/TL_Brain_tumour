# TL_Brain_tumour
Attempt at using a CNN to detect the presence of a brain tumour in MRI scans of the brain. Since the number of samples is low, we transfer the weights of the convolutional blocks from VGG16 (imagenet) and train the FC layers. Additionally, we apply the Laplacian filter to the images and see if that improves the accuracy of the model (same architecture).
