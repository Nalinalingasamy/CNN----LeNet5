# CNN----LeNet5
## LeNet-5 Architecture

LeNet-5 is one of the earliest and most influential Convolutional Neural Network (CNN) architectures, designed by Yann LeCun in 1998 for handwritten digit recognition (e.g., the MNIST dataset). It laid the foundation for modern CNNs used in computer vision tasks.

## Key Features:

Deep Convolutional Layers: LeNet-5 uses a stack of convolutional and pooling layers to automatically extract features from raw input images.
Efficient and Lightweight: With only around 60,000 parameters, LeNet-5 is computationally efficient and was designed to run on the hardware available at the time.
Simple yet Effective: Despite its simplicity, LeNet-5 was a breakthrough in demonstrating the power of deep learning for image recognition tasks.

## Architecture Overview:

LeNet-5 consists of 7 layers, including convolutional, subsampling (pooling), and fully connected layers. The architecture is as follows:

Input Layer: Accepts a 32x32 grayscale image (MNIST digits are typically 28x28, so the image is zero-padded to 32x32).

C1 (Convolutional Layer): 6 convolutional filters of size 5x5 are applied to the input image, producing 6 feature maps of size 28x28.

S2 (Subsampling/Pooling Layer): Average pooling is applied with a 2x2 filter and stride of 2, reducing each feature map to 14x14.

C3 (Convolutional Layer): 16 convolutional filters of size 5x5 are applied to the pooled feature maps from S2, resulting in 16 feature maps of size 10x10.

S4 (Subsampling/Pooling Layer): Another average pooling layer with a 2x2 filter and stride of 2, reducing the size of each feature map to 5x5.

C5 (Fully Connected Convolutional Layer): A fully connected layer with 120 units, connecting all 16 feature maps (flattened) to 120 neurons.

F6 (Fully Connected Layer): 84 units, which connect the output from the previous layer to 84 neurons.

Output Layer: A fully connected layer with 10 units, corresponding to the 10 possible classes (for MNIST digits 0-9).

## How It Works:

Convolutional Layers: Learn spatial hierarchies of features by convolving the input with learned filters.
Pooling Layers: Reduce the dimensionality of feature maps and retain the most important information, improving computational efficiency and reducing overfitting.
Fully Connected Layers: Classify the extracted features into one of the predefined classes (in the case of MNIST, digits 0-9).
LeNet-5 was a pioneer in the development of CNNs and remains a significant milestone in the field of deep learning and image classification.

