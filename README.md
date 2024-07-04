

# Image Super Resolution
## Motivation for the Project
This is a project done under IET NITK where the goal is to enhance the brightness of low light image and then pass it to the super resolution model to enhance the visibility and clarity of the image and removing the noise such that the output image is much more clear and visibily pleasing and aesthetically appealing.
## Description

Our proposed model begins by implementing a mechanism to determine whether the input image exhibits characteristics of low-light conditions.Causes of low-light conditions can be due to insufficient or
absent light source or uneven illumination caused by back-light and shadows. Subsequently, we will develop a function capable of discerning whether the given image meets the criteria for low-light classification. Upon identification of a low-light image, we will employ the Zero DCE model to enhance its brightness.

Following the enhancement process through the Zero DCE model, the image will undergo further refinement using the Super Resolution model. This subsequent step aims to produce a substantially clearer and denoised version of the image, leveraging the sophisticated capabilities inherent to the Super Resolution model.

In essence, our model is designed to automatically detect and address low-light scenarios in images, enhance their brightness using Zero DCE, and further refine them to achieve superior clarity and noise reduction through the Super Resolution model. This holistic approach ensures that images exhibiting low-light conditions are effectively processed to yield optimal visual outcomes.
## Working

The Zero-Reference Deep Curve Estimation (Zero DCE) model is a state-of-the-art method in the field of image enhancement, particularly in addressing low-light conditions. Unlike traditional methods that rely on reference images or prior knowledge, the Zero DCE model operates without any reference input, hence the term "zero-reference."

At its core, the Zero DCE model utilizes deep neural networks to predict a transformation curve that can effectively enhance the brightness and visibility of low-light images. This transformation curve is learned directly from the input low-light image itself, without the need for additional reference images or external information.

The key innovation of the Zero DCE model lies in its ability to capture and exploit the inherent characteristics of low-light images to generate accurate and effective enhancement curves. By leveraging deep learning techniques, the model can adaptively adjust the brightness levels of pixels in the input image, effectively amplifying details and enhancing visibility without introducing excessive noise or artifacts.

ResNet architecture, initially developed for image classification, has been repurposed for the task of super-resolution. By embracing residual learning, where the model learns to predict the difference between low and high-resolution images, ResNet-based models excel at preserving critical visual details while enhancing image resolution. This strategy enables the model to focus on capturing fine-grained features crucial for improving image quality.

Integral to ResNet-based super-resolution models are skip connections, facilitating smoother gradient flow through the network and mitigating the vanishing gradient problem. These connections retain valuable information from earlier layers and help the model capture intricate image structures and long-range dependencies more effectively. During training, loss functions like mean squared error or perceptual loss functions measure the discrepancy between model predictions and ground truth images, guiding the optimization process.

Furthermore, ResNet-based super-resolution models employ upsampling techniques like bicubic interpolation or transposed convolutions to upscale low-resolution feature maps, enhancing the visual quality of generated images. This approach finds applications across diverse domains, including medical imaging, satellite imaging, surveillance, and the enhancement of low-quality videos or images. Leveraging ResNet's robust architecture, these models contribute significantly to advancing image processing and computer vision tasks, enabling the generation of high-quality images from low-quality inputs.

## Implementation
##### Image Enhancement using Zero DCE and Super Resolution with CNN

This repository contains the implementation of Zero-Reference Deep Curve Estimation (Zero DCE) for low-light image enhancement and Super Resolution using Convolutional Neural Networks (CNN) for enhancing image resolution.

Zero-Reference Deep Curve Estimation (Zero DCE) is a state-of-the-art method for enhancing the brightness and visibility of low-light images without requiring reference images. It employs deep neural networks to predict a transformation curve that enhances low-light images while preserving important details and features.

The trained Super Resolution model achieves state-of-the-art performance on standard benchmark datasets, demonstrating significant improvements in image quality and resolution compared to baseline methods.

We provide an implementation of the Zero DCE model in Python using TensorFlow/Keras. The code is available in the `code1.py` file. 

To use the Zero DCE model, follow these steps:

1. Follow code given in the code folder.
2. Load your low-light images.
3. Preprocess the images as required (e.g., resizing, normalization).
4. Use the zero_dce function to brighten the images.
5. Save or display the enhanced images.

We provide an implementation of the Super Resolution model in Python using TensorFlow/Keras. The code is available in the `code2.py` file. 

To use the Super Resolution model, follow these steps:

1. Follow code given in the code folder.
2. Load your unclear,blurry images.
3. Preprocess the images as required (e.g., resizing, normalization).
4. Use the super-resolution function to enhance the images.
5. Save or display the enhanced images.







## Conclusion

Together, these models contribute to advancing the field of image processing and computer vision, offering practical solutions for improving image quality in real-world scenarios. By leveraging deep learning techniques, they demonstrate the potential for significant enhancements in various domains, including surveillance, photography, medical imaging, and more.

Incorporating these models into projects and applications can lead to improved visual outcomes and enhanced user experiences. Their flexibility, efficiency, and effectiveness make them valuable assets for researchers, developers, and practitioners seeking to tackle challenges related to image enhancement and restoration.

Overall, the implementation of the Zero-DCE model and Super Resolution using CNN opens up exciting possibilities for enhancing images and unlocking new opportunities in image processing and computer vision.



## Links

[ZERO-DCE](https://li-chongyi.github.io/Proj_Zero-DCE.html)

[SUPER-RESOLUTION](https://pyimagesearch.com/2022/06/06/super-resolution-generative-adversarial-networks-srgan/)
