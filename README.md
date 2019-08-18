# MCGAN
Multimodal Brain MRI Translation Based on Modular CycleGAN  
## Abstract
In many medical images processing tasks, researchers and doctors expect to improve the feasibility of diagnosis and treatment through obtaining registered multimodal medical images. However, it is very difficult to obtain sufficient registered multimodal medical images. In this paper, based on the current most popular unsupervised image translation model CycleGAN, we propose an improved scheme for medical image translation, which can generate registered multimodal from single modality. We improve parameter initialization method, upsampling method and loss function to speed up model training and improve translation quality. Compared with previous studies that focus only on the overall quality of translation, we focus more on the lesion information in medical images, so we propose a method for the preservation of lesion information in translation process. We perform a series of multimodal translation experiments on the BRATS2015 dataset, verify the effect of each of our improvements, and verify the consistency of the lesion information between translation images and original images, also verify the effectiveness and availability of the lesion information in translation images.  
## Environment
tensorflow-gpu-1.9, python2.7, 4-gpu
## Example
first you need to prepare your dataset and normalize them (except the lesion label).  
then train our simple segment model:  
```
python seg_train.py
```
after finish segment model training, we train the translate model:  
```
python train.py --load_seg_model xxx
```

