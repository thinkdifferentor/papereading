<!-- TOC -->

- [00 A Review of Deep Learning in Medical Imaging: Imaging Traits, Technology Trends, Case Studies With Progress Highlights, and Future Promises](#00-a-review-of-deep-learning-in-medical-imaging-imaging-traits-technology-trends-case-studies-with-progress-highlights-and-future-promises)
- [01 Survey Study Of Multimodality Medical Image Fusion Methods](#01-survey-study-of-multimodality-medical-image-fusion-methods)
- [02 Deep Learning Techniques for Medical Image Segmentation: Achievements and Challenges](#02-deep-learning-techniques-for-medical-image-segmentation-achievements-and-challenges)
- [03 U-Net and Its Variants for Medical Image Segmentation A Review of Theory and Applications](#03-u-net-and-its-variants-for-medical-image-segmentation-a-review-of-theory-and-applications)
- [04 Fully Convolutional Networks for Semantic Segmentation](#04-fully-convolutional-networks-for-semantic-segmentation)
- [05 SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation](#05-segnet-a-deep-convolutional-encoder-decoder-architecture-for-image-segmentation)
- [06 U-Net: Convolutional Networks for Biomedical Image Segmentation](#06-u-net-convolutional-networks-for-biomedical-image-segmentation)
- [07 Unet++: A nested u-net architecture for medical image segmentation](#07-unet-a-nested-u-net-architecture-for-medical-image-segmentation)
- [08 *Unet 3+: A full-scale connected unet for medical image segmentation](#08-unet-3-a-full-scale-connected-unet-for-medical-image-segmentation)
- [09 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation](#09-3d-u-net-learning-dense-volumetric-segmentation-from-sparse-annotation)
- [10 DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution and Fully Connected CRFs](#10-deeplab-semantic-image-segmentation-with-deep-convolutional-nets-atrous-convolution-and-fully-connected-crfs)
- [11 DeepLabV3: Rethinking Atrous Convolution for Semantic Image Segmentation](#11-deeplabv3-rethinking-atrous-convolution-for-semantic-image-segmentation)
- [12 DeepLabV3+: Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](#12-deeplabv3-encoder-decoder-with-atrous-separable-convolution-for-semantic-image-segmentation)
- [13 Pyramid Scene Parsing Network](#13-pyramid-scene-parsing-network)
- [14 SG-One: Similarity Guidance Network for One-Shot Semantic Segmentation](#14-sg-one-similarity-guidance-network-for-one-shot-semantic-segmentation)
- [15 PANet: Few-Shot Image Semantic Segmentation with Prototype Alignment](#15-panet-few-shot-image-semantic-segmentation-with-prototype-alignment)
- [16 Self-Supervision with Superpixels: Training Few-shot Medical Image Segmentation without Annotation](#16-self-supervision-with-superpixels-training-few-shot-medical-image-segmentation-without-annotation)
- [17 C-CAM: Causal CAM for Weakly Supervised Semantic Segmentation on Medical Image](#17-c-cam-causal-cam-for-weakly-supervised-semantic-segmentation-on-medical-image)
- [18 Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis](#18-self-supervised-pre-training-of-swin-transformers-for-3d-medical-image-analysis)

<!-- /TOC -->

## [00 A Review of Deep Learning in Medical Imaging: Imaging Traits, Technology Trends, Case Studies With Progress Highlights, and Future Promises](./survey/A%20Review%20of%20Deep%20Learning%20in%20Medical%20Imaging%20Imaging%20Traits%2C%20Technology%20Trends%2C%20Case%20Studies%20With%20Progress%20Highlights%2C%20and%20Future%20Promises.pdf)
- Zhou S K, Greenspan H, Davatzikos C, et al./2021/Proceedings of the IEEE/142
- **Traits of Medical Imaging:** 
    1. Medical images have multiple modalities and are dense in pixel resolution; 
    2. Medical image data are isolated and acquired in nonstandard settings; 
    3. The disease patterns in medical images are numerous, and their incidence exhibits a long-tailed distribution; 
    4. The labels associated with medical images are sparse and noisy(In fact, the establishment of gold standards for image labeling remains an open issue); 
    5. Medical image processing and analysis tasks are complex and diverse(*reconstruction, enhancement, restoration, classification, detection, segmentation, registration, multimodality*).
- ![Main traits of medical imaging](./images/A%20Review%20of%20Deep%20Learning%20in%20Medical%20Imaging.png)
- **Key Technologies and Deep Learning:**
    1. **Medical image reconstruction**, which aims to form a visual representation from signals acquired by a medical imaging device, such as a CT or MRI scanner.
    2. **Medical image enhancement**, which aims to adjust the intensities of an image so that the resultant image is more suitable for display or further analysis. (denoising, super resolution, MR bias field correction, image harmonization)
    3. **Medical image segmentation**, which aims to assign labels to pixels so that the pixels with the same label form a segmented object.
    4. **Medical image registration**, which aims to align the spatial coordinates of one or more images into a common coordinate system.
    5. **Computer-aided detection (CADe) and diagnosis (CADx)**: CADe aims to localize or find a bounding box that contains an object (typically a lesion) of interest. CADx aims to further classify the localized lesion as benign/malignant or one of the multiple lesion types.
- **Emerging Deep Learning Approaches:**
    1. **Network Architectures:** 1) Making it deeper: AlexNet, VGG, Inception, ResNet, DenseNet,U-Net; 2) Adversarial and attention mechanisms; 3) Neural architecture search(NAS); 
    2. **Annotation Efficient Approaches:** 1) Transfer Learning; 2) Domain adaptation(A key part of transfer learning); 3) Self-supervised learning; 4) Semisupervised learning; 4) Weakly or partially supervised learning; 5) Unsupervised learning and disentanglement.
    3. **Embedding Knowledge Into Learning** 
    4. **Federated Learning**
    5. **Interpretability:** 1) Model-based interpretability; 2) Posthoc interpretability.
    6. **Uncertainty Quantification**
- In general, most challenges are met by continuous improvement of solutions to the well-known **data challenge**. The community as a whole is continuously developing and improving **TL-based solutions** and **data augmentation schemes**.
- **Multimodality**: One immediate step forward is to **combine the image with additional clinical context**, from patient record to additional clinical descriptors (such as blood tests, genomics, medications, vital signs, and nonimaging data, such as ECG). This step will provide a transition from **image space** to **patient-level** information.
- **Medical Data Collectio**: This step requires building complex infrastructure, along with the generation of new privacy and security regulations—between hospitals and academic research institutes, across hospitals, and in multinational consortia. As more and more data become available, DL and AI will enable unsupervised explorations within the data, thus providing for new discoveries of drugs and treatments toward the advancement and augmentation of healthcare as we know it.


## [01 Survey Study Of Multimodality Medical Image Fusion Methods](./survey/Survey%20Study%20Of%20Multimodality%20Medical%20Image%20Fusion%20Methods.pdf)
- Tawfik N, Elnemr H A, Fakhr M, et al./2021/Multimedia Tools and Applications/20
- Each imaging technique has its **practical characteristics and its limitations**, So there is a need to find new imaging technologies or to develop new fusion methods that are able to merge information from several images with different modalities and take the privileges of the complementary characteristics exhibited in these different images. The multimodality medical image fusion generally focuses on three groups: **MRI-CT, MRI-PET, and MRI-SPECT images**.
- ![Steps of image fusion](./images/Survey%20Study%20Of%20Multimodality%20Medical%20Image%20Fusion%20Methods_1.png)
- Image fusion methods can be categorized into **pixel-level, feature-level (Hybrid of pixel and feature level), and decision-level methods**: 
    1. Pixel-level image fusion is the process of directly merging the original information from the source images or from their multi-resolution transforms to create a more informative final image for visual perception. 
    2. Feature-level fusion aims to extract salient features from the source images such as shape, length, edges, segments, and directions. The extracted features from input images are merged to form more meaningful features, which provide a better descriptive and comprehensive image. 
    3. Decision-level fusion represents a high level of fusion that indicates the actual target. It joins the results from several algorithms to produce an ultimate fusion decision.
- ![Multimodality medical image fusion methods](./images/Survey%20Study%20Of%20Multimodality%20Medical%20Image%20Fusion%20Methods_2.png)
- **Pixel-levvel Fusion:**
    1. Spatial-domain pixel-level fusion: the direct image fusion methods depend on generating each pixel in the fusion result **as the weighted average of the corresponding pixels** in the source images.(**fast and easy to implement / may create block artifacts on image boundaries**)
    2. Transform-domain pixel-level fusion:(Wavelet-based pixel-level fusion & Transform-based pixel-level fusion & Hybrid transform-based pixel-level fusion &  Guided-filtering-based pixel-level fusion &  Intelligent pixel-level fusion) the images are first turned into the **transform domain**, thereafter **the fusion process** is performed in this domain. Finally, an **inverse transformation** process is performed to produce the fusion result. In the transform-domain fusion methods, the choice of the multi-scale decomposition technique is an important issue, in addition to the determination of the decomposition levels. The decomposition techniques have their advantages as well as limitations. To integrate the advantages of different transforms (the last two methods), different fusion methods have been proposed based on hybrid transforms.
- **Feature-level fusion:** The feature-level fusion methods aim to deal with the image features since **the features and objects within a scene are more valuable than the individual pixels**. In the feature-level fusion, the features are extracted separately from each source image, and then the fusion method is applied based on the features from the source images. (increase the reliability of the fusion process， decrease the loss of information and avoid the block artifacts / the fusion process at this level is hard to implement since heterogeneous features)
- **Hybrid Pixel- and Feature-Level Fusion:** The hybrid methods avoid the pixel-level drawbacks such as **high sensitivity to noise** and **blurring effects**. Furthermore, they consider the **information content correlated with each pixel**.
- **Decision-level fusion:** Decision-level fusion aims to merge **higher-level aggregation of the results** from several algorithms to get the final decision for the fusion process. Each image to be fused is first handled individually and then supplied to the fusion algorithm.


## [02 Deep Learning Techniques for Medical Image Segmentation: Achievements and Challenges](./survey/Deep%20Learning%20Techniques%20for%20Medical%20Image%20Segmentation%20Achievements%20and%20Challenges.pdf)
- Hesamian M H, Jia W, He X, et al./2019/Journal of digital imaging/616
- **Approaches/Network Structures:**
    1. *Convolutional Neural Networks (CNNs):* 2D CNN; 2.5D CNN(XY, YZ, and XZ planes); 3D CNN(3D convolution); 
    2. *Fully Convolutional Network (FCN):* FCN for Multi-Organ Segmentation; Cascaded FCN (CFCN); Focal FCN(focal loss); Multi-Stream FCN. ![FCN](./images/Deep%20Learning%20Techniques%20for%20Medical%20Image%20Segmentation_1.png)
    3. *U-Net:* 2D U-Net; 3D U-Net; V-Net. ![U-Net](./images/Deep%20Learning%20Techniques%20for%20Medical%20Image%20Segmentation_2.png)
    4. *Convolutional Residual Networks (CRNs):* to train deeper networks and avoid gradient vanishing and model degradation.
    5. *Recurrent Neural Networks (RNNs):* convolutional-LSTM (CLSTM); Contextual LSTM (CLSTM); GRU + FCN -> RFCN; Clockwork RNN (CW-RNN)
- **Network Training Techniques:**
    1. *Deeply Supervised:* The core idea of deep supervision is to provide the **direct supervision of the hidden layers** and propagate it to lower layers, **instead of just doing it at the output layer**.Their presented results not only show a **better convergence** but also **lower training and validation error**.
    2. *Weakly Supervised:* Existing supervised approaches for automated medical image segmentation require the pixel-level annotation which is not always available in various cases(**tedious and expensive**). 
    3. *Transfer Learning:* 1) fine-tuning the network pre-trained on **general images**; 2) fine-tuning a network pre-trained on **medical images** for a different target organ or task; 
- **Challenges and State-of-the-Art Solutions:**
    1. *Limited Annotated Data:* Collecting such huge dataset of annotated cases in medical image processing is often a very tough task and performing the annotation on new images will also be very tedious and expensive. 1) Data augmentation; 2) Transfer Learning; Patch-Wise Trainin(overlapping or random patches); 3) Weakly Supervised Learning; 4) Sparse Annotation(set weight to unlabeled pixels to zero) 
    2. *Effective Negative Set:* To enhance the discrimination power of the network on false positive cases, the negative set must contain cases which are nodule-like but not positive.
    3. *Class Imbalance:*  Training a network with imbalabce data often leads to the trained network being biased toward the background and got trapped in local minima. A popular solution for this issue is sample re-weighting, where a higher weight is applied to the foreground patches during training. Another approach to deal with this issue is sampled loss in which the loss will not be calculated for the entire image and just some random pixels (areas) will be selected for loss calculation.
    4. *Challenges with Training Deep Models:* 1) Overfitting(increase the size of data, creating multiple views of a patch, dropout); 2) Training Time(Pooling layers, Setting stride, Batch normalization); Gradient Vanishing(Deeply supervised, Careful weight initialization);Organ Appearance(heterogeneous appearance of the target organ -> Increasing the depth of network, ambiguous boundary -> Multi-modality-based approaches); 3D Challenges(computationally expensive);


## [03 U-Net and Its Variants for Medical Image Segmentation A Review of Theory and Applications](./survey/U-Net%20and%20Its%20Variants%20for%20Medical%20Image%20Segmentation%20A%20Review%20of%20Theory%20and%20Applications.pdf)
- Siddique N, Paheding S, Elkin C P, et al./2021/IEEE Access/86
- **BASE U-NET:** contracting path &  expansive path (copy & crop)
- **3D U-NET:** To enables 3D volumetric segmentation, all of the 2D operations are replaced with corresponding 3D operations. The core structure still contains a contracting and expansive path.
- **ATTENTION U-NET:** An often-desirable trait in an image processing network is the ability to focus on specific objects that are of importance while ignoring unnecessary areas. **The attention gate** applies a function in which the feature map is weighted according to each class, and the network can be tuned to focus on a particular class and hence pay attention to particular objects in an image. 
- **INCEPTION U-NET:** Most image processing algorithms tend to use fixed-size filters for convolutions. However, tuning the model to find **the correct filter size** can often be cumbersome.
- ![INCEPTION U-NET](./images/U-Net%20and%20Its%20Variants%20for%20Medical%20Image%20Segmentation_1.png)
- **RESIDUAL U-NET:** The motivation behind ResNet was to overcome the difficulty in training highly deep neural networks. The usage
of residual skip connections helps to alleviate the vanishing gradient problem, thereby allowing for U-net models with deeper neural networks to be designed.
- **RECURRENT U-NET:** The recurrent U-net makes use of **recurrent convolutional neural networks**(RCNN) by incorporating the recurrent
feedback loops into a convolutional layer. The feedback is applied after both convolution and an activation function and feeds the feature map produced by a filter back into the associated layer. The feedback property allows the units to update their feature maps based on context from adjoining units, providing better accuracy and performance.
- **DENSE U-NET:** DenseNet is a deep learning architecture built on top of ResNet with two key changes: **1)** every layer in a block receives the feature or identity map from all of its preceding layers; **2)** the identity maps are combined via channel-wise concatenation into tensors. This allows **1)** DenseNet to preserve all identity maps from prior layers and significantly promote gradient propagation and **2)** ensures that any given layer has contextual information from any of the previous layers in the block. The adoption of dense blocks allows for deeper U-net models, which can segment objects in an image with greater distinction.
- ![DENSE U-NET](./images/U-Net%20and%20Its%20Variants%20for%20Medical%20Image%20Segmentation_2.png)
- **U-NET++:** U-net++ is powerful form of the U-net architecture inspired from DenseNet, which uses a dense network of skip connections as an intermediary grid between the contracting and expansive paths. This aids the network by propagating more semantic information between the two paths, thereby enabling it to segment images more accurately.
- **ADVERSARIAL U-NET:** An adversarial model is a setup in which two networks compete against each other in order to improve their performance.  The key difference in adversarial U-nets is that the goal of the generator is not to produce new images but rather **transformed images**. 
- ![ADVERSARIAL U-NET](./images/U-Net%20and%20Its%20Variants%20for%20Medical%20Image%20Segmentation_3.png)
- **ENSEMBLE U-NET:** 1) Cascading two or more U-nets: each stage for the different levels of segmentation from coarse to fine; 2) Parallel two or more U-nets: parallel training same U-nets to improved segmentation accuracy.


## [04 Fully Convolutional Networks for Semantic Segmentation](./segmentation/Fully%20Convolutional%20Networks%20for%20Semantic%20Segmentation.pdf)
- Long J, Shelhamer E, Darrell T./2015/CVPR/26993
- FCN use **classification networks**(AlexNet, VGG, GoogleNet & ResNet) as backbone and just change the fully connection layer to convolutional layer (keep the number of parameters). Then define a skip architecture that combines semantic information from **a deep, coarse layer** with appearance information from **a shallow, fine layer** to produce accurate and detailed segmentations.
- ![FCN 1](./images/FCN_1.png)
- ![FCN 2](./images/FCN_2.png)
- ![FCN 3](./images/FCN_3.png)
- ![FCN 4](./images/FCN_4.png)
- ![FCN 5](./images/FCN_5.png)


## [05 SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation](./segmentation/SegNet%20A%20Deep%20Convolutional%20Encoder-Decoder%20Architecture%20for%20Image%20Segmentation.pdf)
- Badrinarayanan V, Kendall A, Cipolla R./2017/Pattern Analysis And Machine Learning/11224
- The results of semantic pixel-wise labelling appear coarse. This is primarily because **max pooling and sub-sampling reduce feature map resolution**. Our motivation to design SegNet arises from this need to map low resolution features to input resolution for pixel-wise classification. 
- The increasingly lossy (boundary detail) image representation is not beneficial for segmentation where boundary delineation is vital. Therefore, it is necessary to capture and store boundary information in the encoder feature maps before sub-sampling is performed. However, it's need tons of memory to store all the encode feature maps in pratical applications. SgeNet store only the max-pooling indeces for each encode feature map. 
-  ![SegNet](./images/SegNet.png)


## [06 U-Net: Convolutional Networks for Biomedical Image Segmentation](./segmentation/U-Net%20Convolutional%20Networks%20for%20Biomedical%20Image%20Segmentation.pdf)
- Ronneberger O, Fischer P, Brox T./2015/MICCAI/42233
- The architecture consists of a **contracting path** to capture **context** and a symmetric **expanding path** that enables **precise localization** witch relies on the strong use of **data augmentation** to get more efficient using of the annotated samples.
- **Sliding-window based segmentation method:**
    1. Advantage: First, this network can **localize**. Secondly, the training data in terms of **patches** is much larger than the number of training images.
    2. Disadvantage: First, it is quite slow because the network must be run separately for each patch, and there is a lot of **redundancy** due to overlapping patches. Secondly, there is a **trade-off** between **localization** accuracy and the use of **context**.
- ![Model Architecture](./images/UNet_1.png)
- ![Overlap-tile strategy](./images/UNet_2.png)
- *Many cell segmentation tasks is the separation of **touching objects** of the same class. To handle this issue, this paper propose the use of a **weighted loss**, where the separating background labels between touching cells obtain a large weight in the loss function.
- ![Weighted Loss](./images/UNet_3.png)


## [07 Unet++: A nested u-net architecture for medical image segmentation](./segmentation/Unet%2B%2B%20A%20nested%20u-net%20architecture%20for%20medical%20image%20segmentation.pdf)
- Zhou Z, Rahman Siddiquee M M, Tajbakhsh N, et al./2018/DLMIA/2025
- These encoder-decoder networks used for segmentation share a **key similarity**: skip connections, which combine deep, semantic, coarse-grained feature maps from the decoder sub-network with shallow, low-level, fine-grained feature maps from the encoder sub-network.
- This is in contrast to the plain skip connections commonly used in U-Net, which directly fast-forward high-resolution feature maps from the encoder to the decoder network, **resulting in the fusion of semantically dissimilar feature maps**.
- ![Model Architecture](./images/UNet++.png)
- In summary, UNet++ differs from the original U-Net in three ways: (1) **having convolution layers on skip pathways** (shown in green),
which bridges the semantic gap between encoder and decoder feature maps; (2) **having dense skip connections on skip pathways** (shown in blue), which improves gradient flow; and (3) **having deep supervision** (shown in red), which enables model pruning and improves or in the worst case achieves comparable performance to using only one loss layer.


## [08 *Unet 3+: A full-scale connected unet for medical image segmentation](./segmentation/Unet%203%2B%20A%20full-scale%20connected%20unet%20for%20medical%20image%20segmentation.pdf)
- Huang H, Lin L, Tong R, et al./2020/ICASSP/291
- Unet 3+ takes advantage of **full-scale skip connections** and **deep supervisions**. The full-scale skip connections incorporate **low-level details**(rich spatial information, which highlight the boundaries of organs) with **high-level semantics**(embody position information, which locate where the organs are) from feature maps in different scales; while the deep supervision learns **hierarchical representations** from the full-scale aggregated feature maps. The model improve segmentation accuracy specially for organs that appear at **varying scales** and reduce the parameter of networks to get more **efficient computation**. (compare to U-net & U-net++)
- Main contributions: **(i)** devising a novel UNet 3+ to make full use of the multi-scale features by introducing **full-scale skip connections**, which incorporate low-level details with high-level semantics from feature maps in full scales, but with fewer parameters; **(ii)** developing a **deep supervision to learn hierarchical representations** from the full-scale aggregated feature maps, which optimizes a hybrid loss function to enhance the organ boundary; **(iii)** proposing a **classification-guided module** to reduce over-segmentation on none-organ image by jointly training with an image-level classification;
- ![U-net3+_1](./images/U-net3%2B_1.png)
- To further enhance the boundary of organs, we propose a **[multi-scale structural similarity index (MS-SSIM) Loss](../basemethods/Multiscale%20Structural%20Similarity%20For%20Image%20Quality%20Assessment.pdf)** function to assign higher weights to the fuzzy boundary. Benefiting from it, the UNet 3+ will keep eye on fuzzy boundary as the greater the regional distribution difference, the higher the MS-SSIM value. Adapting **[Focal Loss](../basemethods/Focal%20Loss%20for%20Dense%20Object%20Detection.pdf)** to relieve the example inbalance problem.
- ![U-net3+_2](./images/U-net3%2B_2.png)
- ![U-net3+_3](./images/U-net3%2B_3.png)


## [09 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation](./segmentation/3D%20U-Net%20Learning%20Dense%20Volumetric%20Segmentation%20from%20Sparse%20Annotation.pdf)
- Çiçek Ö, Abdulkadir A, Lienkamp S S, et al./2016/MICCAI/3934
- The 3D U-Net network extends the previous u-net architecture from U-Net by **replacing all 2D operations with their 3D counterparts**(3D convolutions, 3D max pooling, and 3D up-convolutional layers). There are two use cases of this model:  (1) In a **semi-automated setup**, the user annotates some slices in the volume to be segmented.
The network learns from these sparse annotations and provides a dense 3D segmentation. (2) In a **fully-automated setup**, we assume that a representative, sparsely annotated training set exists. Trained on this data set, the network densely segments new volumetric images.
- Annotation of large volumes in a slice-by-slice manner is very tedious. It is inefficient, too, since **neighboring slices show almost the same information**.
- ![3D U-Net](./images/3DUNet.png)


## [10 DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution and Fully Connected CRFs](./segmentation/DeepLab%20Semantic%20Image%20Segmentation%20with%20Deep%20Convolutional%20Nets%2C%20Atrous%20Convolution%20and%20Fully%20Connected%20CRFs.pdf)
- Chen L C, Papandreou G, Kokkinos I, et al./2017/Pattern Analysis and Machine Intelligence/11453
- Three **challengens** of  DCNNs to semantic image segmentation: **(1)** reduced feature resolution(caused by repeated combination of max-pooling and downsampling->**atrous convolution**), **(2)** existence of objects at multiple scales(using multiple parallel atrous convolutional layers with different sampling rates, called **ASPP**), and **(3)** reduced localization accuracy due to DCNN invariance (fully connected Conditional Random Field, **CRF**). The DeepLab have three main advantages: *(1) Speed; (2) Accuracy; (3)Simplicity*
- ![DeepLabV1_1](./images/DeepLabV1_1.png)
- ![DeepLabV1_2](./images/DeepLabV1_2.png)
- ![DeepLabV1_3](./images/DeepLabV1_3.png)
- ![DeepLabV1_4](./images/DeepLabV1_4.png)
- ![DeepLabV1_5](./images/DeepLabV1_5.png)


## [11 DeepLabV3: Rethinking Atrous Convolution for Semantic Image Segmentation](./segmentation/DeepLabV3%20Rethinking%20Atrous%20Convolution%20for%20Semantic%20Image%20Segmentation.pdf)
- Chen L C, Papandreou G, Schroff F, et al./2017/CVPR/4868
-  Deep Convolutional Neural Networks (DCNNs) for the semantic segmentation task have two challenges: **(1)** reduced feature resolution(**atrous convolution**); **(2)** existence of objects at multiple scales(**atrous convolution & spatial pyramid pooling**). In DeepLab V3, the authors take different strategy to handle these issues.
- ![DeepLabV3_1](./images/DeepLabV3_2.png)
- ![DeepLabV3_2](./images/DeepLabV3_1.png)
- ![DeepLabV3_3](./images/DeepLabV3_3.png)


## [12 DeepLabV3+: Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](./segmentation/DeepLabV3%2B%20Encoder-Decoder%20with%20Atrous%20Separable%20Convolution%20for%20Semantic%20Image%20Segmentation.pdf)
- Chen L C, Zhu Y, Papandreou G, et al./2018/ECCV/6537
- The former networks are able to **encode multi-scale contextual information** by probing the incoming features with filters or pooling operations at multiple rates and multiple effective fields-of-view, while the latter networks can **capture sharper object boundaries** by gradually recovering the spatial information. DeepLabv3+ extends DeepLabv3 by adding a **simple yet effective decoder module** to refine the
segmentation results especially along object boundaries and apply the **depthwise separable convolution**(depthwise convolution + pointwise convolution, to reduce the model parameter) to both ASPP and decoder modules.
- DeepLabv3+'s work: 1) powerful encoder with atrous convolution and simple yet effective decoder; 2) adapt the Xception model to get faster and stronger encoder-decoder network.
- ![DeepLabV3+_1](./images/DeepLabV3%2B_1.png)
- ![DeepLabV3+_2](./images/DeepLabV3%2B_2.png)
- ![DeepLabV3+_3](./images/DeepLabV3%2B_3.png)


## [13 Pyramid Scene Parsing Network](./segmentation/Pyramid%20Scene%20Parsing%20Network.pdf)
- Zhao H, Shi J, Qi X, et al./2017/CVPR/7411
- Important Observations about FCN in ADE20K: 1) **Mismatched Relationship**: FCN is lack of the ability to **collect contextual information** increases the chance of misclassification. 2) **Confusion Categories**: There are confusing classfication in the ADE20K like field and earth, mountain and hill etc. FCN can not judge them in the segmentation task. This problem can be remedied by utilizing **the relationship between categories**. 3) **Inconspicuous Classes**: Serveral small-size things are hard to find while may be greate importance like streetlights and signboard. However, big objects or stuff may exceed the receptive field of FCN and thus cause discontinuous prediction. To improve performance for remarkably small or large objects, one should **pay much attention to different sub-regions** that contain inconspicuous-category stuff.
- ![FPN](./images/PSPNet.png)


## [14 SG-One: Similarity Guidance Network for One-Shot Semantic Segmentation](./segmentation/SG-One%20Similarity%20Guidance%20Network%20for%20One-Shot%20Semantic%20Segmentation.pdf)
- Zhang X, Wei Y, Yang Y, et al./2020/IEEE Transactions on Cybernetics/177
- Main Contribution: 1) We propose to produce robust object-related representative vectors using **masked average pooling**(Inspiron next work) for incorporating contextual information without changing the input structure of networks. (2) We produce the **pixel-wise guidance** using **cosine similarities** between representative vectors and query features for predicting the segmentation masks.
- ![SG-One_1](./images/SG-One_1.png)
- ![SG-One_2](./images/SG-One_2.png),![SG-One_3](./images/SG-One_3.png)
- ![SG-One_4](./images/SG-One_4.png)
  

## [15 PANet: Few-Shot Image Semantic Segmentation with Prototype Alignment](./segmentation/PANet%20Few-Shot%20Image%20Semantic%20Segmentation%20with%20Prototype%20Alignment.pdf)
- Wang K, Liew J H, Zou Y, et al./2020/ICCV/302
- PANet learns **class specific prototype representations** from a few support images within an embedding space and then performs segmentation over the query images through **matching each pixel to the learned prototypes** (Segmentation over the query images
is performed by labeling each pixel as the class of the nearest prototype). With non-parametric metric learning, PANet offers **high-quality prototypes** that are representative for each semantic class and meanwhile discriminative for different classes.
- ![PANet1](./images/PANet_1.png)
- **Key Steps:** 1) Prototype learning (adopted **masked average pooling**); 2) Non-parametric metric learning(adopted **cos similarity with factor alpha**) 3) Prototype alignment regularization(**swapping the support and query set**): Intuitively, if the model can predict a good segmentation mask for the query using prototypes extracted from the support, the prototypes learned from the query set based on the predicted masks should be able to segment support images well.
- ![PANet_2](./images/PANet_2.png)
- ![PANet_3](./images/PANet_3.png) , ![PANet_4](./images/PANet_4.png)
  

## [16 Self-Supervision with Superpixels: Training Few-shot Medical Image Segmentation without Annotation](./segmentation/Self-Supervision%20with%20Superpixels%20Training%20Few-shot%20Medical%20Image%20Segmentation%20without%20Annotation.pdf)
- Ouyang C, Biffi C, Chen C, et al./2020/ECCV/54
- Most of the existing Few-shot semantic segmentation (FSS) techniques require abundant (compare traditional segmentation model is **much fewer**) annotated semantic classes for training.To address this problem we make several contributions:(1) A novel self-supervised FSS framework for medical images in order to **eliminate the requirement for annotations** during training. Additionally, **superpixel-based pseudo-labels** are generated to provide supervision;(2) An **adaptive local prototype pooling** module plugged into prototypical networks, to solve the common challenging **foreground-background imbalance** problem in medical image segmentation;
- The aim of few-shot segmentation is to obtain a model that can segment an **unseen semantic class**(Dtrain's Query set), by just learning from a **few labeled images**(Dtrain's Support set) of this unseen class during inference without retraining the model. Dataset: **Dtrain & Dtest** have the same structurs but the classes of them is totally different eg. **SupSet = {Image, Mask}, QurSet = {Image}**  
- ![Self-Supervision with Superpixels 1](./images/Self-supervision%20with%20Superpixels_1.png)
- ![Self-Supervision with Superpixels 2](./images/Self-supervision%20with%20Superpixels_2.png)
- ![Self-Supervision with Superpixels 3](./images/Self-supervision%20with%20Superpixels_3.png) , ![Self-Supervision with Superpixels 4](./images/Self-supervision%20with%20Superpixels_4.png) 
- ![Self-Supervision with Superpixels 5](./images/Self-supervision%20with%20Superpixels_5.png)
  

## [17 C-CAM: Causal CAM for Weakly Supervised Semantic Segmentation on Medical Image](./segmentation/C-CAM%20Causal%20CAM%20for%20Weakly%20Supervised%20Semantic%20Segmentation%20on%20Medical%20Image.pdf)
- Chen Z, Tian Z, Zhu J, et al./2022/CVPR/-
-  Main challenges of medical images. **Challenge1:** The object boundary for medical image is more ambiguous than natural image. **Challenge2:** Different organs often occur in the same medical image in training stage. To deal with those challenges, C-CAM(Causality CAM) proposal two cause-effect chains. The **category-causality**(designed to alleviate the problem of **ambiguous boundary**) chain represents the **image content** (cause) affects the **category** (effect). The **anatomy-causality**(designed to solve the co-occurrence problem) chain represents the **anatomical structure** (cause) affects the **organ segmentation** (effect).
- ![C-CAM_1](./images/C-CAM_1.png)
- Common pipeline of CAM-based method could be divided into three stages. **The first stage** is to generate seed regions with CAM method. **The second stage** is to refine seeds regions to generate pseudo masks.(Most work in the stage) **The last stage** is to train segmentation model with pseudo masks.
- **Question 1:** why the accuracy of classification model is very high but the activated region of CAM is not accurate? **Question 2:** why the shape of activated region differs far from the groundtruth contour of object? The answer for the first question is that classification model is essentially an **association model**, which performs well in classification task. The answer for the second question is that current learning-based methods ignore **constraint of output structure** since they use pixel-wise loss functions.
- ![C-CAM_2](./images/C-CAM_2.png)
- ![C-CAM_3](./images/C-CAM_3.png)
- ![C-CAM_4](./images/C-CAM_4.png)


## [18 Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis](./segmentation/Self-Supervised%20Pre-Training%20of%20Swin%20Transformers%20for%203D%20Medical%20Image%20Analysis.pdf)
- Tang Y, Yang D, Li W, et al./2022/CVPR/10
- The **contrastive learning** is used to differentiate various ROIs of different body compositions, whereas the **inpainting** allows for learning the texture, structure and correspondence of masked regions to their surrounding context. The **rotation** task serves as a mechanism to learn the structural content of images and generates various sub-volumes that can be used for contrastive learning.
- ![Swin-UNETR_1](./images/Swin-UNETR_1.png)
- ![Swin-UNETR_2](./images/Swin-UNETR_2.png)