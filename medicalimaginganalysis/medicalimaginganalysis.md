<!-- TOC -->

- [00 A Review of Deep Learning in Medical Imaging: Imaging Traits, Technology Trends, Case Studies With Progress Highlights, and Future Promises](#00-a-review-of-deep-learning-in-medical-imaging-imaging-traits-technology-trends-case-studies-with-progress-highlights-and-future-promises)
- [01 Survey Study Of Multimodality Medical Image Fusion Methods](#01-survey-study-of-multimodality-medical-image-fusion-methods)

<!-- /TOC -->

## [00 A Review of Deep Learning in Medical Imaging: Imaging Traits, Technology Trends, Case Studies With Progress Highlights, and Future Promises](./A%20Review%20of%20Deep%20Learning%20in%20Medical%20Imaging%20Imaging%20Traits%2C%20Technology%20Trends%2C%20Case%20Studies%20With%20Progress%20Highlights%2C%20and%20Future%20Promises.pdf)
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


## [01 Survey Study Of Multimodality Medical Image Fusion Methods](./Survey%20Study%20Of%20Multimodality%20Medical%20Image%20Fusion%20Methods.pdf)
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