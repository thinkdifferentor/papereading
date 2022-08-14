# Weekly

# 00. 20220527

- **阅读深度学习在医疗影像分析领域相关研究的综述：**
A Review of Deep Learning in Medical Imaging: Imaging Traits, Technology Trends, Case Studies With Progress Highlights, and Future Promises;
 Survey Study Of Multimodality Medical Image Fusion Methods; 
Deep Learning Techniques for Medical Image Segmentation: Achievements and Challenges.
**总结&思考：**
a. 医疗影像数据相比普通图片数据具有很大的差异，主要表现在多模态(如何融合多种模态数据提升模型效果)、数据分布差异大(如何提升迁移学习的效果)、数据偏移非常严重(如何引导模型更多的关注正样本→加权损失函数)、数据收集很贵(如何通过自监督的方式在小样本数据集上训练效果较好的模型)、器官或病灶多尺度(如何对图像的内容信息与局部细节进行较好的抽取与融合)、图像噪声(如何在具有噪声的情况提升模型精度和鲁棒性)、没有统一医疗影像标准(很难形成大规模训练数据集，有标准、设备、敏感性等因素)
b. 医疗影像主要任务有：识别、检测、分割、增强、配准等，基本与计算机视觉领域的任务相同。在解决医疗影像任务时，很多模型有借鉴了计算机视觉领域的思想，重要的是根据不同医疗影像场景的数据特点，设计合理的数据预处理方式、模型结构、损失函数、训练方式等。
- **阅读医疗影像分割领域相关研究的综述：**
Deep Learning Techniques for Medical Image Segmentation: Achievements and Challenges；
U-Net and Its Variants for Medical Image Segmentation A Review of Theory and Applications.
**总结&思考：**
a. 医疗影像分割同图像分割一样，对图片进行像素级分类。目前主要的模型有FCN、SegNet、DeepLab系列、U-Net系列。其中U-Net是当下研究的热点。这几个模型的共同的特点是都有对图像的全局内容信息(HighLevel Feature)和局部细节信息(LowLevel Feature)进行抽取与融合，骨干网络(Backbone)也都是基于主流分类网络(VGG, ResNet, GoogLenet)，除了第一个外，其他模型架构均为Encoder-Decoder模式，不同的是特征融合的方式。
b. 相比图片语义分割，医疗影像的语义分割在边缘上的效果更差，主要原因包括：医疗影像数据不够，模型没法学习更多到细节信息；由于设备原因，医疗影像数据噪声较多，对于像素级预测任务来说，会较大的降低模型对细节信息的把握。如何通过设计良好的模型结构和损失函数，让模型能自适应的、有针对性的对图像信息进行抽取并且鲁棒性较强，如边缘信息、尺度大小、感兴趣的区域(ROI)、带有噪声等。
c. 尝试将Vision Transformer(ViT)，对比学习(Contrastive Learning)，Few-shot Learning(Meta Learning) 模型应用到医疗影像领域。
- **医学图像处理技术学习：**

[医学图像处理技术_上海交大_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1iW411G7N9?spm_id_from=333.337.search-card.all.click)

# 01. 20220603

- **阅读医疗影像分割关键模型论文**
    
    Fully Convolutional Networks for Semantic Segmentation;
    SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation;
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    
    **总结&思考：**
    
    a. FCN通过转置卷积(Transposed Convolution)对分类骨干网络(VGG16, 后期改进使用ResNet50, MobileNet等)提取的特征进行上采样，并对不同上采样倍率(FCN-32s, FCN-16s, FCN-8s)对feature map进行像素级融合，达到即关注到全局语义信息，又关注到局部细节信息的效果。从实验效果来看，FCN对图像边缘的分割不是很细致，对每个像素做加的操作，这种特征融合的方式在语义上可能存在跨度过大的问题，**而且对哪些层级的网络Feature Map进行融合？相隔多远的Feature Map进行融合？还有其他可选的融合方式？论文没有给出具体解释。如何理解转置卷积？它到底对Feature Map做了何种的上采样？在学习过程中它到底学到了什么？上采用后有什么样的效果？相比插值呢？**
    
    b. SegNet是在图像分割领域首次提出Encoder&Decoder对称架构的模型，提出了Pooling Indices保存Encoder stage的对Feature Map进行非重叠2*2Max Pooling原始数据来源的具体位置，并在Decoder stage对Feature  Map进行上采样时使用对应的Pooling Indices将Feature Map中的元素填写到上采样的Feature Map中2*2方格中的对应位置，并通过普通卷积操作将空缺位置元素填满。相比随机策略，作者认为这样能减少多层上采样后的像素偏移误差。**相比FCN，SegNet在Decoder stage中通过多次(与Encoder次数相同)上采样和卷积操作，逐渐地将Feature Map还原至原图像大小并进行级像素预测。这样的好处是不会有过大的语义跨度融合，从而导致的信息丢失问题。这种逐级上采样的方式FCN能更好的还原局部细节信息，不过需要额外存储每次下采样时的Pooling Indices。**
    
    c. U-net的提出是针对医疗影像分割任务的，通过contracting path(Encoder) 去抽取图像的上下文信息，以及expanding path(Decoder)对局部精细信息的还原，并使用了大量的数据增强技术来提高模型准确度。模型对相同层级的下采样与上采样的Feature Map在上采样时进行Concat并通过一次Convolution操作进行融合，这样的特征融合方式直觉上比较合理，可是论文没有给出具体说明。**论文中模型在最后一层对下采样后的Feature Map进行卷积操作时没有使用Same padding，这导致在每次上采样时与对应的下采样Feature Map进行Concat操作前需要对下采样的Feature Map进行中心裁剪，这是一个不好的设计，会带来额外的开销。论文提到的镜像对称操作提升模型对图像边缘的关注，而简单的padding操作会使得图像边缘被忽略，这是一个不错想法，可以应用到其他图像分割任务上。文章最后提到，通过加权的损失函数，去引导模型关注器官间的分割边缘，不那么关注大面积的背景，这也是个很好想法，在医疗图像分割中，有一半以上的均为背景。**
    
- **FCN模型实现**
    
    Git上的模型比较久远，环境比较难搭建，pytorchvision的model库有FCN的实现，不过看不到模型的细节信息。自己实现了FCN-32s，跑了效果很不理想(模型过于简单，没有Feature Map的融合)。对着模型看懂了论文作者的实现。
    
- **准备考试，各种结课设计(英语、大数据、机器学习)**

# 02. 20220610

- **阅读医疗影像分割关键模型论文 - U-Net Series**
Unet++: A nested u-net architecture for medical image segmentation；
Unet 3+: A full-scale connected unet for medical image segmentation；
3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation.
    
    **总结&思考：**
    a. U-Net++认为原始的U-Net在Feature Map的融合上存在较大**语义跨度(Semantic Gap)**，这种直接对相同层级的Encoder与Decoder进行Concat操作不够平滑，从而导致模型对分割图像的边缘细节把握较差。U-Net++借鉴了DenseNet的思想，在每个网络层级内提出若干由U-Net Encoder得到的基本Feature Map上采样得到的中间状态，并使用Dense Connection去过渡直接连接存在Semantic Gap。由于这种网络结构的设计，使得网络的最顶层的的Feature Ma会同时具有不同采样深度下的全局语义信息和局部细节信息，因此U-Net++又结合了Deep Supervised思想，分别对不同层级的顶层Feature Map进行监督，从而提升分割精度。**模型在每个层级内很好的结合的DenseNet的思想，对每个由U-Net Encoder采用得到Feature Map进行了很好的融合，十分具有针对性的解决了论文开始提出的U-Net存在的Semantic Gap问题。不过论文没有对Semantic Gap做具体的解释，如何理解Semantic Gap？U-Net对相同层级的Feature Map做融合直觉上也没有问题。我的理解是U-Net中相同层级的Encoder和Decoder虽然在Feature Map的大小和上下采样的次数上相同，不过他们在语义上可能不同，导致的原因可能是上采样的方式，如转置卷积或者双线性插值，这些本质上不是卷积的逆操作。所以在上下采样的过程中可能会导致每个层级间的语义错位，DenseNet很好平滑了这个Gap。模型的实现中，最终输出结果采用了两种策略，一种是对不同层级的顶层Feature Map对平均，另一种是只对最深层级的顶层Feature Map输出，作为最终的预测结果，为什么不取两者的折中？低层级的顶层Feature Map给较大的权重，高层级的顶层Feature Map给较小的权重？Dense Connection和中间转态的引入会带来较大计算量，是否真的需要这么多的中间状态呢？能不能有选择性的去掉一些层次的中间状态？**
    
    b. U-Net3++认为U-Net++中的Dense Connection导致模型参数过多。相比U-Net++，U-Net3+没有提出中间状态的Feature Map和DenseNet去平滑较大语义跨度，而是提出一种**全尺寸跳跃连接(Full-Scale skip Connection)**的机制，对Decoder部分的Feature Map融合所有不同层级的Feature Map信息，从而使得每个Decoder的Feature Map都会具有不同层级的，全局语义和局部细节的信息。因此，作者对Decoder stage的每个Feature Map进行了Deep Supervise。**U-Net3+在减少模型参数的同时提升了分割的精度，Full-Scale skip Connection 是一种很新意的特征融合方式，直接绕过了Semantic Gap问题，因为它输出的Feature Map均有所有语义信息，是一种很好的Feature Map融合方式，值得借鉴。模型实现中，对每个Deep Supervise的Feature Map分别进行了预测输出，作者可能认为这些Feature Map具有相同的重要性。在我看来，越低层级的Decoder的Feature Map融合了更多直接来自于Encoder的Feature Map，越高层级的Decoder的Feature Map融合了更多直接来自于Dncoder自身的Feature Map，相对而言，越低层级的Decoder Feature Map具有更多的Low-level Detail 信息，越高层级的 Decoder Feature Map 具有更多的High-Level Semantic信息，我觉得可以根据具体的图像分割任务去对多个Supervised输出做一些Trades-off。值得关注的是，在U-Net3+的损失函数中，采用了MS-SSIM Lost，增加了模型对图像比较模糊(fuzzy boundary)的边界的权重，以及使用了Focal Loss缓解了样本不均衡的问题，这两种损失在医疗影像分割任务中具有极大价值，值得借鉴。此外，论文还提出class-guided模型，先对Feature Map中是否包含要分割的对象进行预测，从而防止Over Segmentation，这也是一个很好的思想。**
    c. 3D-Unet 将U-Net中的2D操作全部换为对应的3D操作(**replacing all 2D operations with their 3D counterparts**)，其他部分没有太多改变
    
- **阅读医疗影像分割关键模型论文 - DeepLab series**
Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution and Fully Connected CRFs；
Rethinking Atrous Convolution for Semantic Image Segmentation；
Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation.
**总结&思考：**
a. DeepLabV2(DeepLabV1(arXiv)的方法更DeepLabV2差别不大)认为DCNN频繁的Pooling和Downsampling操作会降低Feature Map的分辨率，这样使得模型不能够很好的处理细节信息，从而使用**空洞卷积(Atrous Convolution)**代替标准卷积，在不增加模型参数的前提下提升卷积核的感受野。对于分割图像中的多尺度问题，DeepLabV2提出**Atrous Spatial Pyramid Pooling（ASPP）**,即将多个不同膨胀系数的空洞卷积的卷积结果组合在一起。关于局部细节问题，DeepLabV2采用fully connected Conditional Random Field, **CRF**)策略，由于CRF模块的计算量较大，在DeepLabV3中，作者删除了该模块，并参照Hybrid Dilated Convolution(**HDC**)的思想提出了Atrous Block，并设计了Atrous Block中连续多个空洞卷积的膨胀系数，防止连续使用空洞卷积导致的Gradding问题。此外，DeepLabV3还对ASPP模型进行了调整，添加了一个1*1的卷积和Image-Level Pooling(GAP)。DeepLabV3+采用Encoder-Decoder架构，并使用结合了深度可分离卷积(**Depthwise Separable Convolution**)思想的空洞卷积，在不影响模型精度的情况下大大减少了模型计算复杂度。**对于DeepLab系列方法，最为核心的思想是结合空洞卷积并提出ASPP模 Block对图像的多尺度信息进行抽取，其他方法的改进均围绕ASPP模块进行。直觉上，ASPP的思想与Inception Block类似，前者通过不同膨胀系数的空洞卷积来实现不同尺度或特征的提取，后者通过不同大小的卷积核对图像不同特征和尺度进行抽取，相比而言，前者使用了更少的参数获得了更大的感受野(Receptive Field)，不过需要注意的是设置合理的膨胀系数，避免Gradding问题(若使用不合适的膨胀系数，会导致Feature Map中存在没有使用的像素)，DeepLabV3+将通过ASPP Block提取后的Feature Map与Backbone抽取的Feature Map做Concat融合后输入分割头输出，这种方式借鉴了PSPNet对Low-Level Feature和High-Level Feature进行融合的方式。**
- **Few-shot Learning 初步学习：**

[Few-Shot Learning (1/3): 基本概念_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1V44y1r7cx?spm_id_from=333.999.0.0&vd_source=edbe5cb50fe36be81af030e074bcfdbe)

# 03. 20220617

- **Few-shot Learning在医疗影像分割领域相关论文阅读：**
SG-One: Similarity Guidance Network for One-Shot Semantic Segmentation；
PANet: Few-Shot Image Semantic Segmentation with Prototype Alignment；
Self-Supervision with Superpixels: Training Few-shot Medical Image Segmentation without Annotation
**总结&思考：**
a. SG-One通过提出的**Masked Average Pooling**对输入样本Support Set中的(Image, Mask) Pair类型特征(Representive Vector / Prototype )进行提取(以每个Image的Feature Map每个通道为单位)，得到原型向量V=(v1, v2, …, vc)，并在每个像素上计算Query Set中Image与Representive Vector 的CosineSimilarity，得到SimilarityMap（**Foreground&BackGround**）再与Query Set中Image的FeatureMap在每个通道上做乘积，得到融合SimilarityMap的FeatureMap并以此作为特征预测最终的分割结果。**Few-shot在图像分类任务上应用较多，通过计算QuerySet中Image与SupportSet中已知类别的Image语义相似度（通过BackBone抽取的图像特征）大小来判断QueryImage的类别。同样的，在语义分割任务中，Few-shot Learning在每个像素点或者每个设定的区域内计算与Prototype的相似度，进而对该像素的类别进行判断（语义分割任务本质上就是像素级别的分类任务）。问题关键在如何从SupportSet中提取有效的Prototype，Masked Average Pooling是一个很自然的想法，通过样本的Mask将背景信息掩盖，不参与特征的计算，从而很好的提取了前景信息的Prototype。实验表明，进行像素级相似度计算后的SimilarityMap对被分割图像的前景类型信息得到了很好的激活，而背景信息得到了很好抑制。训练好的Few-shot Framework能够在不做FineTuning的情况下对未知类别图片进行分割。SG-One提出的模型是One-way One-shot Segmentation Task。**
b. ****PANet将SG-One的One-way One -shot扩展到多个类别的语义分割任务，即被分割图片中含有多个语义类别。具体的，PANet分别对要预测的每个类型做原型信息提取，即通过指示函数只让该类别的Mask参与该类别的原型信息提取(**P_c**)，背景类别单独计算(**P_bg**)。与此同时，PANet认为，通过SupportSet提取出来的原型信息能够对QuerySet中的图片进行很好的分割，那么同样的，由QuerySet中提取出来的原型信息也能够对SupportSet中的图片进行很好的分割，整体来看，就是交换SupportSet与QuerySet的角色，对模型再做一次训练。**这样一方面能够对其从SupportSet与QuerySet提取的原型信息，使得在每个像素点计算相似度时更加准确，从而能够更好的对每个像素点的类型进行预测；另一方面，这使得训练数据增加了一倍，不过，此时模型的训练时间变为原来的两倍。这是一个很好想法，SupportSet和QuerySet的数据结构时一样的，为什么不能反着过来训练。需要注意的是做为训练Few-shot的样本质量要比较高，或者SupportSet与QuerySet的数据分布要尽可能的一致，这样对原型信息对齐才有意义。在实验部分，作者将PANet泛化到弱标签(Scribbles&Bounding box )的分割场景中，这个方向值得进一步研究，PANet提出的模型是C-Way K-shot Segmentation Task。**
c. Self-Supervision with Superpixels提出基于**Superpixels**的伪标签生成方法，并提出**Adaptive local prototype pooling**。作者认为，Few-shot semantic segmentation在训练阶段需要大量的带标号数据，只是在推理阶段，所给的带标号样本较少，所以作者提出基于Superpixels的伪标签生成模型，并通过图像的Intensity &Geometric transforms生成Few-shot训练所需的episode(**Support & Query Pair**)。作者认为在医疗影像中存在严重的**foreground-background imbalance**问题，这样先前的基于**Masked Average Pooling**不能很好地从全局角度提取对应的原型信息，所以作者提出基于局部窗口的原型信息提取模型。**ALPP模型确实能够提升对细节信息的提取，不过相应的计算量也大大增加了，对每个Window提取原型信息&需要与更多的原型信息做相似度计算。此外Window Size需要很好的选取，过小的话计算量会大大增加，且会存在很多冗余的原型信息；过大的话对图像分割的细节把握会下降(Train:4*4, Test: 2*2)**。**在医疗影像分割领域，获取数据的代价是很大的，如何在小样本情况下完成医疗影像分析任务是很有意义和价值的工作。总的来说，相关的解决方案有以下几点：1.合成带标签的图像；2.提升模型在弱标签下的性能，降低数据标号的难度；3.使用只需少量样本的模型，降低带标签数据的需求；**
- **以上关于Few-shot模型的改进思路：1. 针对特定模态或医学场景下，设计特定的图像合成方式以及伪标签生成方式；2. 在Few-shot模型中，除了Mask Average Pooling外，如何设计更有效的，更高效原型信息提取模型，或者取精确度和计算量之间的Trade-off；3.三篇文章都使用了cosin-similarity，是否可以使用其他相似度计算方法，从而取得更为高效和精准的结果。**
- **阅读医疗影像分割关键模型论文实现：**
由于环境搭建、代码调试很花时间，还有数据收集问题，没有逐一跑相关论文的代码了，只是对照着模型看懂了实验代码，并从实验代码中获取关于模型的细节。

# 04. 20220624

- **论文阅读Swin Transformer在图像分割领域相关论文：**
Transformer：Attention Is All You Need；
An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale;
Swin Transformer Hierarchical Vision Transformer using Shifted Windows；
Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis
**总结&思考**
a. Transformer摆脱了CNN、RNN框架的束缚，是一种只基于注意力机制新框架，**Key、Value & Query**，论文提出的背景是自然语言处理任务，整个模型与传统的语言模型一样，分为Encoder和Decoder。其中Encoder部分使用不带掩码的多头注意力机制（Multi-Head Attention）提取输入文本全局语义信息，Decoder部分使用带掩码的多头注意力机制（Masked Multi-Head Attention），根据当前输入字符在由Encoder生成的Key-Value pair中计算所得到的注意力来预测下一个输出的字符。使用带掩码的注意力，是作者认为，在预测阶段模型只能看到预测字符前的文本，而无法看到整个句子的，通过将注意力机制不可见部分的注意力设置为-100，在通过softmax便能使得这些部分的注意力权重输出接近于0，从而达到Mask的效果。为什么使用多头注意力机制，作者认为这样能像CNN中多个Filter那样，在不同的特征子空间提取到不同的语义信息，最后再Concat起来作为输出。作者使用Scaled Dot-Product Attention来计算Query与Key之间的注意力，所谓Scaled就是在计算注意力后除以sqrt(d_k)，然后再过一个softmax，这样能缓解向量d_k中值过大导致的梯度消失问题，加速了模型训练。不像CNN和RNN架构天然的包含了位置信息，所以使用Transformer需要额外添加位置编码。至于为什么要使用Attention，作者从每层的计算复杂度、序列操作以及点对之间的最长路径做了说明。**Transformer开创性的提出了简单而有效的架构，使得一个能够同时处理自然语言和计算机视觉任务的模型成为可能。**
b. ViT是Transformer在计算机视觉领域开创性工作，它向研究者证明了Transformer能在计算机视觉任务上取得很好的成绩。在图像上使用Transformer最大挑战是一张图片的像素较大，直接输入Transformer Encoder会带来不可承受的计算压力。作者提出将图片打成多个（14*14）patch，再通过一个投射层将每个patch变成512维的向量输入Transformer，这样大大降低了输入数据的长度，很好的解决了这个问题。相关的想法还有通过BackBone提取图像的高维FeatureMap，并将其作为Transformer的输入，同样也能达到同样的效果。为了论证文章提出的观点，作者没有对Transformer做任何改变，按照上述方式将一张图片转成序列（位置编码，CLS标签），直接输出Transformer，并在CLS标签加上分类头监督图片类别序列整个模型。实验表明，在极大的数据集（ImageNet-21K、JFT-300M）上做预训练具有很好泛华效果，能够很好的迁移到其他下游任务。文章在附录部分还讨论了 相对位置编码/绝对位置编码、一维编码/二维编码；使用[CLS]token/对注意力输出做平均，实验表明最终效果没有太大区别。**ViT向研究者证明了Transformer在计算机视觉领域同样能带来很好的效果，这使得视觉领域不再局限于CNN架构，同时也使得基于Transformer的多模态模型成为未来研究方向。**
c. Swin Transformer 提出一种层级的Transformer模型，意在更好的提取图像不同尺度的信息。同样的在图片打成一个个Patch（4*4），并在patch的基础上提出window（7*7），每次只在window内计算Attention，这样不论图片多大，Attention的计算复杂度会是一个固定值。为什么要使用移动窗口，作者认为这样能够使得不同窗口间的信息进行通信，从而提升模型的效果。在做窗口循环位移时，作者精心设计了一套掩码机制，实现了在循环位移后，在不改变计算复杂度的情况下，通过一次操作实现窗口内的注意力计算。因此，在每个Transformer Block内，包含了两个操作，一个是基于窗口的多头注意力计算，另一个是基于移动窗口的多头注意力计算。模型架构上，借鉴了很多CNN的思想，window以及shifted window模拟了CNN的卷积操作；Patch Merging（2*2） 模拟了CNN的Pooling操作。也就是通过一次次的Patch Merging操作，使得Transformer能够在不同层级的窗口内提取图片的语义信息。**总的来看，Swin Transformer借鉴了许多CNN的思想，并在计算机视觉下游任务上取得了很好的效果，作者同样认为，存在一个横跨于计算机视觉和自然语言处理的统一（Unified Architecture）架构，不再分别使用不同模型。**

# 05. 20220701

- **Few-shot Learning with Contrastive Learning在图像分割领域相关论文阅读：**
    
    Learning Non-target Knowledge for Few-shot Semantic Segmentation;
    
    Momentum Contrast for Unsupervised Visual Representation Learning.
    
    **总结&思考**
    
    a. 本文作者认为，传统的FSS只关注对Target的Prototype信息的提取，从而导致对目标的边缘分割不准确。因此，本文对分割图片的背景（BackGround）& 易混淆对象（Distracting Object）的信息进行建模，并通过Background Eliminating Module & Distracting Objects Eliminating模块对Backbone抽取的FeatureMap进行相应的过滤，从而达到对目标边缘更加精确分类的目的。其中，在Distracting Objects Eliminating阶段利用对比学习，将P_q与P_s看做正样本对，P_q与SupportSet和QuerySet中的P_DO看做负样本对，并采用Memory Bank策略进行对比学习，从而得到更好的P_q，提升模型精度。**该模型从BG&DO角度考虑问题，首次将两者的信息融合的FSS,提升了模型的对边分割的精度。这是一个很好，也很自然的想法，对一副图片来说，可以分为目标、背景以及易混淆对象。但是作者并没有对BGMM, GBEM, DOEM是如何做到对相应的部分进行过滤的具体操作进行说明，为什么这么做可以达到相应的效果？消融实验只是证明了相应的模块确实能够提升性能。**
    
    b. MOCO是对比学习的开创性工作，它通过Queue数据结构和Momentum很好的达到了对比学习中，对比的负样本集合要1）足够大（large）2）特征连续性（consistent）两个要求，这也是本篇论文的两大共享。具体的，通过Queue解耦负样本集合（Dictionary）大小和批量（Mini_batch）大小，让最新的mini_batch Encoder结果入队，最旧的mini_batch Encoder结果出队，另一方面，为了保证整个队列的特征连续性，作者对Negative Encoder采用MoMentum的方式更新（m=0.999）。**MOCO很好的解决了对比学习中End to End（Dict的大小与mini_batch大小绑定在一起，使得Dict无法做到很大）& Memory Bank（将Negative的编码结果都存起来，但每次训练只更新mini_batch部分的Encoder结果，使得特征的连续性很差）对应的问题，从而学习到了很好的Encoder，并在下游任务上取得了很好的结果。**
    
    ***无监督学习的目标是学习可迁移的特征，预训练的目标是在做fine-turning时对模型进行初始化**
    
- **Zero-shot  Learning with CLIP在图像分割领域相关论文阅读：**
    
    Decoupling Zero-Shot Semantic Segmentation;
    
    **总结&思考**
    
    a. 本文作者认为，人们对图片进行语义分割的步骤是，先对像素进行分组（Group），得到相应的分割块，然后再对分割块进行分类，最终得到图像的所有实例分割对象。具体的，模型通过Transformer对Query图片做Segment-level embedings，再通过Class-agnostic 对Segment embedings进行Group操作，从而获得每个Segment的Mask；对分类部分，本文利用language-version模型（CLIP），使用预训练好的Image-Encoder和Text-Encoder，将每个Segment以及每个类别的Prompt分别输入Image-Encoder和Text-Encoder，在通过计算Image_feature & Text_feature的距离判断对应的Segment所属的具体类别。这也是模型叫Zero-shot的原因。**模型想法很清晰，从人的角度将分割任务拆分成两个stage，然后用不同的模型来实现两个部分的任务（segment mask & classfication），这样使得图像的实例分割任务比较稳定，不会出现以往模型对细节把握过多而导致的过度分割和判断的问题。**
    
- **大数据考试准备**

# 06 20220708

- **Zero-shot Semantic Segmentation 相关论文阅读：**
    
    CLIP: Learning Transferable Visual Models From Natural Language Supervision 
    ****（Decoupling Zero-Shot Semantic Segmentation论文利用CLIP做分类部分）
    本文作者认为目前的图片分类任务都是基于给定的图像类别进行分类的，如ImageNet、Cifar等。然而这样的事先固定类别的方式限制了模型的泛化能力，即对于一个新类型的图像，模型无法对其进行正确的预测。本文通过大量的与文本相关的图像-文本对(Image, Test)，参照对比学习的损失函数（与文本语义相关图片与该文本的特征向量具有较高的相似度，反之），训练得到TextEncoder &  ImageEncoder（Transformer）。经过大量的图像-文本对训练后的TE&IE能够捕捉到具有相同语义信息的图像文本对，所以在应对新类别图像分类任务时，只需将类别信息Prompt后，输入TE得到所有新类别的特征向量，再将要分类的图像通过IE得到图像的特征向量，最后通过计算图像的特征向量与各个文本的特征向量的相似度即可判断要分类图像的类别。需要注意的是，新类别的图像以及文本信息在训练阶段均为出现。**CLIP利用文本信息来监督视觉任务的训练，这样中多模态的方式能够带来以下几点好处：1.打破了预先定义类的限制，使得模型的泛化能力得到极大的提升，做到了真正的Zero-shot；2.数据收集成本低；3.多模态是人类感知世界的方式，同样也是深度学习研究的重要方向，CLIP打通了CV与NLP的边界；4.实验表明，这种方式训练得到特征具有很好表达以及区分能力，具有极强的泛化能力，同样的类别的物体在不同数据分布下同样能够很好地工作（真实图像、素描、卡通、对抗样本等）5.模型简单且高效，能够快速且低成本地应用到其他相关任务。改进的方向有如何设计更好是损失函数使模型训练更快；如何设计更好的Prompt使文本信息能够更好的监督图像特征的抽取。**
    
- **Few-shot Semantic Segmentation 相关论文阅读：**
Dynamic Convolution: Attention over Convolution Kernels；
Squeeze-and-Excitation Networks；
Dynamic Prototype Convolution Network for Few-Shot Semantic Segmentation.
a. DCN认为在有限的计算资源下，限制CNN模型容量的原因有模型的宽度（卷积的通道数）和深度（卷积的层数）。在不增加模型深度和宽度的前提下提升模型的容量，需要对卷积操作进行改进。Dynamic Convolution相比传统使用单个卷积核的卷积操作，它使用了多个并行的卷积核在一次卷积操作中对feature map的特征进行提取（类似于Inception，不过FeatureMap并没有经过多个Kernel的计算，而是先对Kernel聚合再让FeatureMap通过聚合的Kernel），从而提升模型的容量。与Inception不同的是多个并行的卷积在合并时是通过计算输入Feature Map的Attention（借鉴SENet，论文也从优化角度说明这种Aggregation方式能够得到最优解）来进行加权合并的，这也是模型取名Dynamic的缘由，与Inception直接将多个Kernel的结果Concat不同。计算Attention以及对多个Kernel合并的操作都是轻量级计算任务，因此该模型能够在很少计算复杂度下提升CNN的容量。**由于计算资源的限制，我们无法将CNN做得又“宽”又“深”，因此很多改进的工作都是从CNN的构件（Component）着手，如Convolution相关的DCN、SENet、ACN、DSN等，如Pooling相关的ROI Pooling、PS-ROI Pooling等。这些工作的特点一方面是通过少量的计算复杂度换来更大的模型容量；另一方是减少既有模型的计算复杂度，让既有模型得以做宽做深。总的来说就是-提升模型容量。**
b. SENet认为传统的卷积操作多个Kernel作用在FeatureMap上的结果是独立的（dependencies）,而真实的情况是这些输出的相互依存的（interdependencies），因此作者提出特征重校准（f**eature recalibration**）对这些输出进行选择性的增强和压制（emphasis & suppress），从而提升模型的表征能力（容量）。SENet分为两个阶段：squeeze stage通过GAP将卷积操作输出的FeatureMap压缩成1x1xC的向量；excitation stage通过连个FC对压缩向量进行融合，最终输出1x1xC的激活向量，最后再与最初输入的FeatureMap做Channel-wise的乘积作为最终输出。**SENet通过这种高效计算的方式对输出FeatureMap的各个通道数据进行建模，从而学得更具有区分能力的图像表征，而且这种结构即插即用，能够低成本的应用到现有的CNN网络。从这篇Paper也可看出，一方面，思考问题角度对提出新模型的重要性，当大家都对CNN的Component进行该进时，本文的作者却从大家都没注意到的Channel维度进行建模，从而使模型效果得到提升；另一方面，模型并不是越复杂越好，这篇工作提出模型简单且好用，因此后续有大量的工作对其进行研究和使用。**
c. DPCN主要针对现有语义分割模型对图像细节（holes & Slots）分割效果不理想问题提出改进模型，本文认为，以往FSS对Support 和 Query Feature的交互信息把握得不够（cosine similarity， feature Concat操作等），这也是导致FSS模型不能对图像细节进行很好的分割的主要原因。DPCN分为三个模块，SAM（Support Activation Module）、FFM（Feature Filtering Module）、DCM（Dynamic Convolution Module）。SAM基于对称和非对称的滑动窗口，去生成输入Query图像的初步伪Mask，本质上和FSS计算伪Mask的原理一样。FFM旨在过滤掉背景信息，从而得到更为精细的伪Mask，具体是将初步伪mask与query特征进行Element-wise乘法，再与抽取的support的prototype（expended）进行Element-wise add操作，提升mask对分割对象的关注度，最终的得到更为精细的Query Image的Feature。DCM再对FFM输出的Feature机型Dynamic卷积操作，获得不同维度的信息并送入最后的Mask Prediction模块（连同以上几个模块的输出）。在Prediction阶段，DPCN参照PANet，对Support和Query数据进行对换再次进行训练以获得更好的效果。**DPCN从最终效果来看确实对图像细节信息进行很好的分割，不过该模型过于复杂，堆叠了很多模块和参照了过去很多工作，导致后期人们对其进行改进和Follow很困难。模型的很多思想是值得借鉴的，比如FFM过滤掉图像的背景信息，DCM生成动态Kernel对FeatureMap进行进一步提取与融合等。为了得到更精细的、更具有辨别力特征，一方面可以抑制无关信息；另一方面可以加强主体信息，当然也可以两者同时作用。**

# 07. 20220715

- **语义分割数据熟悉&收集**
[https://www.notion.so/Datasets-7317e9226d434508a4ce58783ce6d20f](https://www.notion.so/Datasets-7317e9226d434508a4ce58783ce6d20f)
- **论文复现：PANet**
1.读懂代码&添加注释；2. 复现论文结果；3.重要参数调试运行

[computervisionalgorithmimplements/00panet at main · thinkdifferentor/computervisionalgorithmimplements](https://github.com/thinkdifferentor/computervisionalgorithmimplements/tree/main/00panet)

# 08. 20220722

- **论文阅读：nnU-Net: A Self-configuring Method for Deep Learning-based Biomedical Image Segmentation**
本文作者认为医疗影像分割任务的网络结构设计（Architecture Design）极大地依赖于特定的数据集和硬件设备，这样会导致在不同数据集上的进行分割任务需要领域专家（**expert-driven**）设置不同的网络结构和参数，或者在拥有大量数据（**data-driven**）的前提下通过AutoML，搜索到当前任务最优的网络结构。前者需要大量的时间调试参数和结构；后者需要大量的数据和计算资源，而且两者得到的模型的泛化性能不好。因此，本文提出一种介于这两种方式之间的方法，提出一种Self-configuring的模型，在应对任何新的医疗影像数据时，无需任何手动介入，根据数据的特点、硬件资源的限制以及交叉验证的结果对分割流程（*including* *preprocessing, network architecture, training and post-processing*）进行调整，最终得到具有竞争力的分割结果。本文将决定分割网络结构的因素归纳（通过分析大量比赛top100的模型架构）为固定的（**Fixed**）、基于规则的（**Rule-based**）以及基于经验的（**Empirical**）参数，并设计一套依赖关系，以原始数据和硬件限制作为输入，通过这套依赖关系以及交叉验证计算并调整得出适用于该数据集的分割网络结构，且期间无需人的干预。
**改进点：
1.通过优化Rule-based参数的依赖规则将模型的精度提高，比如原始图像采样的策略（原文使用中心裁剪，能否使用基于注意力的裁剪，将我们关注的主体部分进行裁剪并作为输入）、使用Deformable Convolution、Squeeze-and-Excitation以及Dynamic Convolution结构提升模型对感兴趣区域特征的提取能力、提出更好的最优配置选择模型或算法，能够在更多配置选择中较快（无需那么多次的交叉验证）地选择出最优模型，同时也减少了模型的计算开销、使用非对称的滑动窗口（原文使用半重叠的固定窗口），能够提升模型对图像细节的把握，从而能够提升对图像边缘的分割效果；
2.通过删除一些依赖规则或者更换一些策略降低最终网络结构的时间复杂度和空间复杂度，比如使用Depthwise Separable Convolution减少单词卷积操作的计算复杂度，数据增强是否能够也根据数据集的特征选择性的使用，而不是对所有数据集都使用大量的数据增强，有些数据增强在特定的数据集上带来的效果不是那么好，这样会带来很多额外的计算和存储开销；
3.模型是否能应用在分布式数据集上并保证医疗数据的安全性，比如同一类型的医疗影像数据在不同的医疗机构存储，如何在保证数据安全的前提下，然该模型也能高效的工作，训练得到具有竞争力的分割模型。**
- **论文复现：nnU-Net**
1.代码阅读&添加注释；代码量较大，目前还没有跑过数据集，计划下周跑下

[computervisionalgorithmimplements/01nnu-net at main · thinkdifferentor/computervisionalgorithmimplements](https://github.com/thinkdifferentor/computervisionalgorithmimplements/tree/main/01nnu-net)

# 09 20220729

- **论文阅读：V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation**
V-Net主要针对医疗影像中3D数据的分割，提出一种end-to-end的3D影像分割方法。相比U-Net3D，V-Net在每层Encoder和Decoder内加入了Residual Connection，且上采样方式使用的是De-convolution，而不是三线性插值。使用残差连接能够加快网络收敛，减少训练时间。V-Net的另一个卖点是使用Dice Coefficient Loss而不是Weighted Cross-entropy Loss (U-Net)，这样天然的解决了FG&BG分布不均衡的问题，且不需要像Weighted那样通过Cross Validation去确定相关的权重，从而获得更好的分割效果。**V-Net本质上还是遵循着U-Net的架构，使用逐级&Skip Connection的方式去获得更为细致的分割效果。V-Net和U-Net3D是同年（2016）的工作，V-Net通过引入Residual Connection和Dice Coefficient Loss获得了更好效果，现在看来，前者的影响力显然大于后者，从论文书写的角度看，前者比后者更具有信服力，故事的完整性更好。由于是同时期的工作，两者没有进行相互比较，计划后面对两者性能的差别进行实验验证。**
- **论文阅读**：**KiU-Net Overcomplete Convolutional Architectures for Biomedical Image and Volumetric Segmentation**
本文作者认为，层级式U-Net结构由于在Encoder阶段不断的下采样，从而导致模型过多的关注High-Level 特征&丢失了Low-Level特征，使得模型对细小结构和边界区域的分割效果较差。此外，对这些细小结构的误分类并不会影像整体的Dice Loss，因为分割主体中大块结构占绝大部分，从而进一步导致了模型对细节分割的不准确性。因此，作者提出具有两个分支的分割网络，一部分是原有的U-Net（**undercomplete convolutional network**），另一部分是Kit-Net（**overcomplete convolutional network**）。U-Net分支负责对图像整体进行分割，Kit-Net负责对图像的细节进行把握，从而提升模型对细节的 分割效果。从结构上看，两者刚好相反，合在一起就是一个圆形的架构，因此模型叫KiU-Net。具体的，Kit-Net在Encoder阶段进行上采样，将图像放大后从而更好的提取到Low-Level特征，在Encoder阶段再通过下采样将图像还原至原图像大小。作者还提出Cross Residual Feature Block（**CRFB**）对每个层级Kit-Net和U-Net输出的FeatureMap进行融合，这种交互方式使得模型同时对图像整体结构和细节信息做了很好的提取，从而缓解了U-Net对图像细节分割不准确的问题。与此同时，作者还提出了Residual Connnection和Dense Connection的KiU-Net版本，通过实验表明，两者能进一步带来分割效果上的提升。**KiU-Net的创新点有两个：1.结合U-Net和Kit-Net，提出具有双分支的网络结构；2.提出Cross Residual Feature Block（CRFB）对同层级的U-Net&Kit-Net输出的FeatureMap进行融合。overcomplete representations是信号处理领域的概念，overcomplete convolutional在深度学习领域很少被采用，从直观上来说，Kit-Net确实能够提升模型对图像细节的关注，实验也表明了结果，这是一个很好的跨学科创新。本质上来看，本文没有提出新的架构，而是对现有的方法进行组合，CRFB可以看做一个很自然的想法，但是很具有新意度，给审稿人和读者不一样的感觉，且都是基于现有模型的组合，模型相对简单，不过效果很好，读起来也很容易接受且记忆深刻。这样的工作我觉得是比较好的工作，而且也可以启发新的想法，不一定要提出新全新的模型，试试组合现有的结构呢？**
- **论文阅读：Focal Loss for Dense Object Detection**
Focal Loss是在目标检测任务里提出的损失概念，该损失函数能够根据时设定的超参数伽马，调整模型对难区分样本的关注程度，伽马等于1是等同于cross entropy loss。伽马越大，产生误差的预测阈值越小（预测概率越小，表示模型越不确定该样本的分类结果），则模型对难区分样本的关注度越大，反之则为模型对所有样本的关注度一致，不论预测的概率如何，都会产生较大的误差值计入模型总的误差。此外，Focal Loss能够自然的处理样本的不均衡问题，这样就不需要在采样阶段对正负样本进行额外的筛选。**Focal Loss本质上对了交叉熵损失做了改进，也就是以前在使用交叉熵损失的地方就能使用Focal Loss，虽然是一点小改动，但是涉及的问题很大且很有效，因此带来的影响力十分巨大。**

# 10 20220805

- **论文阅读：A Simple Framework for Contrastive Learning of Visual Representations**
SimCLR在对比学习领域没有提出任何全新的组件，而是对现有的对比学习组件进行分析以及实验验证，进而将这些既有的组件进行组装，得到一种全新的且有效的对比学习的架构。simCLR通过对原图像进行不同数据增强后，再输入相同的编码器，然后过一个非线性投射层（下游任务中直接丢弃），得到一组正样本对。学习的过程采用对比学习中End-to-End的方式，将所有正负样本全部加载到显存中，保证了训练时Encoder的一致性（MOCO中对这种方式进行了详细的说明）。通过SimCLR的分析，我们可以知道，1）随机裁剪&色调变换（保证不同Crop的色调分布不同）的组合数据增强对无监督学习来说至关重要（无监督学习需要更多的样本多样性）；2）在编码器输出和对比学习损失函数间添加可学习的非线性层能够提升Encoder的特征抽取能力（只是通过实验表明，没有做理论上的分析）；3）更多的负样本和更多的训练次数能够提升Encoder的特征提取能力（能够通过对比学习更多负样本来提升Encoder的对不同图像的，不同特制的区分能力）4）对输入的FeatureMap做Normalized能够提升表征学习能力以及加快训练进程。SimCLRv2主要是将SimCLR在模型上做得更大，在编码器上采用Res-152以及3倍宽的隐含层；提升非线性投射层的层数；并借鉴MOCO的Momentum机制，采用moving average of weights for stabilization，得失模型整体效果得到提升。**SimCLR在对既有对比学习组件做详细的分析的基础上，再对相关组件进行组合得到最终模型。这种对现有现有模型做分析一方面能够提升对整个方向上相关模型的把握，有哪些做得好的，哪些做得不好，以及各自的原因；另一方面，在对这些组件的原理深度分析的基础上，较容易提出更有效的组合以及改进模型。对于非线性投射层，后续工作MOCO没有使用，具体的作用有待进一步解释，直观上看，该层是对Encoder输出的FeatureMap进一步做非线性融合，再去做对比学习，有利于提升对比的多样性。那么为什么要在下有任务中丢弃它呢？如何从理论的角度理解。**
- **论文阅读：BEIT: BERT Pre-Training of Image Transformers**
    
    BEIT与ViT极为相似，同样也是将图片打成一个个patch，交给Transformer Encoder进行编码操作，不同的是BEIT没有采用图像重构的代理任务进行学习，而是通过一种图片编码器（Ref Zero-shot Text-to-Image）将patch编码成“image token”（like word token）。与ViT一样，BEIT随机的掩盖一些patch，并通过模型去预测这些被掩盖patch对应的token编码。**文章通过将patch编码成类似于word的token，从而使得Transformer在视觉领域的应用可以直接向NLP中的BERT一样，采用基于token的掩码模型进行自监督训练。在我看来，将Patch进行token化能够提升单个Patch的语义能力，就像word token一样，天然的带着很强的语义信息，这样学习得到的模型应该具有更强的泛化性能。而文章中并没有对其进行说明，觉得可以进一步探索。**
    
- **论文阅读：Context Autoencoder for Self-Supervised Representation Learning**
    
    CAE认为对于CV的自监督学习任务，将“Encoder”和“Pretext Task”进行“分离”能够提升模型Encoder对图像特征抽取的能力，以前的相关工作没有做“分离”的操作，在做自监督任务时Decoder也会完成一部分工作，尽管Encoder没有学习到一个很好的表达，从而导致最终学习到的Encoder泛化性能会减弱。CAE通过Encoder将可见Patch进行编码作为Key，将不可见Patch进行编码作为Query送入Transformer，再将预测的不可见特征与Encoder直接抽取的不可见Patch的特征进行对其操作，从而达到“分离”的目的。后续的Decoder网络将输出的不可见特征解码为原始的RGB图像或者token（like BEIT），从而达到自监督学习的目的。**直观上看，这样确实能够迫使Encoder尽可能多的学习到图像的特征信息，从而使得模型在下游任务中取得更好地泛化性能。不过文章没有解释为什么这样能达到效果，单纯的从结果上看，CAM并没有比BEIT、MAE等基于掩码的视觉模型好（FT取最优，LE的结果并不理想），而且对其操作所带来的计算量也会比较大。不过“分离”的思想还是挺好的，如何突破现有基于掩码的视觉模型的泛化能力，是一个值得思考的方向，“分离”是一个很好的角度，其他角度呢？**
    

# 11 20220812

- **论文阅读：High-Resolution Swin Transformer for Automatic Medical Image Segmentation**
本文作者认为基于**UNet**架构的Transformer Segmentation模型存在细节信息丢失的问题，因此作者提出基于**HRNet**的Swin Transformer分割模型。相比UNet，更好的保持了High-Resolution信息，缓解了UNet架构在不断下采样过程中图像细节丢失的问题。具体的，通过Patch Merging & Patch Expending操作进行下采样 & 上采样，并提出Multi-Resolution Feature Fusion（**MRFF**）块对每个Stage的SwinTrans进行多尺度融合，从而达到保留High-Res的目的。**HRSNet一方面没有采用UNet架构而是采用HRNet模型，并设计MRFF Block提升模型对图像细节的把握；另一方面，HRSNet没有使用ViT做Backbone而是采用SwinTran做Backbone，并在3D医疗影像分割任务上取得具有竞争力的结果。工作的新意度和有效性都较高，多尺度融合模块的设计有点UNet++的感觉，本质上就是将每个Stage下的所有不同尺度特征进行融合（通过上下采样获得相同size的FeatureMap再做Concat）**
- **论文阅读：ScaleFormer: Revisiting the Transformer-based Backbones from a Scale-wise Perspective for Medical Image Segmentation**
ScaleFormer先对既有的基于Transformer的Segmentation模型进行了高度的总结，具体分为Pure Transformer、Deepest Transformer & Scale-wise hybrid Transformer（this work）。并认为现有的Transformer 分割模型存在两点不足：1）**intra-scale problem** 即现有模型不能很好的处理局部-整体之间的线索（*local-global cues*），这将导致模型对细小物体的分割不准确；2）**intra-scale problem** 即现有模型不能很好的处理具有多尺度目标的图像，比如腹部CT，从而对具有目标大小变化较大，具有细小物体的图像的分割效果较差。Transformer的优点在于能够对图像物体间的长程（long-range）信息做很好的提取（**Global Info**），而CNN的优点在于具有平移不变性以及很好图像特征抽取能力（**Local Info**）。所以本文作者采用Transformer&CNN特征融合（Concat）的方式作为FeatureMap输入相应的Transformer。此外，为了提升Transformer的计算效率，作者采用**Dual-axis MSA。**同样的，本文也提出了一种基于原始Transformer的多尺度特征融合组件（**spatial-aware inter-scale transformer**），融合基于Transformer输出的多尺度特征，再将输出的特征与CNN抽取的特征进行融合输入到Decoder。**文章的写作方式值得借鉴，与MOCO类似，先对现有基于Transformer的Segmentation架构进行总结，在此基础上，提出自己的具有新意且有效的模型。这样的写作方式给审稿人以及读者清晰的认识，使人容易信服，值得学习。方法层面，将CNN与Transformer抽取的特征进行融合直观上能够综合两者的优势，从而达到提升模型准确度的目的。从模型上看，ScaleFormer结合了CNN与Transformer的优势，相比Pure&Deepest Transformer架构，ScaleFormer去了两者的折中，对更多层级的尺度信息进行了融合。**
- **论文阅读：Selective Kernel Networks**    
SKNet提出一种根据图像信息动态选择卷积的模型，SKNet是一种多分支、基于注意力的卷积网络。多分支思想主要借鉴于Inception；注意力思想主要借鉴于SENet；SKNet包含多个不同的卷积操作去提取不同尺度的信息，计算每个分支的“注意力”时，先将不同的分支抽取的FeatureMap Concat起来，再由SENet中操作动态学习每个分支里，每个通道的权重，最后与各个分支所得的FeatureMap做element-wise product 后，在将所有分支的结果做element-wise sum输出最终的SK结果。**对卷积操作进行改进的几个方向：1）在不改变参数个数的前提下提升模型容量（Deformable Convolution、SENet）；2）在不提升模型容量的前提下减少模型的计算复杂度（Deepth-Separable Convolution）；3）采用多个分支（卷积核）对不同尺度的信息进行提取**