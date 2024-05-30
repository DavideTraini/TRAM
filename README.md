# ViT-Visual-Interpretability

This is the official implementation of the paper: Efficient Token Pruning in Vision Transformers Using an Attention-Based Multilayer Network.

## Abstract


Vision Transformers (ViTs) have demonstrated promising performance in a multitude of computer vision tasks, leading to their deployment in a diverse range of scenarios. However, the training of these systems and the inference tasks performed through them require a significant amount of computational resources. To address this challenge, researchers have proposed solutions to limit the computational efforts required to work with ViTs. These solutions are generally based on manipulating the input of the attention layers present in a ViT to reduce the data to be processed. However, to the best of our knowledge, no approach uses only the attention matrix of the patches; moreover, no approach works directly with any ViT without requiring fine-tuning. In this paper, we fill this gap and propose an approach called Token Reduction via an Attention-Based Multilayer Network (TRAM). TRAM represents the attention layers of the ViT via a multilayer network and derives the importance of a token by computing an appropriate centrality measure. Then, it removes the least impactful tokens based on this importance. In the paper, we also report the results of testing TRAM from scratch and on pre-trained ViTs using three datasets, namely CIFAR10, Imagenette and FMNIST. The results demonstrate promising efficiency gains in terms of frames per second (FPS) and gigaflops (GFlops) while maintaining near-vanilla model accuracy. Finally, the paper presents a qualitative analysis that provides a visual representation of the selection process performed by TRAM to reduce the computational load of ViTs.

TRAM's workflow is shown in <a href="#Multilayer creation">Figure 1</a>. Specifically, for each layer, it first creates a multilayer network based on the current and previous attention matrices obtained from the input image. Then it uses the multilayer network previously created to generate tokens score. Only the best $\kappa$ tokens will be used in the next layers. 

![Multilayer creation](Readme_imgs/Workflow.png)


The following image shows our approach applied to a subset of the Imagenet dataset.

<div style="display: flex; justify-content: center; align-items: center; margin: 0;">
  <img src="https://github.com/DavideTraini/TRAM/blob/main/Readme_imgs/ImmagineGitHub.png" style="width: 570px; height: 430px;">
</div>


## Repository structure

This repository can be directly downloaded and executed locally. The required libraries are displayed in Section [Requirements](#requirements)
The **Code** folder contains 3 subfolders: 
- **Benchmark**: contains the code for testing the efficiency of our approach against the other ones. This benchmark it's just about FPS, GFlops, and parameters and is performed both for ViT and SimpleViT architecture.
- **Pretrained**: contains the code for testing TRAM and the other approaches with fine-tuning. It also contains the notebook for visualizing the patches selected by TRAM. 
- **Scratch**: contains the code for testing TRAM and the other approaches when trained from scratch. This benchmark is done both for ViT and SimpleViT architecture


> [!NOTE]
> Some classes are replicated in different folders because:
> - their implementation varies between ViT and SimpleViT architectures;
> - their implementation varies between scratch and fine-tuning.


## Requirements <a name="requirements"></a>

In our notebook we used the following libraries:
```
timm=0.9.2
einops=0.4.1
torch=2.2.2+cu121  
torchvision=0.17.2+cu121  
numpy=1.24.4  
matplotlib=3.8.2
thop=0.1.1
```


## Citation

If you use this model for your research please cite our paper.
