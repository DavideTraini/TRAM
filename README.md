# ViT-Visual-Interpretability

This is the official implementation of the paper: Efficient Token Pruning in Vision Transformers Using Attention-Based Multilayer.

## Abstract


Vision Transformers have demonstrated promising performances in many computer vision tasks, which lead to their implementation in many different scenarios. However, the training of Vision Transformers (ViTs) requires a lot of computational resources, the same applies to the inferences. To address the resource constraint, the researchers have proposed solutions for limiting the computational effort required to work with Vision Transformers. These techniques are based on the manipulation of the input of the attention layers present in a Vision Transformer to reduce the data to process. However, to the best of our knowledge, no approach leverages only the attention of the patches and can be used directly to a Vision Transformer without needing fine-tuning. To fill this gap, we proposi an approach called Token Reduction via Attention-based Multilayer (TRAM). Specifically, TRAM is based on the multilayer network representation of the ViT attention layers, and the importance of a token is derived from a suitable centrality measure, so to filter out the least impactful tokens. We tested TRAM on the training from scratch of two configurations of a ViT and a SimpleViT on three datasets, namely CIFAR10, Imagenette, and FMNIST. Afterward, we evaluated the TRAM performances on pre-trained ViTs. We observed a promising efficiency gain in terms of FPS and GFlops while preserving an accuracy similar to the vanilla case. Finally, we report a qualitative analysis to visually appreciate the selection process made by TRAM to reduce the computational load of ViTs.

TRAM's workflow is shown in <a href="#Multilayer creation">Figure 1</a>. Specifically, for each layer, it first creates a multilayer network based on the current and previous attention matrices obtained from the input image. Then it uses the multilayer network previously created to generate tokens score. Only the best $\kappa$ tokens will be used in the next layers. 

![Multilayer creation](Readme_imgs/Workflow.png)


The following image shows our approach applied to a subset of Imagenet dataset.

<div style="display: flex; justify-content: center; align-items: center; margin: 0;">
  <img src="https://github.com/DavideTraini/TRAM/blob/main/Readme_imgs/ImmagineGitHub.png" style="width: 570px; height: 430px;">
</div>


## Repository sctructure

This repository can be directly downloaded and executed locally. The required libraries are displayed in Section [Requirements](#requirements)
The **Code** folder contains 3 subfolders: 
- **Benchmark**: contains the code for testing the efficiency of our approach against the other ones. This benchmark it's just about FPS, GFlops and parameters and is performed both for ViT and SimpleViT architecture.
- **Pretrained**: contains the code for testing TRAM and the other approaches with fine tuning. It also contains the notebook for visualizing the patches selected by TRAM. 
- **Scartch**: contains the code for testing TRAM and the other approaches when trained from scratch. This benchmark is done both for ViT and SimpleViT architecture


> [!NOTE]
> Some class are replicated in different folders because:
> - their implementation varies between ViT and SimpleViT architectures;
> - their implementation varies between scratch and fine tuning.


## Requirements <a name="requirements"></a>

In our notebook we used the following libraries:
```
PIL=9.2.0  
scipy=1.9.3  
transformers=4.30.2  
torch=2.1.2+cu121  
torchvision=0.16.2+cu121  
sklearn=1.4.1.post1  
numpy=1.24.4  
pandas=2.1.4  
matplotlib=3.8.2
seaborn=0.13.1  
```


## Citation

If you use this model for your research please cite our paper.