## Overview of LQ-rPPG
LQ-rPPG: A Label-Quantized Coarse-to-Fine Learning Framework for Remote Physiological Measurement
<p align="center">
  <img src="images/Overview.png" alt="Framework Overview" width="800"/>
</p>

## ⚙️ Setup
To set up the environment for LQ-rPPG, please execute the following script:
```bash
bash setup.sh
```
This will automatically create a dedicated conda environment and install all necessary dependencies. The setup script has been tested and verified on Linux systems. If you are using Windows or macOS, manual installation of dependencies may be required.



## 📁 Datasets
To use LQ-rPPG, it is necessary to prepare appropriate benchmark datasets. Please refer to the following publications for details on each dataset.

- **MMPD (2023)**  
  - Tang, J.; Chen, K.; Wang, Y.; Shi, Y.; Patel, S.; McDuff, D.; and Liu, X. 2023. MMPD: Multi-domain mobile video physiology dataset. In Proceedings of the IEEE Engineering in Medicine and Biology Society, 1–5.

- **V4V (2021)**  
  - Revanur, A.; Li, Z.; Ciftci, U. A.; Yin, L.; and Jeni, L. A. 2021. The First Vision for Vitals (V4V) Challenge for Non-contact Video-based Physiological Estimation. In Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops, 2760–2767.

- **UBFC-rPPG (2019)**  
  - Bobbia, S.; Macwan, R.; Benezeth, Y.; Mansouri, A.; and Dubois, J. 2019. Unsupervised skin tissue segmentation for remote photoplethysmography. Pattern Recognition Letters, 124: 82–90.

- **COHFACE (2017)**  
  - Heusch, G.; Anjos, A.; and Marcel, S. 2017. A reproducible study on remote heart rate measurement. arXiv preprint, arXiv:1709.00962.

- **PURE (2014)**  
  - Stricker, R.; Müller, S.; and Gross, H.-M. 2014. Non-contact video-based pulse rate measurement on a mobile service robot. In Proceedings of the IEEE International Symposium on Robot and Human Interactive Communication, 1056–1062.


## 🖥️ Testing with Pre-trained Models
Please refer to the configuration files located in `./configs`.

### Intra-Dataset Evaluation (Example)
To run the pre-trained model for **intra-dataset** evaluation (i.e., training and testing on the same dataset), you can use the following example:

- **MMPD → MMPD**:
```bash
python main.py --config_file ./configs/MMPD_MMPD_LQC2F.yaml
```

### Cross-Dataset Evaluation (Examples)
To run the pre-trained model for **cross-dataset** evaluation (i.e., training and testing on different datasets), you can refer to the following examples:

- **PURE → UBFC**:
```bash
python main.py --config_file ./configs/PURE_UBFC_LQC2F.yaml
```

- **UBFC → PURE**:
```bash
python main.py --config_file ./configs/UBFC_PURE_LQC2F.yaml
```



## 🧮 Computing Model Complexity
To calculate the model's number of parameters, multiply–accumulate operations (MACs), and throughput:
```
python model_Para_MACs_Thro.py
```



## 🎓 Acknowledgement
This work builds upon the [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) [1].  
We sincerely thank the authors for providing this valuable resource to the community.



## References
[1] Liu, Xin, et al. "rppg-toolbox: Deep remote ppg toolbox." *Advances in Neural Information Processing Systems*, vol. 36, 2024.
