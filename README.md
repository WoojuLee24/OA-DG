# OA-DG: Object-Aware Domain Generalization

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/object-aware-domain-generalization-for-object/robust-object-detection-on-cityscapes-1)](https://paperswithcode.com/sota/robust-object-detection-on-cityscapes-1?p=object-aware-domain-generalization-for-object)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/object-aware-domain-generalization-for-object/robust-object-detection-on-dwd)](https://paperswithcode.com/sota/robust-object-detection-on-dwd?p=object-aware-domain-generalization-for-object)

**_OA-DG_** is an effective method for single-domain object detection generalization (S-DGOD). It consists of two components: _OA-Mix_ for data augmentation and _OA-Loss_ for reducing domain gaps.

![oadg_introduction](./resources/oadg_introduction.gif)

> [Object-Aware Domain Generalization for Object Detection](https://arxiv.org/abs/2312.12133), **Wooju Lee<sup>*</sup>** , **Dasol Hong<sup>*</sup>** , Hyungtae Lim<sup>†</sup>, and Hyun Myung<sup>†</sup>, AAAI 2024 ([arXiv:2312.12133](https://arxiv.org/abs/2312.12133))


## ✨Features

- OA-DG consists of OA-Mix for data augmentation and OA-Loss for reducing the domain gap.

- OA-Mix increases image diversity while preserving important semantic feature with multi-level transformations and object-aware mixing.

    <details onclose>
    <summary>👀 View some example images</summary>

    ![ex_screenshot](./resources/oamix_examples.png)
    
    </details>

- OA-Loss reduces the domain gap by training semantic relations of foreground and background instances from multi-domain.

- Extensive experiments on standard benchmarks (Cityscapes-C and Diverse Weather Dataset) show that OA-DG outperforms SOTA methods on unseen target domains.

- OA-DG can be generally applied to improve robustness regardless of the augmentation set and object detector architectures.


## 🚣 Getting Started

Follow these steps to set up the project on your local machine for training and testing.

### Prerequisites

Ensure you have the following prerequisites installed on your local system.

1. Install mmdetection: There are several installation guides. Follow one of the below:

   > Our code is forked from mmdetection 2.28.x version.
   
   a. Customize Installation (recommended)

      ```bash
      # Install MMCV using MIM.
   
      $ pip install -U openmim
      $ mim install mmcv-full
   
      # Clone this repository
      $ git clone https://github.com/WoojuLee24/OA-DG.git
   
      # Go into the repository
      $ cd OA-DG
   
      # Install mmdetection
      $ pip install -v -e .
      ```
   
   b. Refer to [the mmdetection's installation instructions](https://mmdetection.readthedocs.io/en/v2.28.2/get_started.html#installation).

   c. Use [Dockerfile](https://github.com/open-mmlab/mmdetection/blob/2.x/docker/Dockerfile) from mmdetection to setup the environment.


2. Install other dependencies
   
   ```bash
   # For image processing operations.
   $ pip install Pillow
   # For spectral-residual saliency algorithm in OA-Mix.
   $ pip install opencv-python
   $ pip install opencv-contrib-python
   ```

3. Prepare the following datasets:

   - [Cityscapes](https://www.cityscapes-dataset.com/): A dataset that contains urban street scenes from 50 cities with detailed annotations.
   - [Diverse Weather Dataset](https://drive.google.com/drive/folders/1IIUnUrJrvFgPzU8D6KtV0CXa8k1eBV9B): This dataset includes various weather conditions for robust testing and development of models, essential for applications in autonomous driving.


## 🏃 How To Run

### Training

```bash
python3 -u tools/train.py $(CONFIG).py --work-dir $(WORK_DIR)
```

<details onclose>
<summary>Example: OA-DG trained on Cityscapes dataset</summary>

```bash
python3 -u tools/train.py configs/OA-DG/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes_oadg.py --work-dir /ws/data/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes_oadg
```

</details>

<details onclose>
<summary>Example: OA-DG trained on DWD dataset</summary>

```bash
python3 -u tools/train.py configs/OA-DG/dwd/faster_rcnn_r101_dc5_1x_dwd_oadg.py --work-dir /ws/data/dwd/faster_rcnn_r101_dc5_1x_dwd_oadg
```

</details>


### Evaluation

- Cityscapes-C
   
   ```bash
    python3 -u tools/analysis_tools/test_robustness.py \
      $(CONFIG).py $(CKPT_FILE).pth --out $(OUT_PATH).pkl \
      --corruptions benchmark --eval bbox
   ```
   
    <details onclose>
    <summary>Example: OA-DG evaluated on Cityscapes-C dataset</summary>
    
   ```bash
    python3 -u tools/analysis_tools/test_robustness.py \
      configs/OA-DG/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes_oadg.py \
      /ws/data/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes_oadg/epoch_2.pth \ 
      --out /ws/data/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes_oadg/test_robustness_result_2epoch.pkl \
      --corruptions benchmark --eval bbox
   ```
    
    </details>


- Diverse Weather Dataset (DWD)

   ```bash
    python3 -u tools/test_dwd.py \
      $(CONFIG).py $(CKPT_FILE).pth --out $(OUT_PATH).pkl \
      --eval mAP
   ```

    <details onclose>
    <summary>Example: OA-DG evaluated on DWD dataset</summary>
    
   ```bash
    python3 -u tools/analysis_tools/test_dwd.py \
      configs/OA-DG/dwd/faster_rcnn_r101_dc5_1x_dwd_oadg.py \
      /ws/data/dwd/faster_rcnn_r101_dc5_1x_dwd_oadg/epoch_10.pth \ 
      --out /ws/data/dwd/faster_rcnn_r101_dc5_1x_dwd_oadg/test_robustness_result_10epoch.pkl \
      --eval mAP
   ```
    </details>

### Demo

You can run [demo](./demo/inference_demo.ipynb).


## Results
We evaluated the robustness of our method for 
common corruptions and various weather conditions in urban scenes.
mPC is an evaluation metric of robustness against out-of-distribution (OOD).


- Cityscapes-C: ![cityacpes-c](./resources/table1.png)
- DWD: 
    <p align="center">
        <img src="./resources/table2.png" width="400"/>
    </p>

    <details onclose>
    <summary>☀️ Daytime-Sunny</summary>
        
    | Class     | GTs   | Dets   | Recall | AP        |
    | --------- | ----- | ------ | ------ | --------- |
    | aeroplane | 1738  | 9711   | 0.799  | 0.561     |
    | bicycle   | 1046  | 6165   | 0.716  | 0.491     |
    | bird      | 95339 | 325982 | 0.880  | 0.763     |
    | boat      | 537   | 3151   | 0.702  | 0.462     |
    | bottle    | 12309 | 76318  | 0.764  | 0.557     |
    | bus       | 787   | 3410   | 0.654  | 0.489     |
    | car       | 5029  | 28229  | 0.835  | 0.582     |
    | **mAP**   |       |        |        | **0.558** |

    </details>

    <details onclose>
    <summary>🌃 Night-Sunny</summary>

    | Class     | GTs    | Dets    | Recall | AP        |
    | --------- | ------ | ------- | ------ | --------- |
    | aeroplane | 2012   | 15307   | 0.688  | 0.395     |
    | bicycle   | 1410   | 9151    | 0.616  | 0.371     |
    | bird      | 241616 | 1409587 | 0.846  | 0.639     |
    | boat      | 665    | 13191   | 0.498  | 0.178     |
    | bottle    | 17566  | 185415  | 0.710  | 0.439     |
    | bus       | 841    | 4907    | 0.447  | 0.271     |
    | car       | 4853   | 41633   | 0.714  | 0.412     |
    | **mAP**   |        |         |        | **0.386** |

    </details>

    <details onclose>
    <summary>🌧️ Dusk-Rainy</summary>

    | Class     | GTs   | Dets   | Recall | AP        |
    | --------- | ----- | ------ | ------ | --------- |
    | aeroplane | 820   | 3953   | 0.604  | 0.382     |
    | bicycle   | 322   | 2469   | 0.481  | 0.285     |
    | bird      | 34240 | 180293 | 0.835  | 0.681     |
    | boat      | 110   | 1508   | 0.336  | 0.132     |
    | bottle    | 5144  | 27022  | 0.525  | 0.325     |
    | bus       | 169   | 1186   | 0.331  | 0.214     |
    | car       | 2235  | 13158  | 0.703  | 0.449     |
    | **mAP**   |       |        |        | **0.353** |

    </details>

    <details onclose>
    <summary>🌙 Night-Rainy</summary>

    | Class     | GTs   | Dets   | Recall | AP        |
    | --------- | ----- | ------ | ------ | --------- |
    | aeroplane | 248   | 1158   | 0.468  | 0.289     |
    | bicycle   | 121   | 1088   | 0.223  | 0.123     |
    | bird      | 21655 | 174857 | 0.668  | 0.356     |
    | boat      | 49    | 1635   | 0.143  | 0.010     |
    | bottle    | 1532  | 20963  | 0.378  | 0.139     |
    | bus       | 71    | 560    | 0.169  | 0.120     |
    | car       | 499   | 4383   | 0.463  | 0.220     |
    | **mAP**   |       |        |        | **0.180** |

    </details>
    
    <details onclose>
    <summary>🌫️ Daytime-Foggy</summary>

    | Class     | GTs   | Dets  | Recall | AP        |
    | --------- | ----- | ----- | ------ | --------- |
    | aeroplane | 554   | 1882  | 0.493  | 0.324     |
    | bicycle   | 4920  | 17470 | 0.500  | 0.324     |
    | bird      | 33392 | 81460 | 0.714  | 0.626     |
    | boat      | 911   | 4301  | 0.497  | 0.319     |
    | bottle    | 21530 | 62759 | 0.527  | 0.420     |
    | bus       | 2363  | 6609  | 0.530  | 0.426     |
    | car       | 736   | 6068  | 0.497  | 0.267     |
    | **mAP**   |       |       |        | **0.387** |

    </details>



## TODO list
- [x] Add instructions
- [x] Add codes
- [x] Add checkpoint files
- [x] Add configuration files for DWD
- [x] Pull request to MMDetection (Please click [here](https://github.com/open-mmlab/mmdetection/pull/11916#issue-2476810620) to review the pull request.)

## 📢 License

Please see the [LICENSE.md](LICENSE.md) file.

## 📫 Contact Information
If you have any questions, please do not hesitate to contact us:


- Wooju Lee ✉️ dnwn24@kaist.ac.kr
- Dasol Hong ✉️ ds.hong@kaist.ac.kr


## 📎 Citation

```shell
@inproceedings{lee2024object,
  title={Object-Aware Domain Generalization for Object Detection},
  author={Lee, Wooju and Hong, Dasol and Lim, Hyungtae and Myung, Hyun},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={4},
  pages={2947--2955},
  year={2024}
}
@misc{lee2023objectaware,
      title={Object-Aware Domain Generalization for Object Detection}, 
      author={Wooju Lee and Dasol Hong and Hyungtae Lim and Hyun Myung},
      year={2023},
      eprint={2312.12133},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
