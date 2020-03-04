# RECT (python source code)
Network Embedding with Completely-imbalanced Labels. TKDE2020 [paper](https://zhengwang100.github.io/pdf/TKDE20_wzheng.pdf)

Breifly, RECT contains two parts:
---
- RECT-L is the supervised part in which a semantic loss is used. 
- RECT-N is the unsupervised part in which the network structure is preserved. Note, this part can be replaced by any unsupervised NRL methods.


Usage (abstract):
---
- set the dataset 
- python main_rect.py

```
------ evaluate RECT-N ---------
Training an SVM classifier with the pre-defined split setting...
(0.7335058214747736, 0.670830503861163)
------ evaluate RECT-L ---------
Training an SVM classifier with the pre-defined split setting...
(0.7141871496334627, 0.6402691559469643)
------ evaluate RECT ---------
Training an SVM classifier with the pre-defined split setting...
(0.7441138421733506, 0.6805281849343917)
```


Citing
---
If you find this useful in your research, please cit our paper, thx:
```
@article{wang2020RECT,
  title={Network Embedding with Completely-imbalanced Labels},
  author={Wang, Zheng and Ye, Xiaojun and Wang, Chaokun and Cui, Jian and Yu, Philip S},
  journal={TKDE},
  year={2020},
  publisher={IEEE}
}
```
