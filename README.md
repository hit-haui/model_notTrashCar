#  End To End model for HAUI.notTrashCar

## Model Architecture

- Base on NVIDIA end-to-end CNN

![Preview](https://raw.githubusercontent.com/maxritter/SDC-End-to-end-driving/master/images/nvidia.png)

- Summary of the model architectures

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
batch_normalization_1 (Batch (None, 66, 200, 3)        264
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 98, 24)        1824
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 47, 36)        21636
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 22, 48)         43248
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 20, 64)         27712
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 18, 64)         36928
_________________________________________________________________
flatten_1 (Flatten)          (None, 1152)              0
_________________________________________________________________
dense_1 (Dense)              (None, 1164)              1342092
_________________________________________________________________
dense_2 (Dense)              (None, 100)               116500
_________________________________________________________________
dense_3 (Dense)              (None, 50)                5050
_________________________________________________________________
dense_4 (Dense)              (None, 10)                510
_________________________________________________________________
dense_5 (Dense)              (None, 2)                 22
=================================================================
Total params: 1,595,786
Trainable params: 1,595,654
Non-trainable params: 132
_________________________________________________________________
```

## Data 

- Hiện tại đang lấy dựa trên simulator của udacity, download [here](https://drive.google.com/file/d/1bczezH23t4FRLmLq8N9u90239d9dyEJZ/view?usp=sharing)
- Chỉ cần tải về giải nén and run `python3 main.py`.
