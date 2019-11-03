
# 基于深度学习的计算机视觉 - 垃圾分类（附源码）

## 1. 实验介绍

### 1.1 实验背景

自今年 7 月 1 日起，上海市将正式实施 《上海市生活垃圾管理条例》。
垃圾分类，看似是微不足道的“小事”，实则关系到13亿多人生活环境的改善，理应大力提倡。
垃圾识别分类数据集中包括玻璃 (glass) 、硬纸板 (cardboard) 、金属 (metal) 、纸 (paper) 、塑料 (plastic) 、一般垃圾 (trash) ，共6个类别。
生活垃圾由于种类繁多，具体分类缺乏统一标准，大多人在实际操作时会“选择困难”，基于深度学习技术建立准确的分类模型，利用技术手段改善人居环境。

### 1.2 实验要求

a）建立深度神经网络模型，并尽可能将其调到最佳状态。
b）绘制深度神经网络模型图、绘制并分析学习曲线。
c）用准确率等指标对模型进行评估。

### 1.3 实验环境

可以使用基于 Python 的 OpenCV 库进行图像相关处理，使用 Numpy 库进行相关数值运算，使用 Keras 等框架建立深度学习模型等。

### 1.4 参考资料

OpenCV：https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html
Numpy：https://www.numpy.org/
Keras: https://keras.io/

## 2.实验内容

### 2.1 介绍数据集

该数据集包含了 2507 个生活垃圾图片。数据集的创建者将垃圾分为了 6 个类别，分别是：

| 序号 |   中文名 |    英文名 |    数据集大小 |
| ---: | -------: | --------: | ------------: |
|    1 |     玻璃 |     glass | 共 497 个图片 |
|    2 |       纸 |     paper | 共 590 个图片 |
|    3 |   硬纸板 | cardboard | 共 400 个图片 |
|    4 |     塑料 |   plastic | 共 479 个图片 |
|    5 |     金属 |     metal | 共 407 个图片 |
|    6 | 一般垃圾 |     trash | 共 134 个图片 |

- 物品都是放在白板上在日光/室内光源下拍摄的，压缩后的尺寸为 512 * 384

  图片预览
  
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019110318422267.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0MjE4NjU0,size_16,color_FFFFFF,t_70)
## 3.实验源码  
实验源码如下图所示
```python
from keras.layers import Input, Dense, Flatten, Dropout, Activation 
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from keras.preprocessing import image
from keras.models import load_model
from keras.models import Model
import matplotlib.pyplot as plt
import glob, os, cv2, random,time
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D,Flatten,MaxPooling2D,Dense 
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16

def processing_data(data_path):
    """
    数据处理
    :param data_path: 数据集路径
    :return: train, test:处理后的训练集数据、测试集数据
    """
    train_data = ImageDataGenerator(
            # 对图片的每个像素值均乘上这个放缩因子，把像素值放缩到0和1之间有利于模型的收敛
            rescale=1. / 225,  
            # 浮点数，剪切强度（逆时针方向的剪切变换角度）
            shear_range=0.1,  
            # 随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]
            zoom_range=0.1,
            # 浮点数，图片宽度的某个比例，数据提升时图片水平偏移的幅度
            width_shift_range=0.1,
            # 浮点数，图片高度的某个比例，数据提升时图片竖直偏移的幅度
            height_shift_range=0.1,
            # 布尔值，进行随机水平翻转
            horizontal_flip=True,
            # 布尔值，进行随机竖直翻转
            vertical_flip=True,
            # 在 0 和 1 之间浮动。用作验证集的训练数据的比例
            validation_split=0.1  
    )

    # 接下来生成测试集，可以参考训练集的写法
    validation_data = ImageDataGenerator(
            rescale=1. / 255,
            validation_split=0.1)

    train_generator = train_data.flow_from_directory(
            # 提供的路径下面需要有子目录
            data_path, 
            # 整数元组 (height, width)，默认：(256, 256)。 所有的图像将被调整到的尺寸。
            target_size=(150, 150),
            # 一批数据的大小
            batch_size=16,
            # "categorical", "binary", "sparse", "input" 或 None 之一。
            # 默认："categorical",返回one-hot 编码标签。
            class_mode='categorical',
            # 数据子集 ("training" 或 "validation")
            subset='training', 
            seed=0)
    validation_generator = validation_data.flow_from_directory(
            data_path,
            target_size=(150, 150),
            batch_size=16,
            class_mode='categorical',
            subset='validation',
            seed=0)

    return train_generator, validation_generator

def model(train_generator, validation_generator, save_model_path):
    """
    模型的建立
    本次实验采用Vgg16模型
    """
    vgg16_model = VGG16(weights='imagenet',include_top=False, input_shape=(150,150,3))
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
    top_model.add(Dense(256,activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(6,activation='softmax'))

    model = Sequential()
    model.add(vgg16_model)
    model.add(top_model)
    # 编译模型, 采用 compile 函数: https://keras.io/models/model/#compile
    model.compile(
             # 是优化器, 主要有Adam、sgd、rmsprop等方式。
            optimizer=SGD(lr=1e-3,momentum=0.9),
            # 损失函数,多分类采用 categorical_crossentropy
            loss='categorical_crossentropy',
            # 是除了损失函数值之外的特定指标, 分类问题一般都是准确率
            metrics=['accuracy'])

    model.fit_generator(
            # 一个生成器或 Sequence 对象的实例
            generator=train_generator,
            # epochs: 整数，数据的迭代总轮数。
            epochs=200,
            # 一个epoch包含的步数,通常应该等于你的数据集的样本数量除以批量大小。
            steps_per_epoch=2259 // 16,
            # 验证集
            validation_data=validation_generator,
             # 在验证集上,一个epoch包含的步数,通常应该等于你的数据集的样本数量除以批量大小。
            validation_steps=248 // 16,
            )
    model.save(save_model_path)

    return model

def evaluate_mode(validation_generator, save_model_path):
     # 加载模型
    model = load_model('results/Ynnex1.h5')
    # 获取验证集的 loss 和 accuracy
    loss, accuracy = model.evaluate_generator(validation_generator)
    print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))

def predict(img):
    """
    加载模型和模型预测
    主要步骤:
        1.加载模型(请加载你认为的最佳模型)
        2.图片处理
        3.用加载的模型预测图片的类别
    :param img: PIL.Image 对象
    :return: string, 模型识别图片的类别, 
            共 'cardboard','glass','metal','paper','plastic','trash' 6 个类别
    """
    # 把图片转换成为numpy数组
    img = img.resize((150, 150))
    img = image.img_to_array(img)
    
    # 加载模型,加载请注意 model_path 是相对路径, 与当前文件同级。
    # 如果你的模型是在 results 文件夹下的 dnn.h5 模型，则 model_path = 'results/dnn.h5'
    model_path = 'results/Ynnex1.h5'
    try:
        # 作业提交时测试用, 请勿删除此部分
        model_path = os.path.realpath(__file__).replace('main.py', model_path)
    except NameError:
        model_path = './' + model_path
    
    # -------------------------- 实现模型预测部分的代码 ---------------------------
    # 加载模型
    model = load_model(model_path)
    
    # expand_dims的作用是把img.shape转换成(1, img.shape[0], img.shape[1], img.shape[2])
    x = np.expand_dims(img, axis=0)

    # 模型预测
    y = model.predict(x)

    # 获取labels
    labels = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}

    # -------------------------------------------------------------------------
    predict = labels[np.argmax(y)]

    # 返回图片的类别
    return predict

def main():
    """
    深度学习模型训练流程,包含数据处理、创建模型、训练模型、模型保存、评价模型等。
    如果对训练出来的模型不满意,你可以通过调整模型的参数等方法重新训练模型,直至训练出你满意的模型。
    如果你对自己训练出来的模型非常满意,则可以提交作业!
    :return:
    """
    data_path = "./datasets/la1ji1fe1nle4ishu4ju4ji22-momodel/dataset-resized"  # 数据集路径
    save_model_path = 'results/Ynnex1.h5'  # 保存模型路径和名称
    # 获取数据
    train_generator, validation_generator = processing_data(data_path)
    # 创建、训练和保存模型
    model(train_generator, validation_generator, save_model_path)
    # 评估模型
    evaluate_mode(validation_generator, save_model_path)


if __name__ == '__main__':
    main()
```
## 4.结果展示

模型的loss值为0.43，在测试集上的准确率大概为90%

![模型的一些参数](https://img-blog.csdnimg.cn/20191103184359474.png)

一些预测的结果展示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191103184603824.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0MjE4NjU0,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191103184630728.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0MjE4NjU0,size_16,color_FFFFFF,t_70)
在momodel.cn上的测试结果为
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191103184735253.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0MjE4NjU0,size_16,color_FFFFFF,t_70)

