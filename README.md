# Cat-Dog Classifier

## Project Overview
This project implements a deep learning-based classifier to distinguish between cats and dogs using a VGG-16 model. The model is trained on a Kaggle dataset and leverages transfer learning by utilizing pretrained weights from ImageNet.

## VGG-16 Model Visualisation
[./VGG_16_model.png]

## Model Architecture
- **Base Model:** VGG-16 pretrained on ImageNet
- **Frozen Layers:** First 4 convolutional layers
- **Trainable Layers:** Last convolutional layer
- **Fully Connected Layers:**
  - Dense layer with 256 nodes and ReLU activation
  - Output layer with a single neuron and sigmoid activation (binary classification: 0 for cat, 1 for dog)

## Dataset
- Source: Kaggle (Dogs vs. Cats Dataset)
- The dataset contains labeled images of cats and dogs used for supervised learning.

## Training Process
- **Loss Function:** Binary Cross-Entropy
- **Optimizer:** RMSprop (learning rate: 1e-5)
- **Batch Size:** 32
- **Evaluation Metrics:** Accuracy, Precision, Recall
- **Epochs:** 10

## Installation & Dependencies
To run this project, ensure you have the following dependencies installed:
```bash
pip install tensorflow keras numpy pandas matplotlib kaggle
```

## Dataset Download & Extraction
```bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!kaggle datasets download -d salader/dogs-vs-cats

import zipfile
zip_ref = zipfile.ZipFile('/content/dogs-vs-cats.zip', 'r')
zip_ref.extractall('/content')
zip_ref.close()
```

## Model Implementation
```python
import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Flatten
from keras.applications.vgg16 import VGG16

conv_base = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(150,150,3)
)

conv_base.trainable = True
set_trainable = False

for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

## Data Preprocessing & Training
```python
# Generators
train_ds = keras.utils.image_dataset_from_directory(
    directory='/content/train',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(150,150)
)

validation_ds = keras.utils.image_dataset_from_directory(
    directory='/content/test',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(150,150)
)

# Normalize
def process(image, label):
    image = tensorflow.cast(image / 255., tensorflow.float32)
    return image, label

train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)

model.compile(
    optimizer=keras.optimizers.RMSprop(lr=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(train_ds, epochs=10, validation_data=validation_ds)
```

## Model Performance Visualization
```python
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], color='red', label='train')
plt.plot(history.history['val_accuracy'], color='blue', label='validation')
plt.legend()
plt.show()

plt.plot(history.history['loss'], color='red', label='train')
plt.plot(history.history['val_loss'], color='blue', label='validation')
plt.legend()
plt.show()
```

## Layers in the model
[./VGG_16_model_expanded.png]

## Results
- Achieved an validation accuracy of [95.2]% on the test dataset.

## Future Improvements
- Fine-tune more layers for better accuracy
- Implement data augmentation for improved generalization
- Experiment with different optimizers and learning rates

