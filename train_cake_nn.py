import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# Paths
base_dir = 'dataset/combined_cakes'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15


# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=30,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.7, 1.3],
    shear_range=0.2,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
val_gen = val_datagen.flow_from_directory(val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
test_gen = test_datagen.flow_from_directory(test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

# Class Weights
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
class_labels = list(train_gen.class_indices.keys())
class_indices = train_gen.class_indices
labels = train_gen.classes
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights_dict = {i: w for i, w in enumerate(class_weights)}


# Model builder with regularization
from tensorflow.keras.regularizers import l2
def build_model(base_model):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)
    predictions = Dense(train_gen.num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


# MobileNetV2 Fine-tuning (unfreeze last 20 layers)
mobilenet_base = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
for layer in mobilenet_base.layers[:-20]:
    layer.trainable = False
for layer in mobilenet_base.layers[-20:]:
    layer.trainable = True
mobilenet_model = build_model(mobilenet_base)
mobilenet_model.compile(optimizer=Adam(learning_rate=5e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# ResNet50 Fine-tuning (unfreeze last 20 layers)
resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
for layer in resnet_base.layers[:-20]:
    layer.trainable = False
for layer in resnet_base.layers[-20:]:
    layer.trainable = True
resnet_model = build_model(resnet_base)
resnet_model.compile(optimizer=Adam(learning_rate=5e-5), loss='categorical_crossentropy', metrics=['accuracy'])


# Early stopping callback
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train MobileNetV2
print('Training MobileNetV2...')
history_mobilenet = mobilenet_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights_dict,
    callbacks=[early_stop]
)

# Train ResNet50
print('Training ResNet50...')
history_resnet = resnet_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights_dict,
    callbacks=[early_stop]
)


# Evaluate with threshold tuning
def predict_with_threshold(model, test_gen, threshold=0.7):
    y_true = []
    y_pred = []
    for batch_x, batch_y in test_gen:
        preds = model.predict(batch_x)
        for i, pred in enumerate(preds):
            max_idx = np.argmax(pred)
            max_conf = pred[max_idx]
            if max_conf < threshold:
                y_pred.append(class_labels.index('not_cake'))
            else:
                y_pred.append(max_idx)
            y_true.append(np.argmax(batch_y[i]))
        if len(y_true) >= test_gen.samples:
            break
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    return accuracy

print('Evaluating MobileNetV2...')
mobilenet_acc = predict_with_threshold(mobilenet_model, test_gen, threshold=0.7)
print(f'MobileNetV2 accuracy (with threshold): {mobilenet_acc*100:.2f}%')
print('Evaluating ResNet50...')
resnet_acc = predict_with_threshold(resnet_model, test_gen, threshold=0.7)
print(f'ResNet50 accuracy (with threshold): {resnet_acc*100:.2f}%')

# Save best model
if mobilenet_acc > resnet_acc:
    mobilenet_model.save('best_cake_model_mobilenet.keras')
    print('MobileNetV2 saved as best model (Keras format).')
else:
    resnet_model.save('best_cake_model_resnet.keras')
    print('ResNet50 saved as best model (Keras format).')
