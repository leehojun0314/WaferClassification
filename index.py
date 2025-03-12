import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# GPU 설정 (CUDA 활성화 및 메모리 동적 할당)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU 설정 완료: CUDA 사용 가능")
    except RuntimeError as e:
        print(e)

# 데이터셋 로딩 및 전처리 (이미 512x512 타일로 준비됨)
DATASET_PATH = "./wafer_tiles"  # 512x512 타일들이 저장된 폴더
BATCH_SIZE = 32
IMG_SIZE = (512, 512)

# 데이터 증강 및 전처리
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# CNN 모델 구축 (EfficientNet 사용)
base_model = keras.applications.EfficientNetB0(
    input_shape=(512, 512, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False  # 사전 학습된 가중치 고정 (Transfer Learning)

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

# 모델 컴파일 (혼합 정밀도 적용)
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 콜백 설정 (조기 종료 및 학습률 스케줄링)
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

# 모델 학습
EPOCHS = 20
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks
)

# 학습 결과 시각화
def plot_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.legend()
    plt.title('Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')
    
    plt.show()

plot_history(history)

# 모델 저장 및 로드
tf.keras.models.save_model(model, "wafer_cnn_model.h5")
print("모델 저장 완료: wafer_cnn_model.h5")

# 테스트 데이터로 분류 수행 함수
def predict_image(image_path, model):
    img = keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img_array = keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    class_labels = list(train_generator.class_indices.keys())
    
    print(f"예측 결과: {class_labels[predicted_class[0]]}")
    return class_labels[predicted_class[0]]

# 예제 이미지 분류 실행 (사용자가 테스트할 이미지 경로 입력)
test_image_path = "./test_wafer.png"
predict_image(test_image_path, model)
