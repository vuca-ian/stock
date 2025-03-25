import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator
from typing import Dict, Any
import matplotlib.pyplot as plt

# 在导入tensorflow后添加
tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)
class TimeSeriesModel(BaseEstimator):
    _config: Dict[str, Any]
    _model = None
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self.model_name = config.get("model_name")
        self.total_window_size = self._config.get("input_window") + self._config.get("label_window")
        # tf.debugging.set_log_device_placement(True)  # 打印设备分配日志
        gpus = tf.config.list_physical_devices('GPU')
    
    
    def build_model(self, input_shape: tuple):
        self._model = tf.keras.models.Sequential()
        units = self._config.get("units")
        l2_rate = self._config.get("l2_rate")
        # 添加通道压缩层
        self._model.add(tf.keras.layers.Conv1D(16, 3, activation='relu'))  # 在LSTM前添加
        self._model.add(tf.keras.layers.GlobalAveragePooling1D())  # 替代原Flatten
        self._model.add(tf.keras.layers.LSTM(units=units, 
                                       return_sequences=self._config.get("lstm_layers") > 1, 
                                       input_shape=input_shape, 
                                       kernel_regularizer=tf.keras.regularizers.l2(l2_rate)))
        self._model.add(tf.keras.layers.Dropout(self._config.get("dropout_rate")))
        for _ in range(1, self._config.get("lstm_layers")):
            self._model.add(tf.keras.layers.LSTM(units=units, 
                                           return_sequences=self._config.get("lstm_layers") > 1,
                                           kernel_regularizer=tf.keras.regularizers.l2(l2_rate)))
            self._model.add(tf.keras.layers.Dropout(self._config.get("dropout_rate")))
        for _ in range(1, self._config.get("dense_layers")):
            self._model.add(tf.keras.layers.Dense(units=units, activation='relu'))
        
        self._model.add(tf.keras.layers.Dense(units=1))
        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self._config.get("learning_rate")),
            loss='mean_squared_error',
            clipvalue=0.5, # 梯度裁剪
            metrics=['mae'])
        self._model.summary() 
        return self._model

    def make_dataset(self, X, y, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.batch(batch_size).cache().shuffle(1000)
        return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    def fit(self, X, y, validation_data, **kwargs):

        input_shape = (X.shape[1], X.shape[2])
        self._model = self.build_model(input_shape)

        train_ds = self.make_dataset(X, y, self._config.get("batch_size"))
        self.hisotry = self._model.fit(train_ds, epochs=self._config.get("epochs"),
                        validation_data=validation_data,
                        verbose=1, 
                        callbacks=[
                            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                            ])
        return self.hisotry
    
    def predict(self, X):
        return self._model.predict(X)
    def plot_history(self, history, fold =1):
        plt.figure(figsize=(16, 6))
        plt.plot(history.history['loss'], label=f'Fold {fold + 1} - Training Loss', color='blue', linewidth=2)
        # 绘制验证损失曲线（如果有验证集）
        plt.plot(history.history['val_loss'], label=f'Fold {fold + 1} - Validation Loss', color='orange', linewidth=2)

        plt.title(f'Fold {fold + 1} -Model Training History')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        plt.show()
    """数据增强"""
def moving_average_smoothing(series, window_size=3):
    smoothed_data = np.empty_like(series)  # 创建一个与原始数据形状相同的空数组
    for col in range(series.shape[1]):
        # 对每一列(每个特征)进行平滑处理
        smoothed_data[:,col] = np.convolve(series[:,col], np.ones(window_size)/window_size, mode='same')
    return smoothed_data

def random_noise(data, noise_factor=0.01):
    """随机噪声"""
    noise = noise_factor + np.random.randn(*data.shape)
    return data + noise

def time_series_shift(series, shift_range=5):
    """时间序列平移"""
    shift = np.random.randint(-shift_range, shift_range + 1)
    return np.roll(series, shift, axis=0)

def data_augmentation(X, num_augmentations=5):
    augmented_X = []
    for i in range(len(X)):
        # 移动平均平滑
        augmented_X.append(X[i])
        for _ in range(num_augmentations):
            # 增强方法1 平滑处理
            X_smooth = moving_average_smoothing(X[i])
            # 增加方法2 添加噪声
            X_noise = random_noise(X_smooth)
            # 增加方法3 时间偏移
            X_shift = time_series_shift(X_noise)
            augmented_X.append(X_shift)
    return np.array(augmented_X)