# テンソルフロー
import tensorflow as tf
# 計算やデータ処理のライブラリ
import numpy as np
import pandas as pd
# データ可視化のライブラリ
import matplotlib.pyplot as plt
# データセットの取得&処理のライブラリ
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
# インポートの確認
print(tf.__version__)
print(np.__version__)
print(pd.__version__)

# データの読み込み
boston = load_boston()
# Pandasのデータフレーム形式へ変換
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['target'] = boston.target
# データの最初の5行を表示
df.head()

# 特徴量とターゲットに切り分け
X_data = np.array(boston.data)
y_data = np.array(boston.target)
# １行目のデータの特徴量（X)とターゲット（y）を確認
print(X_data[0:1])
print(y_data[0:1])

# 正規化
def norm(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std
# データを変更
X_data = norm(X_data)
print(X_data[0:1])

# 1を追加する前のサイズ
print(X_data.shape)
# 1を作成
ones = np.ones((506, 1))
# 1を追加
X_data = np.c_[ones, X_data]
X_data.shape

# 訓練データとテストデータへ切り分け
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
y_train = y_train.reshape(404,1)
y_test = y_test.reshape(102,1)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# 学習率とエポック（反復処理回数）
learning_rate = 0.01
training_epochs = 100
# 特徴量の数
n_dim = X_data.shape[1]
# 特徴量（X)とターゲット（y）のプレースホルダー
X = tf.placeholder(tf.float32,[None,n_dim])
Y = tf.placeholder(tf.float32,[None,1])
# 係数（W）と定数項（b）の変数
W = tf.Variable(tf.ones([n_dim,1]))
b = tf.Variable(0.0)

# 線形モデル
y = tf.add(b, tf.matmul(X, W))
# コスト関数
cost = tf.reduce_mean(tf.square(y - Y))
# 最適化
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

