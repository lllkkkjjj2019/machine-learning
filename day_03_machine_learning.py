import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,SGDRegressor,Ridge
from sklearn.metrics import mean_squared_error
import joblib

def linear1():
    """正规方程优化方法--波士顿房价"""

    # 1）获取数据集
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    # 2）划分数据集
    x_train, x_test, y_train, y_test = train_test_split(data, target, random_state=22)

    # 3）特征工程：无量纲化 - 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4）预估器流程
    estimator = LinearRegression()
    estimator.fit(x_train,y_train)

    # 5）模型获得
    print("正规方程权重系数：\n",estimator.coef_)
    print("正规方程偏置：\n",estimator.intercept_)

    # 6）模型评估
    y_predict = estimator.predict(x_test)
    # print("预测房价：/n",y_predict)
    error = mean_squared_error(y_test,y_predict)
    print("正规方程均方误差\n",error)


def linear2():
    """梯度下降优化方法--波士顿房价"""

    # 1）获取数据集
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    # 2）划分数据集
    x_train, x_test, y_train, y_test = train_test_split(data, target, random_state=22)

    # 3）特征工程：无量纲化 - 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4）预估器流程
    estimator = SGDRegressor(learning_rate="constant",eta0=0.001,max_iter=10000)
    estimator.fit(x_train, y_train)

    # 5）模型获得
    print("梯度下降权重系数：\n", estimator.coef_)
    print("梯度下降偏置：\n", estimator.intercept_)

    # 6）模型评估
    y_predict = estimator.predict(x_test)
    # print("预测房价：/n",y_predict)
    error = mean_squared_error(y_test,y_predict)
    print("梯度下降均方误差\n",error)

def linear3():
    """岭回归-波士顿房价"""

    # 1）获取数据集
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    # 2）划分数据集
    x_train, x_test, y_train, y_test = train_test_split(data, target, random_state=22)

    # 3）特征工程：无量纲化 - 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4）预估器流程
    # estimator = Ridge()
    # estimator.fit(x_train, y_train)

    # 保存模型
    # joblib.dump(estimator,"my_ridge.pkl")

    # 加载模型
    estimator = joblib.load("my_ridge.pkl")

    # 5）模型获得
    print("岭回归下降权重系数：\n", estimator.coef_)
    print("岭回归下降偏置：\n", estimator.intercept_)

    # 6）模型评估
    y_predict = estimator.predict(x_test)
    # print("预测房价：/n",y_predict)
    error = mean_squared_error(y_test,y_predict)
    print("岭回归均方误差\n",error)


if __name__ == '__main__':
    # 正规方程
    # linear1()
    # 梯度下降
    # linear2()
    # 岭回归
    linear3()