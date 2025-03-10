from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import jieba
import pandas as pd

def datasets_demo():
    """sklearning datasets using"""

    # get dataset
    iris = load_iris()
    print(iris.data)
    # print(iris["DESCR"])
    # print(iris.feature_names)
    # print(iris.data.shape)
    print(iris.target)

    # dataset split
    x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.2,random_state=22)
    return None

def dict_demo():
    """dict feature extract"""

    data = [{'city': '北京','temperature':100}, {'city': '上海','temperature':60}, {'city': '深圳','temperature':30}]
    transfer = DictVectorizer(sparse=False)
    # sparse矩阵把非零值位置表示出来
    data_new = transfer.fit_transform(data)
    print(data_new)
    print(transfer.get_feature_names_out())

def count_demo():
    """text feature extract：CountVecotrizer"""

    data = ["life is short,i like like python", "life is too long,i dislike python"]
    # data = ["我 爱 北京 天安门", "天安门 上 太阳 升"]
    transfer = CountVectorizer(stop_words=["is","too"])
    data_new = transfer.fit_transform(data)
    print(transfer.get_feature_names_out())
    print(data_new.toarray())


def cut_word(text):
    """进行中文分词："我爱北京天安门" --> "我 爱 北京 天安门" """
    return " ".join(list(jieba.cut(text)))

def count_chinese_demo2():
    """中文文本特征抽取，自动分词"""

    # 将中文文本进行分词
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]

    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))
    # print(data_new)
    # transfer = CountVectorizer(stop_words=["一种", "所以"])
    transfer = TfidfVectorizer(stop_words=["一种", "所以"])
    data_final = transfer.fit_transform(data_new)
    print("特征名字：\n", transfer.get_feature_names_out())
    print("data_new:\n", data_final.toarray())

def min_max_demo():
    """归一化，鲁棒性较差，只适合传统精确小数据场景, 一般用标准化，适合现代嘈杂大数据处理"""
    # 1.获取数据
    data = pd.read_csv("dating.txt")
    data_new = data.iloc[:,:3]
    # print(data_new)
    # 2.转换
    # 归一化
    # transfer = MinMaxScaler(feature_range=(0,1))
    # 标准化
    transfer = StandardScaler()
    data_final = transfer.fit_transform(data_new)
    print("特征名字：\n", transfer.get_feature_names_out())
    print("data_new:\n", data_final)

def variance_demo():
    """过滤低方差特征"""
    # 1.获取数据
    data = pd.read_csv("factor_returns.csv")
    data = data.iloc[:,1:-2]
    # print(data_new)
    # 2.过滤
    transfer = VarianceThreshold(threshold=10)
    data_final = transfer.fit_transform(data)
    print("特征名字：\n", transfer.get_feature_names_out(),data_final.shape)
    print("data_new:\n", data_final)
    """
    计算两个变量的相关系数
    """
    r = pearsonr(data["pe_ratio"],data["pb_ratio"])
    print(r)

def pca_demo():
    """PCA降维"""
    data = [[2,8,4,5], [6,3,0,8], [5,4,9,1]]

    # 1、实例化一个转换器类
    transfer = PCA(n_components=0.95)

    # 2、调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)
    return None

if __name__ == '__main__':
    datasets_demo()
    # dict_demo()
    # count_demo()
    # count_chinese_demo2()
    # min_max_demo()
    # variance_demo()
    # pca_demo()