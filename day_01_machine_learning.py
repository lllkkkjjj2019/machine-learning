from sklearn.datasets import load_iris


def datasets_demo():
    """sklearning datasets using"""
    # get dataset
    iris = load_iris()
    # print(iris["DESCR"])
    print(iris.feature_names)
    return None

if __name__ == '__main__':
    datasets_demo()