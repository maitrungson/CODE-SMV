import numpy as np
from sklearn import svm, datasets
import matplotlib.pyplot as plt

def getData(): # biểu diễn dữ liệu bằng đồ thị 2D
    iris = datasets.load_iris()
    return iris

def get2DPlot(iris): # lấy hai thuộc tính đầu tiên .
    X = iris.data[:, :2] 
    Y = iris.target
    X_min, X_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    Y_min, Y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    plt.figure(2, figsize=(8, 6))
    plt.clf()

    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired) # biểu diễn dữ liệu bằng đồ thị
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')

    plt.xlim(X_min, X_max)
    plt.ylim(Y_min, Y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

def getSVMPlot(iris):
    X = iris.data[:, :2]  
    y = iris.target

    h = .01  # chỉnh độ mỏng của lưới tọa độ trên đồ thị vì càng nhỏ thì càng sắc nét

    C = 1.0  # Tham số chính quy SVM
    svc = svm.SVC(kernel='linear', C=C).fit(X, y)
    rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
    poly_svc = svm.SVC(kernel='poly', degree=2, C=C).fit(X, y)
    lin_svc = svm.LinearSVC(C=C).fit(X, y)

   
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # nội dung biểu đồ
    titles = ['SVM với linear kernel',
              'LinearSVC (linear kernel)',
              'SVM với RBF kernel',
              'SVM với polynomial (degree 2) kernel']

    for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):

        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
        plt.xlabel('Dài')
        plt.ylabel('Cao')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])

    plt.show()

if __name__ == "__main__":
    iris = getData()
    if iris is not None:
        getSVMPlot(iris)
© 2021 GitHub, Inc.
Terms
