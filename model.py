import pandas as pd
import numpy as np
import warnings
import datetime
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from bayes_opt import BayesianOptimization
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm.sklearn import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error,  make_scorer
from utlis import *

warnings.filterwarnings('ignore')


sample_feature = reduce_mem_usage(pd.read_csv('data_for_tree.csv'))

continuous_feature_names = [x for x in sample_feature.columns if x not in ['price', 'brand', 'model', 'brand']]

sample_feature = sample_feature.dropna().replace('-', 0).reset_index(drop=True)
sample_feature['notRepairedDamage'] = sample_feature['notRepairedDamage'].astype(np.float32)
train = sample_feature[continuous_feature_names + ['price']]
# 这里采用的训练集样本特征有点迷
train_X = train[continuous_feature_names]
train_y = train['price']

model = LinearRegression()
# scaler = preprocessing.StandardScaler()
# train_X = scaler.fit_transform(train_X)
# train_X = pd.DataFrame(data=train_X, index=range(len(train_y)), columns=continuous_feature_names)
# print(train_X)
model = model.fit(train_X, train_y)

print('intercept:' + str(model.intercept_))

print(sorted(dict(zip(continuous_feature_names, model.coef_)).items(), key=lambda x: x[1], reverse=True))
subsample_index = np.random.randint(low=0, high=len(train_y), size=50)

plt.scatter(train_X['v_9'][subsample_index], train_y[subsample_index], color='black')
plt.scatter(train_X['v_9'][subsample_index], model.predict(train_X.loc[subsample_index]), color='blue')
plt.xlabel('v_9')
plt.ylabel('price')
plt.legend(['True Price', 'Predicted Price'], loc='upper right')
print('The predicted price is obvious different from true price')
plt.show()

# 通过作图我们发现数据的标签（price）呈现长尾分布，不利于我们的建模预测。
# 原因是很多模型都假设数据误差项符合正态分布，而长尾分布的数据违背了这一假设。
# 参考博客：https://blog.csdn.net/Noob_daniel/article/details/76087829
print('It is clear to see the price shows a typical exponential distribution')
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
sns.distplot(train_y)
plt.subplot(1, 2, 2)
sns.distplot(train_y[train_y < np.quantile(train_y, 0.9)])
plt.show()

train_y_ln = np.log(train_y + 1)


print('The transformed price seems like normal distribution')
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.distplot(train_y_ln)
plt.subplot(1,2,2)
sns.distplot(train_y_ln[train_y_ln < np.quantile(train_y_ln, 0.9)])
plt.show()

model = model.fit(train_X, train_y_ln)

print('intercept:'+ str(model.intercept_))
sorted(dict(zip(continuous_feature_names, model.coef_)).items(), key=lambda x:x[1], reverse=True)

plt.scatter(train_X['v_9'][subsample_index], train_y[subsample_index], color='black')
plt.scatter(train_X['v_9'][subsample_index], np.exp(model.predict(train_X.loc[subsample_index])), color='blue')
plt.xlabel('v_9')
plt.ylabel('price')
plt.legend(['True Price','Predicted Price'],loc='upper right')
print('The predicted price seems normal after np.log transforming')
plt.show()


# 五折交叉验证
def log_transfer(func):
    def wrapper(y, yhat):
        result = func(np.log(y), np.nan_to_num(np.log(yhat)))
        return result
    return wrapper

scores = cross_val_score(model, X=train_X, y=train_y, verbose=1, cv = 5, scoring=make_scorer(log_transfer(mean_absolute_error)))

print('AVG:', np.mean(scores))

scores = cross_val_score(model, X=train_X, y=train_y_ln, verbose=1, cv = 5, scoring=make_scorer(mean_absolute_error))

print('AVG:', np.mean(scores))

scores = pd.DataFrame(scores.reshape(1,-1))
scores.columns = ['cv' + str(x) for x in range(1, 6)]
scores.index = ['MAE']
print(scores)


# # #### 4.4.2 - 3 模拟真实业务情况
# # 但在事实上，由于我们并不具有预知未来的能力，
# 五折交叉验证在某些与时间相关的数据集上反而反映了不真实的情况。
# 通过2018年的二手车价格预测2017年的二手车价格，这显然是不合理的，
# 因此我们还可以采用时间顺序对数据集进行分隔。在本例中，我们选用靠前时间的4/5样本当作训练集，
# 靠后时间的1/5当作验证集，最终结果与五折交叉验证差距不大
sample_feature = sample_feature.reset_index(drop=True)

split_point = len(sample_feature) // 5 * 4

train = sample_feature.loc[:split_point].dropna()
val = sample_feature.loc[split_point:].dropna()

train_X = train[continuous_feature_names]
train_y_ln = np.log(train['price'] + 1)
val_X = val[continuous_feature_names]
val_y_ln = np.log(val['price'] + 1)

model = model.fit(train_X, train_y_ln)
print(mean_absolute_error(val_y_ln, model.predict(val_X)))

# # #### 4.4.2 - 4 绘制学习率曲线与验证曲线
# # get_ipython().run_line_magic('pinfo', ' learning_curve')


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_size=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('Training example')
    plt.ylabel('score')
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                                                            train_sizes=train_size,
                                                            scoring=make_scorer(mean_absolute_error))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()  # 区域
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r',
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()
    return plt


plot_learning_curve(LinearRegression(), 'Liner_model', train_X[:1000], train_y_ln[:1000], ylim=(0.0, 0.5), cv=5,
                    n_jobs=1)

# # #### 4.4.3 多种模型对比
train = sample_feature[continuous_feature_names + ['price']].dropna()

train_X = train[continuous_feature_names]
train_y = train['price']
train_y_ln = np.log(train_y + 1)

# # #### 4.4.3 - 1 线性模型 & 嵌入式特征选择
# # 本章节默认，学习者已经了解关于过拟合、模型复杂度、正则化等概念。否则请寻找相关资料或参考如下连接：
# #
# #   - 用简单易懂的语言描述「过拟合 overfitting」？ https://www.zhihu.com/question/32246256/answer/55320482
# #   - 模型复杂度与模型的泛化能力 http://yangyingming.com/article/434/
# #   - 正则化的直观理解 https://blog.csdn.net/jinping_shi/article/details/52433975
#
# # 在过滤式和包裹式特征选择方法中，特征选择过程与学习器训练过程有明显的分别。
# 而嵌入式特征选择在学习器训练过程中自动地进行特征选择。嵌入式选择最常用的是L1正则化与L2正则化。
# 在对线性回归模型加入两种正则化方法后，他们分别变成了岭回归与Lasso回归。

models = [LinearRegression(),
          Ridge(),
          Lasso()]

result = dict()
for model in models:
    model_name = str(model).split('(')[0]
    scores = cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv=5, scoring=make_scorer(mean_absolute_error))
    result[model_name] = scores
    print(model_name + ' is finished')

# 对三种方法的效果对比
result = pd.DataFrame(result)
result.index = ['cv' + str(x) for x in range(1, 6)]
print(result)

model = LinearRegression().fit(train_X, train_y_ln)
print('intercept:' + str(model.intercept_))
sns.barplot(x=abs(model.coef_), y=continuous_feature_names)
plt.show()

# L2正则化在拟合过程中通常都倾向于让权值尽可能小，最后构造一个所有参数都比较小的模型。
# 因为一般认为参数值小的模型比较简单，能适应不同的数据集，也在一定程度上避免了过拟合现象。
# 可以设想一下对于一个线性回归方程，若参数很大，那么只要数据偏移一点点，就会对结果造成很大的影响；
# 但如果参数足够小，数据偏移得多一点也不会对结果造成什么影响，专业一点的说法是『抗扰动能力强』
model = Ridge().fit(train_X, train_y_ln)
print('intercept:' + str(model.intercept_))
sns.barplot(x=abs(model.coef_), y=continuous_feature_names)
plt.show()

# L1正则化有助于生成一个稀疏权值矩阵，进而可以用于特征选择。
# 如下图，我们发现power与userd_time特征非常重要。
model = Lasso().fit(train_X, train_y_ln)
print('intercept:' + str(model.intercept_))
sns.barplot(x=abs(model.coef_), y=continuous_feature_names)
plt.show()

# # 除此之外，决策树通过信息熵或GINI指数选择分裂节点时，优先选择的分裂特征也更加重要，这同样是一种特征选择的方法。XGBoost与LightGBM模型中的model_importance指标正是基于此计算的
# # #### 4.4.3 - 2 非线性模型
# # 除了线性模型以外，还有许多我们常用的非线性模型如下，在此篇幅有限不再一一讲解原理。我们选择了部分常用模型与线性模型进行效果比对。
models = [LinearRegression(),
          DecisionTreeRegressor(),
          RandomForestRegressor(),
          GradientBoostingRegressor(),
          MLPRegressor(solver='lbfgs', max_iter=100),
          XGBRegressor(n_estimators=100, objective='reg:squarederror'),
          LGBMRegressor(n_estimators=100)]


result = dict()
for model in models:
    model_name = str(model).split('(')[0]
    scores = cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv=5, scoring=make_scorer(mean_absolute_error))
    result[model_name] = scores
    print(model_name + ' is finished')


result = pd.DataFrame(result)
result.index = ['cv' + str(x) for x in range(1, 6)]
print(result)

# # 可以看到随机森林模型在每一个fold中均取得了更好的效果
#
# # #### 4.4.4  模型调参
#
# # 在此我们介绍了三种常用的调参方法如下：
# #
# #   - 贪心算法 https://www.jianshu.com/p/ab89df9759c8
# #   - 网格调参 https://blog.csdn.net/weixin_43172660/article/details/83032029
# #   - 贝叶斯调参 https://blog.csdn.net/linxid/article/details/81189154
#
# ## LGB的参数集合：
objective = ['regression', 'regression_l1', 'mape', 'huber', 'fair']

num_leaves = [3, 5, 10, 15, 20, 40, 55]
max_depth = [3, 5, 10, 15, 20, 40, 55]
bagging_fraction = []
feature_fraction = []
drop_rate = []

# # #### 4.4.4 - 1 贪心调参
best_obj = dict()
for obj in objective:
    model = LGBMRegressor(objective=obj)
    score = np.mean(
        cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv=5, scoring=make_scorer(mean_absolute_error)))
    best_obj[obj] = score

best_leaves = dict()
for leaves in num_leaves:
    model = LGBMRegressor(objective=min(best_obj.items(), key=lambda x: x[1])[0], num_leaves=leaves)
    score = np.mean(
        cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv=5, scoring=make_scorer(mean_absolute_error)))
    best_leaves[leaves] = score

best_depth = dict()
for depth in max_depth:
    model = LGBMRegressor(objective=min(best_obj.items(), key=lambda x: x[1])[0],
                          num_leaves=min(best_leaves.items(), key=lambda x: x[1])[0],
                          max_depth=depth)
    score = np.mean(
        cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv=5, scoring=make_scorer(mean_absolute_error)))
    best_depth[depth] = score

sns.lineplot(x=['0_initial', '1_turning_obj', '2_turning_leaves', '3_turning_depth'],
             y=[0.143, min(best_obj.values()), min(best_leaves.values()), min(best_depth.values())])
plt.show()

# # #### 4.4.4 - 2 Grid Search 调参
parameters = {'objective': objective, 'num_leaves': num_leaves, 'max_depth': max_depth}
model = LGBMRegressor()
clf = GridSearchCV(model, parameters, cv=5)
clf = clf.fit(train_X, train_y)


print(clf.best_params_)


model = LGBMRegressor(objective='regression',
                      num_leaves=55,
                      max_depth=15)


print(np.mean(cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv=5, scoring=make_scorer(mean_absolute_error))))

# # #### 4.4.4 - 3 贝叶斯调参
def rf_cv(num_leaves, max_depth, subsample, min_child_samples):
    val = cross_val_score(
        LGBMRegressor(objective='regression_l1',
                      num_leaves=int(num_leaves),
                      max_depth=int(max_depth),
                      subsample=subsample,
                      min_child_samples=int(min_child_samples)
                      ),
        X=train_X, y=train_y_ln, verbose=0, cv=5, scoring=make_scorer(mean_absolute_error)
    ).mean()
    return 1 - val


rf_bo = BayesianOptimization(
    rf_cv,
    {
        'num_leaves': (2, 100),
        'max_depth': (2, 100),
        'subsample': (0.1, 1),
        'min_child_samples': (2, 100)
    }
)

print(rf_bo.maximize())

print(1 - rf_bo.max['target'])

# # 总结
# # 在本章中，我们完成了建模与调参的工作，并对我们的模型进行了验证。此外，我们还采用了一些基本方法来提高预测的精度，提升如下图所示。
plt.figure(figsize=(13, 5))
sns.lineplot(x=['0_origin', '1_log_transfer', '2_L1_&_L2', '3_change_model', '4_parameter_turning'],
             y=[1.36, 0.19, 0.19, 0.14, 0.13])
plt.show()
























