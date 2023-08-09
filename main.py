import warnings
from utlis import *
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

# 1 载入数据
car_train = pd.read_csv("data/used_car_train_20200313.csv", sep=' ')
car_train.columns = ['SaleID', 'name', 'regDate', 'model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'power',
                     'kilometer', 'notRepairedDamage', 'regionCode', 'seller', 'offerType', 'creatDate',
                     'price', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11',
                     'v_12', 'v_13', 'v_14']
car_test = pd.read_csv("data/used_car_testB_20200421.csv", sep=' ')
car_test.columns = ['SaleID', 'name', 'regDate', 'model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'power',
                    'kilometer', 'notRepairedDamage', 'regionCode', 'seller', 'offerType', 'creatDate',
                    'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11',
                    'v_12', 'v_13', 'v_14']

print("原始训练集大小：{}".format(car_train.shape))
print("原始测试集大小：{}".format(car_test.shape))

car_train = outliers_proc(car_train, 'power', scale=3)

print("处理power后的训练集大小：{}".format(car_train.shape))
print("处理power后的测试集大小：{}".format(car_test.shape))

car_train['used_time'] = (pd.to_datetime(car_train['creatDate'], format='%Y%m%d', errors='coerce') -
                          pd.to_datetime(car_train['regDate'], format='%Y%m%d', errors='coerce')).dt.days
car_test['used_time'] = (pd.to_datetime(car_test['creatDate'], format='%Y%m%d', errors='coerce') -
                         pd.to_datetime(car_test['regDate'], format='%Y%m%d', errors='coerce')).dt.days

# 查看空数据数量，先不删除，因为缺失量为1万5多，占总样本量7.5%。可以使用XGBoost之类的决策树，其本身就能处理缺失值
print(car_train['used_time'].isnull().sum())
print(car_test['used_time'].isnull().sum())

# 从邮编中提取城市信息，相当于加入了先验知识，这里不太懂
car_train['city'] = car_train['regionCode'].apply(lambda x: str(x)[0])
car_test['city'] = car_test['regionCode'].apply(lambda x: str(x)[0])
print(car_train['city'].isna().sum())
print(car_test['city'].isna().sum())

# 计算品牌的销售统计量，这里以train的数据计算统计量
car_train_gb = car_train.groupby("brand")
all_info = {}
for kind, kind_data in car_train_gb:
    info = {}
    kind_data = kind_data[kind_data['price'] > 0]
    info['brand_amount'] = len(kind_data)
    info['brand_price_max'] = kind_data.price.max()
    info['brand_price_median'] = kind_data.price.median()
    info['brand_price_min'] = kind_data.price.min()
    info['brand_price_sum'] = kind_data.price.sum()
    info['brand_price_std'] = kind_data.price.std()
    info['brand_price_average'] = round(kind_data.price.sum() / (len(kind_data) + 1), 2)
    all_info[kind] = info
brand_fe = pd.DataFrame(all_info).T.reset_index().rename(columns={"index": "brand"})
car_train = car_train.merge(brand_fe, how='left', on='brand')
car_test = car_test.merge(brand_fe, how='left', on='brand')

print(car_train.shape)
print(car_test.shape)

print(car_train.shape)
print(car_test.shape)

# 将power数据分桶
power_bin = [i * 10 for i in range(61)]
car_train['power_bin'] = pd.cut(car_train['power'], power_bin, labels=False)
print(car_train[['power_bin', 'power']].head())
car_train['power_bin'].fillna(0, inplace=True)

car_test['power_bin'] = pd.cut(car_test['power'], power_bin, labels=False)
print(car_test[['power_bin', 'power']].head())
car_test['power_bin'].fillna(0, inplace=True)

print(car_train['power_bin'].isnull().sum())
print(car_test['power_bin'].isnull().sum())

print(car_train.shape)
print(car_test.shape)

# 保留一份给lr和nn作特征
car_train_lr = car_train
car_test_lr = car_test

# 删除不需要的数据和严重倾斜的数据
car_train = car_train.drop(['creatDate', 'regDate', 'regionCode', 'seller', 'offerType'], axis=1)
car_test = car_test.drop(['creatDate', 'regDate', 'regionCode', 'seller', 'offerType'], axis=1)
print("训练集大小：{}".format(car_train.shape))
print("训练集特征：{}".format(car_train.columns))
print("测试集大小：{}".format(car_test.shape))
print("测试集特征：{}".format(car_test.columns))
print(car_train.info())
print(car_test.info())

# 将数据保存，tree模型使用
# car_train.to_csv('data_for_tree_train.csv', index=0)
# car_test.to_csv('data_for_tree_test.csv', index=0)

# 筛选出训练用的特征，处理数据使得能够输入到模型
feature_cols = [col for col in car_train.columns if col not in ['SaleID', 'name', 'model', 'brand', 'price']]
test_ID = car_test.SaleID
car_train_Y = car_train.price
car_train_X = car_train[feature_cols]
car_test = car_test[feature_cols]

car_train_X = car_train_X.replace('-', -1)
car_test = car_test.replace('-', -1)

car_train_X['notRepairedDamage'] = car_train_X['notRepairedDamage'].astype(np.float32)
car_test['notRepairedDamage'] = car_test['notRepairedDamage'].astype(np.float32)

car_train_X['city'] = car_train_X['city'].astype(np.float32)
car_test['city'] = car_test['city'].astype(np.float32)

car_train_X.fillna(-1, inplace=True)
car_test.fillna(-1, inplace=True)

print(car_train.info())
print(car_test.info())

# xgboost参数
xgb_params = {'booster': 'gbtree',
              'objective': 'reg:squarederror',
              'eval_metric': 'mae',
              'gamma': 0.1,
              'min_child_weight': 1.1,
              'max_depth': 7,
              'lambda': 10,
              'subsample': 0.7,
              'colsample_bytree': 0.7,
              'colsample_bylevel': 0.7,
              'eta': 0.01,
              'tree_method': 'exact',
              'seed': 0,
              'nthread': 12
              }

x_train, x_val, y_train, y_val = train_test_split(car_train_X, car_train_Y, test_size=0.3)

model = model_xgb(xgb_params, x_train, y_train, 3500)
val_set = xgb.DMatrix(x_val)
test_set = xgb.DMatrix(car_test)

xgb_val_pred = model.predict(val_set)
# xgb_val_pred = pd.DataFrame(data=xgb_val_pred)

xgb_test_pred = model.predict(test_set)
# xgb_test_pred = pd.DataFrame(data=xgb_test_pred)

MAE_xgb = mean_absolute_error(y_val, xgb_val_pred)
print('MAE of val with xgb:', MAE_xgb)

# lgb参数
lgb_params = {
    'boosting': 'gbdt',
    'objective': 'regression',
    'num_leaves': 55,
    'subsample': 0.8,
    'bagging_freq': 1,
    'feature_fraction ': 0.7,
    'learning_rate ': 0.01,
    'seed': 0,
    'max_depth': 12,
    'metric': 'mae',
    'verbosity': 10,
    'early_stopping_round': 50
}


model = model_lgb(lgb_params, x_train, y_train, x_val, y_val, 2000)

lgb_val_pred = model.predict(x_val)
MAE_lgb = mean_absolute_error(y_val, lgb_val_pred)

lgb_test_pred = model.predict(car_test)

print("MAE of val with lgb:" + str(MAE_lgb))

val_Weighted = (1 - MAE_lgb / (MAE_xgb + MAE_lgb)) * lgb_val_pred + (1 - MAE_xgb / (MAE_xgb + MAE_lgb)) * xgb_val_pred
val_Weighted[val_Weighted < 0] = 10  # 由于我们发现预测的最小值有负数，而真实情况下，price为负是不存在的，由此我们进行对应的后修正
print('MAE of val with Weighted ensemble:', mean_absolute_error(y_val, val_Weighted))

sub = (1 - MAE_lgb / (MAE_xgb + MAE_lgb)) * lgb_test_pred + (1 - MAE_xgb / (MAE_xgb + MAE_lgb)) * xgb_test_pred

weight_mix = pd.DataFrame()
weight_mix['SaleID'] = test_ID
weight_mix['price'] = sub
weight_mix.to_csv('weight_mix.csv', index=False)
# 最终部分到这里
# ---------------------------------------------------------------------------------------------------------------

# 绘制学习率曲线与验证曲线
# plot_learning_curve(LinearRegression(), 'Liner_model', train_X[:1000], train_y_ln[:1000], ylim=(0.0, 0.5), cv=5,
#                     n_jobs=1)

# ------------------------------------------------------------------------------------------
# 随机森林 时间太长了
# x_train, x_val, y_train, y_val = train_test_split(car_train_X, car_train_Y, test_size=0.3)
#
# model = RandomForestRegressor(max_depth=7,
#                               criterion='absolute_error', verbose=10)
# model = model.fit(x_train, y_train)
# val_pred = model.predict(x_val)
# score = mean_absolute_error(y_val, val_pred)
# print("val_mae：" + str(score))

# print(np.mean(cross_val_score(model, X=car_train_X, y=car_train_Y,
#                               verbose=10, cv=5, scoring=make_scorer(mean_absolute_error))))

# ----------------------------------------------------------------------------------------
# 再构造一份特征给lr和nn使用
# 取log，归一化
# car_train_lr['power'] = np.log(car_train_lr['power'] + 1)
# car_train_lr['power'] = ((car_train_lr['power'] - np.min(car_train_lr['power'])) / (
#         np.max(car_train_lr['power']) - np.min(car_train_lr['power'])))
# car_train_lr['power'].plot.hist()
# plt.show()
#
# car_test_lr['power'] = np.log(car_test_lr['power'] + 1)
# car_test_lr['power'] = ((car_test_lr['power'] - np.min(car_test_lr['power'])) / (
#         np.max(car_test_lr['power']) - np.min(car_test_lr['power'])))
# car_test_lr['power'].plot.hist()
# plt.show()
#
# # 归一化
# car_train_lr['kilometer'] = max_min(car_train_lr['kilometer'])
# car_train_lr['kilometer'].plot.hist()
# plt.show()
#
# car_test_lr['kilometer'] = max_min(car_test_lr['kilometer'])
# car_test_lr['kilometer'].plot.hist()
# plt.show()
#
# car_train_lr['brand_amount'], car_test_lr['brand_amount'] = max_min(car_train_lr['brand_amount']), \
#     max_min(car_test_lr['brand_amount'])
# car_train_lr['brand_price_average'], car_test_lr['brand_price_average'] = max_min(car_train_lr['brand_price_average']) \
#     , max_min(car_test_lr['brand_price_average'])
# car_train_lr['brand_price_max'], car_test_lr['brand_price_max'] = max_min(car_train_lr['brand_price_max']) \
#     , max_min(car_test_lr['brand_price_max'])
# car_train_lr['brand_price_median'], car_test_lr['brand_price_median'] = max_min(car_train_lr['brand_price_median']) \
#     , max_min(car_test_lr['brand_price_median'])
# car_train_lr['brand_price_min'], car_test_lr['brand_price_min'] = max_min(car_train_lr['brand_price_min']), \
#     max_min(car_test_lr['brand_price_min'])
# car_train_lr['brand_price_std'], car_test_lr['brand_price_std'] = max_min(car_train_lr['brand_price_std']) \
#     , max_min(car_test_lr['brand_price_std'])
# car_train_lr['brand_price_sum'], car_test_lr['brand_price_sum'] = max_min(car_train_lr['brand_price_sum']) \
#     , max_min(car_test_lr['brand_price_sum'])
#
# # OneHotEncoder
# car_train_lr = pd.get_dummies(car_train_lr, columns=['model', 'brand', 'bodyType', 'fuelType',
#                                                      'gearbox', 'notRepairedDamage', 'power_bin'])
# car_test_lr = pd.get_dummies(car_test_lr, columns=['model', 'brand', 'bodyType', 'fuelType',
#                                                    'gearbox', 'notRepairedDamage', 'power_bin'])
# print("训练集大小：{}".format(car_train_lr.shape))
# print("测试集大小：{}".format(car_test_lr.shape))
# print("训练集特征：{}".format(car_train_lr.columns))
# print("测试集特征：{}".format(car_test_lr.columns))
#
# print("训练集信息：")
# print(car_train_lr.info())
# print("测试集信息：")
# print(car_test_lr.info())
#
# # 删除不需要的数据和严重倾斜的数据
# car_train_lr = car_train_lr.drop(['creatDate', 'regDate', 'regionCode', 'seller', 'offerType'], axis=1)
# car_test_lr = car_test_lr.drop(['creatDate', 'regDate', 'regionCode', 'seller', 'offerType'], axis=1)
# print("训练集大小：{}".format(car_train_lr.shape))
# print("训练集特征：{}".format(car_train_lr.columns))
# print("测试集大小：{}".format(car_test_lr.shape))
# print("测试集特征：{}".format(car_test_lr.columns))
# print(car_train_lr.info())
# print(car_test_lr.info())
#
# # 这份数据给 LR 用
# # car_train_lr.to_csv('car_total_data_for_lr_train.csv', index=0)
# # car_test_lr.to_csv('car_total_data_for_lr_test.csv', index=0)
#
# # 筛选出训练用的特征，同上
# features = [col for col in car_train_lr.columns if col in car_test_lr.columns]
# feature_cols = [col for col in features if col not in ['SaleID', 'name', 'model', 'brand', 'price']]
# print(feature_cols)
# test_ID_lr = car_test_lr.SaleID
# car_train_Y_lr = car_train_lr.price
# car_train_X_lr = car_train_lr[feature_cols]
# car_test_lr = car_test_lr[feature_cols]
#
# car_train_X_lr = car_train_X_lr.replace('-', -1)
# car_test_lr = car_test_lr.replace('-', -1)
#
# car_train_X_lr['city'] = car_train_X_lr['city'].astype(np.float32)
# car_test_lr['city'] = car_test_lr['city'].astype(np.float32)
#
# car_train_X_lr.fillna(-1, inplace=True)
# car_test_lr.fillna(-1, inplace=True)
#
# print(car_train_X_lr.info())
# print(car_test_lr.info())
# -------------------------------------------------------------------------------------
