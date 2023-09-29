
# Import all the libraries needed
import cudf
import cupy
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
from catboost import CatBoostClassifier
from catboost import CatBoostRegressor
from sklearn.linear_model import LogisticRegression
import pickle
import gc
import tensorflow as tf
import tensorflow.keras.backend as K
import os
import json
learning_rate_init = 0.02
epochs = 50

import pandas as pd
import numpy as np
from tqdm import tqdm
cudf.__version__

# Define and return a list of column names that are not going to be used for training the machine learning models
def get_not_used():
    # cid is the label encode of customer_ID
    # row_id indicates the order of rows
    misscols= ['D_88','D_110','B_39','D_73','B_42','D_134','B_29','D_76','D_132','D_42','D_142','D_53']
    skew=['B_31', 'D_87']
    return ['row_id', 'customer_ID', 'target', 'cid', 'S_2','month']+skew+misscols[:-5]

# Preprocessing and transforming a DataFrame (df) before it is used for training machine learning models
def preprocess(df):
    df['row_id'] = cupy.arange(df.shape[0])
    not_used = get_not_used()
    cat_cols = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120',
                'D_126', 'D_63', 'D_64', 'D_66', 'D_68']

    for col in df.columns:
        if col not in not_used+cat_cols:
            df[col] = df[col].round(2)

    df['S_2'] = cudf.to_datetime(df['S_2'])
    df['cid'], _ = df.customer_ID.factorize()
    df= df.sort_values(['customer_ID','S_2'])
    df['month']= df['S_2'].dt.day
    df['month'] =df.to_pandas().groupby('customer_ID')['month'].diff()
    
    important = ['P_2','B_1','B_4','D_39']
    num_cols = [col for col in df.columns if col not in cat_cols+not_used]+['month']
    nth_cols = [col for col in df.columns if col not in cat_cols+not_used][:30]+['customer_ID']
    
    dgs = add_stats_step(df, num_cols, cat_cols)
        
    # cudf merge changes row orders
    # restore the original row order by sorting row_id
    df= df.merge(df[nth_cols].groupby('customer_ID').nth(-2),on='customer_ID',how='left',suffixes=["_last_1","_last_2"])
    df = df.sort_values('row_id')
    df = df.drop(['row_id'],axis=1)
    return df, dgs

# Generating statistical features from both numeric and categorical columns in the input DataFrame ('df'). It processes these columns in batches and returns a list of dataframes containing the computed statistics
def add_stats_step(df, numcols, catcols):
    n = 50
    dgs = []
    for i in range(0,len(numcols),n):
        s = i
        e = min(s+n, len(numcols))
        dg = add_stats_one_shot_num(df, numcols[s:e])
        dgs.append(dg)
    for i in range(0,len(catcols),n):
        s = i
        e = min(s+n, len(catcols))
        dg = add_stats_one_shot_cat(df, catcols[s:e])
        dgs.append(dg)
    return dgs

# computing statistical features from a subset of numeric columns for each unique customer.
def add_stats_one_shot_num(df, cols):
    stats = ['mean','max','min']
    dg = df.groupby('customer_ID').agg({col:stats for col in cols})
    out_cols = []
    for col in cols:
        out_cols.extend([f'{col}_{s}' for s in stats])
    dg.columns = out_cols
    dg = dg.reset_index()
    return dg

# computing statistical features from a subset of categorical columns for each unique customer. It computes various statistics for each categorical column and generates additional derived features
def add_stats_one_shot_cat(df, cols):
    stats = ['count', 'nunique','mean','last','max','min']
    dg = df.groupby('customer_ID').agg({col:stats for col in cols})
    out_cols = []
    for col in cols:
        out_cols.extend([f'{col}_{s}' for s in stats])
    dg.columns = out_cols
    for col in cols:
        df[f'{col}_cat_diff'] =df.to_pandas().groupby('customer_ID')[col].diff().iloc[[-1]]
    for col in cols:
        df[f'{col}_cat_pct_change'] =df.to_pandas().groupby('customer_ID')[col].pct_change().iloc[[-1]]
    for col in cols:
        dg[f'{col}_last_mean'] = dg[f'{col}_last'] - dg[f'{col}_mean']
    for col in cols:
        dg[f'{col}_max_min'] = dg[f'{col}_max'] - dg[f'{col}_min']
    dg = dg.reset_index()
    return dg

# Generator function that iteratively loads and processes chunks of test data from a parquet file. It divides the test data into smaller chunks for processing to avoid loading the entire dataset into memory at once
def load_test_iter(path, chunks=15):
    
    test_rows = 11363762
    chunk_rows = test_rows // chunks
    
    test = cudf.read_parquet(f'{path}/test.parquet',
                             columns=['customer_ID','S_2'],
                             num_rows=test_rows)
    test = get_segment(test)
    start = 0
    while start < test.shape[0]:
        if start+chunk_rows < test.shape[0]:
            end = test['cus_count'].values[start+chunk_rows]
        else:
            end = test['cus_count'].values[-1]
        end = int(end)
        df = cudf.read_parquet(f'{path}/test.parquet',
                               num_rows = end-start, skiprows=start)
        start = end
        yield process_data(df)

# loading and preprocessing the training data
def load_train(path):
    train = cudf.read_parquet(f'{path}/train.parquet')
    train = process_data(train)
    trainl = cudf.read_csv(f'../input/amex-default-prediction/train_labels.csv')
    train = train.merge(trainl, on='customer_ID', how='left')
    return train

# processing and enhancing the input DataFrame
def process_data(df):
    df,dgs = preprocess(df)
    df = df.drop_duplicates('customer_ID',keep='last')
    for dg in dgs:
        df = df.merge(dg, on='customer_ID', how='left')
#     diff_cols = [col for col in df.columns if col.endswith('_diff')]
#     df = df.drop(diff_cols,axis=1)
    return df

# Processing and segmenting the input test DataFrame.
def get_segment(test):
    dg = test.groupby('customer_ID').agg({'S_2':'count'})
    dg.columns = ['cus_count']
    dg = dg.reset_index()
    dg['cid'],_ = dg['customer_ID'].factorize()
    dg = dg.sort_values('cid')
    dg['cus_count'] = dg['cus_count'].cumsum()
    
    test = test.merge(dg, on='customer_ID', how='left')
    test = test.sort_values(['cid','S_2'])
    assert test['cus_count'].values[-1] == test.shape[0]
    return test

# training an XGBoost classifier with certain hyperparameters and evaluating its performance
def xgb_train(x, y, xt, yt):
    print("-----------xgb starts training-----------")
    print("# of features:", x.shape[1])
    assert x.shape[1] == xt.shape[1]
    dtrain = xgb.DMatrix(data=x, label=y)
    dvalid = xgb.DMatrix(data=xt, label=yt)
    params = {
            'objective': 'binary:logistic', 
            'tree_method': 'gpu_hist', 
            'max_depth': 7,
            'subsample':0.88,
            'colsample_bytree': 0.5,
            'gamma':1.5,
            'min_child_weight':8,
            'lambda':70,
            'eta':0.03,
#             'scale_pos_weight': scale_pos_weight,
    }
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    bst = xgb.train(params, dtrain=dtrain,
                num_boost_round=20500,evals=watchlist,
                early_stopping_rounds=500, feval=xgb_amex, maximize=True,
                verbose_eval=100)
    print('best ntree_limit:', bst.best_ntree_limit)
    print('best score:', bst.best_score)
    return bst.predict(dtrain, iteration_range=(0,bst.best_ntree_limit)), bst.predict(dvalid, iteration_range=(0,bst.best_ntree_limit)), bst

#  training a LightGBM (Light Gradient Boosting Machine) classifier with specified hyperparameters and evaluating its performance
def lgb_train(x, y, xt, yt):
    print("----------lgb starts training----------")
    print("# of features:", x.shape[1])
    assert x.shape[1] == xt.shape[1]
    lgb_train = lgb.Dataset(x.to_pandas(), y.to_pandas())
    lgb_eval = lgb.Dataset(xt.to_pandas(), yt.to_pandas(), reference=lgb_train)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting':'gbdt',
        'seed': 42,
        'num_leaves': 100,
        'learning_rate': 0.01,
        'feature_fraction': 0.20,
        'bagging_freq': 10,
        'bagging_fraction': 0.50,
        'n_jobs': -1,
        'lambda_l2': 2,
        'min_data_in_leaf': 40
       
    }
    gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20500,
                valid_sets=[lgb_train, lgb_eval],
                early_stopping_rounds=500,feval=amex_metric_mod_lgbm, 
                verbose_eval=100,)


    print('best iterations:', gbm.best_iteration)
    print('best score:', gbm.best_score)
    return gbm.predict(x.to_pandas(), num_iteration =gbm.best_iteration),gbm.predict(xt.to_pandas(), num_iteration =gbm.best_iteration), gbm

# Training a CatBoostRegressor, a gradient boosting algorithm from the CatBoost library, with specified hyperparameters and evaluating its performance
def cat_train(x, y, xt, yt):
    print("-----------catboost starts training-----------")
    print("# of features:", x.shape[1])
    assert x.shape[1] == xt.shape[1]
    cat_train = cat.Pool(x.to_pandas(), y.to_pandas())
    cat_eval = cat.Pool(xt.to_pandas(), yt.to_pandas())
    
    clf = CatBoostRegressor(iterations=3000, 
                             task_type='GPU',
                             bagging_temperature = 0.2,
                             od_type='Iter',
                             metric_period = 50,
                             od_wait=20)
    clf.fit(cat_train, eval_set=cat_eval, verbose=100,early_stopping_rounds=500)
    return  clf.predict(cat_eval), clf

# Define a learning rate scheduler for use in training a machine learning model. Specifically, it seems to be designed for a scenario where the learning rate should change during training based on the current epoch
def lr_scheduler(epoch):
    if epoch <= epochs*0.8:
        return learning_rate_init
    else:
        return learning_rate_init * 0.1

# Define a convolutional neural network (CNN) model using TensorFlow/Keras. This model is designed for a binary classification task, as it has a single output neuron with a sigmoid activation function
def get_model(features):
    inp = tf.keras.layers.Input((features,))
    x = tf.keras.layers.Reshape((features,1))(inp)
    x = tf.keras.layers.Conv1D(64,5,strides=5, activation='elu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(32,1, activation='elu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(16,1, activation='elu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(4,1, activation='elu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs=inp, outputs=out)

# training a convolutional neural network (CNN) model using TensorFlow/Keras for a binary classification task
def CNN_train(x, y, xt, yt,features):
    print("-----------CNN starts training-----------")
    callbacks = []
    callbacks.append(tf.keras.callbacks.LearningRateScheduler(lr_scheduler))
    optimizer = tf.keras.optimizers.Adam(lr = learning_rate_init, decay = 0.00001)
    model = get_model(features)
    loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    model.fit(x.to_pandas(), y.to_pandas(), validation_data=(xt.to_pandas(), yt.to_pandas()), epochs=50, verbose=2, batch_size=256,callbacks=callbacks)
    return model.predict(xt.to_pandas(),batch_size=256), model

# custom evaluation metric for LightGBM (Light Gradient Boosting Machine) during training
def lgb_amex(y_pred, y_true):
    return 'amex', amex_metric_np(y_pred,y_true.get_label()), True

# custom evaluation metric for XGBoost (Light Gradient Boosting Machine) during training
def xgb_amex(y_pred, y_true):
    return 'amex', amex_metric_np(y_pred,y_true.get_label())

# Created by https://www.kaggle.com/yunchonggan
# https://www.kaggle.com/competitions/amex-default-prediction/discussion/328020
# a combination of Gini index and a weighted percentage of positive target values.
def amex_metric_np(preds: np.ndarray, target: np.ndarray) -> float:
    indices = np.argsort(preds)[::-1]
    preds, target = preds[indices], target[indices]

    weight = 20.0 - target * 19.0
    cum_norm_weight = (weight / weight.sum()).cumsum()
    four_pct_mask = cum_norm_weight <= 0.04
    d = np.sum(target[four_pct_mask]) / np.sum(target)

    weighted_target = target * weight
    lorentz = (weighted_target / weighted_target.sum()).cumsum()
    gini = ((lorentz - cum_norm_weight) * weight).sum()

    n_pos = np.sum(target)
    n_neg = target.shape[0] - n_pos
    gini_max = 10 * n_neg * (n_pos + 20 * n_neg - 19) / (n_pos + 20 * n_neg)

    g = gini / gini_max
    return 0.5 * (g + d)

# we still need the official metric since the faster version above is slightly off
# function calculates a custom evaluation metric for binary classification. 
# This metric appears to be a combination of two sub-metrics: the weighted Gini index and the percentage of positive targets captured in the top 4% of predictions.
# The final metric combines these two sub-metrics to assess the model's performance.
def amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:

    def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()
        
    def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['target'] * df['weight']).sum()
        df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()

    def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        y_true_pred = y_true.rename(columns={'target': 'prediction'})
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)

    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)

    return 0.5 * (g + d)

# custom evaluation metric designed for LightGBM models used in binary classification tasks.
# This metric combines two components: the Gini index and the percentage of positive targets captured in the top predictions, similar to the previous custom evaluation metric.
def amex_metric_mod_lgbm(y_pred: np.ndarray, data: lgb.Dataset):

    y_true = data.get_label()
    labels     = np.transpose(np.array([y_true, y_pred]))
    labels     = labels[labels[:, 1].argsort()[::-1]]
    weights    = np.where(labels[:,0]==0, 20, 1)
    cut_vals   = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four   = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])

    gini = [0,0]
    for i in [1,0]:
        labels         = np.transpose(np.array([y_true, y_pred]))
        labels         = labels[labels[:, i].argsort()[::-1]]
        weight         = np.where(labels[:,0]==0, 20, 1)
        weight_random  = np.cumsum(weight / np.sum(weight))
        total_pos      = np.sum(labels[:, 0] *  weight)
        cum_pos_found  = np.cumsum(labels[:, 0] * weight)
        lorentz        = cum_pos_found / total_pos
        gini[i]        = np.sum((lorentz - weight_random) * weight)

    return 'AMEX', 0.5 * (gini[1]/gini[0]+ top_four), True


path = '../input/amex-data-integer-dtypes-parquet-format'
train = load_train(path)
# train2 = load_train(path,2)
train.shape

not_used = get_not_used()
not_used = [i for i in not_used if i in train.columns]
msgs = {}
folds = 5
score = 0

#set diff cols if u wanna try different features on xgb & lgbm
#diff_cols= [col for col in train.columns if col.endswith('_max') or col.endswith('_min') or col.endswith('_mean') or col.endswith('_std') or col.endswith('_count')]


for i in range(folds):
    print(f"==============Folds {i}===============")
    mask = train['cid']%folds == i
    tr,va = train[~mask], train[mask]

    x, y = tr.drop(not_used, axis=1), tr['target']
    xt, yt = va.drop(not_used, axis=1), va['target']
    features = len(x.columns)
    
    xp, yp, bst = xgb_train(x, y, xt, yt)
    bst.save_model(f'xgb_{i}.json')

    x = tr.drop(not_used+diff_cols, axis=1)
    xt = va.drop(not_used+diff_cols, axis=1)
    
    xp2,yp2,gbm = lgb_train(x, y, xt, yt)
    gbm.save_model(f'lgb_{i}.json')
    
    yp3,cats = cat_train(x, y, xt, yt)
    cats.save_model(f'cat_{i}.json')
    
    yp4,cnn = CNN_train(x, y, xt, yt,features)
    model_json = cnn.to_json()
    # 写入json文件
    with open(f'cnn_train_{i}.json', 'w') as f:
        json.dump(model_json, f)
    preds = yp * 0.35+yp2 * 0.45+yp3 * 0.1+yp4 * 0.1
    amex_score = amex_metric(pd.DataFrame({'target':yt.values.get()}), 
                                    pd.DataFrame({'prediction':preds}))
    msg = f"Fold {i} amex {amex_score:.4f}"
    print(msg)
    score += amex_score
    del tr,va,x,y
    del xt,yt,cnn,cats,gbm
    _ = gc.collect()

score /= folds
print(f"Average amex score: {score:.4f}")
del train
gc.collect()

cids = []
yps = []
# set chunks 
chunks = 15


for df in tqdm(load_test_iter(path,chunks),total=chunks):
    cids.append(df['customer_ID'])
    not_used = [i for i in not_used if i in df.columns]

    preds=0
    for i in range(folds):
        bst = xgb.Booster()
        bst.load_model(f'xgb_{i}.json')
        dx = xgb.DMatrix(df.drop(not_used, axis=1))
        
        gbm = lgb.Booster(model_file=f'lgb_{i}.json')
        dx2 = df.drop(not_used, axis=1).to_pandas()
        
        cats.load_model(f'cat_{i}.json')
        
        with open(r'cnn_train_{i}.json', 'r') as f:
             model_json = json.load(f)
        cnn = tf.keras.models.model_from_json(model_json)
        
        yp = bst.predict(dx, iteration_range=(0,bst.best_ntree_limit))
        yp2 = gbm.predict(dx2, num_iteration =gbm.best_iteration)
        yp3 = cats.predict(dx2)
        yp4 = cnn.predict(dx2)
        #preds+=final_estimator.predict_proba(np.concatenate((np.expand_dims(yp, 1), np.expand_dims(yp2, 1)), 1))[:,1]
        preds+=(yp * 0.35+yp2 * 0.45+yp3 * 0.1+yp4 * 0.1)
    yps.append(preds/folds)
    
df = cudf.DataFrame()
df['customer_ID'] = cudf.concat(cids)
df['prediction'] = np.concatenate(yps)
df.head()
sub = pd.read_csv('../input/stacking-first/submission1.csv')
sub.to_csv('submission.csv', index = False)

# https://www.kaggle.com/code/zb1373/xgb-lgbm-catboost-cnn-stacking
