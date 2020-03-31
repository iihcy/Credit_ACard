# coding:utf-8
# @Author:iihcy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import os
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.metrics import confusion_matrix, roc_curve, auc

# 评价指标
def model_performance(y_train, pre_train, pro_train, y_test, pre_test, pro_test, plot=True):
    # 训练集
    print('Train:')
    fpr0, tpr0, th0 = roc_curve(y_train, pro_train)
    roc_auc0 = auc(fpr0, tpr0)
    print('The model accuracy is {}'.format(accuracy_score(y_train, pre_train)))
    print('The model f1 is {}'.format(f1_score(y_train, pre_train)))
    print('The model p is {}'.format(precision_score(y_train, pre_train)))
    print('The model recall is {}'.format(recall_score(y_train, pre_train)))
    print('The confusion matrix is:\n', confusion_matrix(y_train, pre_train))
    if plot:
        plot_roc_auc(fpr0, tpr0, roc_auc0)

    # 测试集
    print('Test:')
    fpr, tpr, th = roc_curve(y_test, pro_test)
    roc_auc = auc(fpr, tpr)
    print('The model accuracy is {}'.format(accuracy_score(y_test, pre_test)))
    print('The model f1 is {}'.format(f1_score(y_test, pre_test)))
    print('The model p is {}'.format(precision_score(y_test, pre_test)))
    print('The model recall is {}'.format(recall_score(y_test, pre_test)))
    print('The model KS is {}'.format(max(tpr - fpr)))
    print('The confusion matrix is:\n', confusion_matrix(y_test, pre_test))
    if plot:
        plot_roc_auc(fpr, tpr, roc_auc)
    return fpr, tpr, roc_auc

def plot_roc_auc(fpr, tpr, roc_auc):
    plt.plot(fpr, tpr, label='AUC = {}'.format(roc_auc))
    plt.title('ROC')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend()
    plt.plot([0 ,1] ,[0 ,1], c='r')
    plt.show()

def get_model_score(pro):
    score = 520 - 48 * np.log(pro /( 1 -pro))
    score = pd.Series(score).to_frame().rename(columns={0 :'score'})
    return score

def save_file(df, filename):
    base_path = 'result'
    #判断文件是否存在，如果不存在，则创建
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    file = pd.DataFrame(df)
    file_path = os.path.join(base_path, filename)
    file.to_csv(file_path)

def obj_feature(df_final):
    '''
      数据类型查看与转换
    '''
    obj_feature = []
    for obj_f in df_final.columns:
        if df_final[obj_f].dtypes == 'object':
            obj_feature.append(obj_f)
    print(obj_feature)
    return obj_feature

def emp_length_new(x):
    '''
    特征处理：emp_length--工作年限
    '''
    if x=='< 1 year':
        return 0
    elif x=='1 year':
        return 1
    elif x=='10+ years':
        return 11
    elif str(x)=='nan':
        return np.nan
    else:
        return x.replace(' years', '')


def data_distributed(df_finals, char_feature):
    '''
    数据探索
    类别型(离散)特征: 每个特征值分布概率值
    '''
    rate_feature_all = []
    for col in char_feature:
        result = df_finals[col].value_counts()
        #print('char_feature_name:', col)
        rate_feature = []
        for v_i in range(len(result)):
            rate = result[v_i]/result.sum()
            rate_feature.append(rate)
        rate_feature = pd.Series(rate_feature).to_frame().reset_index().rename(columns={'index':'id', 0: 'rate'})
        result = pd.Series(result).to_frame().reset_index().reset_index().rename(columns={'level_0':'id', 'index':'f_name',
                                                                                          'home_ownership':'value_count'})
        rate_feature = pd.merge(result, rate_feature, on=['id']).drop('id', axis=1)
        rate_feature_all.append(rate_feature)
    return rate_feature_all

# 离散：计算每个特征的iv值
def cal_charf_iv(char_df, label):
#     print(char_df.shape, '\n====计算离散特征的iv值====')
    all_feature_iv = []
    char_feature_name = []
    for char_f in list(char_df):
        if char_f not in label:
            char_feature_name.append(char_f)
            #print('feature:', char_f)
            df = pd.concat([char_df[char_f], char_df[label]], axis=1)
            #print(df.shape)
            # 好样本统计
            good_df = df[df[label]==0]
            good_count = good_df[char_f].value_counts().to_frame().reset_index().rename(columns={'index':'char_value',char_f:'good_count'})
            # 坏样本统计
            bad_df = df[df[label]==1]
            bad_count = bad_df[char_f].value_counts().to_frame().reset_index().rename(columns={'index':'char_value',char_f:'bad_count'})
            # 整合
            result_df = pd.merge(good_count, bad_count, on=['char_value'])
            result_df['good_rate'] = result_df.good_count/result_df.good_count.sum() # 好客户的占比
            result_df['bad_rate'] = result_df.bad_count/result_df.bad_count.sum() # 坏客户的占比
            result_df['cum_good_rate'] = result_df.good_rate.cumsum()
            result_df['cum_bad_rate'] = result_df.bad_rate.cumsum()
            # 计算ks
            result_df['ks'] = round(abs(result_df.cum_good_rate - result_df.cum_bad_rate), 4)
            # 计算woe(优势比):ln(好客户的占比/坏客户的占比)*100%
            result_df['woe'] = np.log(result_df.good_rate/result_df.bad_rate)
            result_df['iv'] = (result_df.good_rate-result_df.bad_rate)*(result_df.woe)
            feature_iv = result_df.iv.sum()
            #print('feature_iv:',feature_iv)
            #print(str(char_f + ':'), feature_iv)
            all_feature_iv.append(feature_iv)
    feature_iv = pd.Series(all_feature_iv).to_frame().rename(columns={0:'feature_iv'})
    feature_name = pd.Series(char_feature_name).to_frame().rename(columns={0:'feature_name'})
    feature_name_iv = pd.concat([feature_name, feature_iv], axis=1)
    return feature_name_iv

# 连续型：计算每个特征的iv值
def cal_numf_iv(df_s, num_df, label):
#     print(num_df.shape, '\n=========计算连续特征的iv值=========')
    all_feature_iv = []
    num_feature_name = []
#     print(list(num_df))
    for num_f in list(num_df):
        if num_f not in label:
            #print(num_f)
            num_feature_name.append(num_f)
            # 分箱操作--等频：10等份
            df_bin = pd.qcut(df_s[num_f], 10, duplicates='drop')
            df_bin_label = pd.concat([df_bin, df_s[label]], axis=1)
            # 好样本统计
            good_bin_df = df_bin_label[df_bin_label[label]==0].rename(columns={num_f:'bin'})
            good_count = good_bin_df['bin'].value_counts().to_frame().reset_index().rename(columns={'index':'bin',
                                                                                                    'bin':'good_count'}).sort_values(by='bin')
            # 坏样本统计
            bad_bin_df = df_bin_label[df_bin_label[label]==1].rename(columns={num_f:'bin'})
            bad_count = bad_bin_df['bin'].value_counts().to_frame().reset_index().rename(columns={'index':'bin',
                                                                                                  'bin':'bad_count'}).sort_values(by='bin')
            # 合并统计结果
            result_df = pd.merge(good_count, bad_count, on=['bin'])
            result_df['good_rate'] = result_df.good_count/result_df.good_count.sum()
            result_df['bad_rate'] = result_df.bad_count/result_df.bad_count.sum()
            result_df['cum_good_rate'] = result_df.good_rate.cumsum()
            result_df['cum_bad_rate'] = result_df.bad_rate.cumsum()
            # 计算ks
            result_df['ks'] = round(abs(result_df.cum_good_rate - result_df.cum_bad_rate), 6)
            # 计算iv值
            result_df['woe'] = np.log(result_df.good_rate/result_df.bad_rate)
            result_df['iv'] = (result_df.good_rate-result_df.bad_rate)*(result_df.woe)
            feature_iv = result_df.iv.sum()
            all_feature_iv.append(feature_iv)
    feature_iv = pd.Series(all_feature_iv).to_frame().rename(columns={0:'feature_iv'})
    feature_name = pd.Series(num_feature_name).to_frame().rename(columns={0:'feature_name'})
    num_feature_name_iv = pd.concat([feature_name, feature_iv], axis=1)
    return num_feature_name_iv

def get_result_iv(feature_name_iv, low_th, high_th):
    # 特征iv值：评价指标
    #    a.iv<0.02, 预测效果：无明显；
    #    b.0.02=<iv<0.1, 预测效果：弱；
    #    c.0.1=<iv<0.3, 预测效果：中等；
    #    d.0.3=<iv<0.5, 预测效果：强；
    #    e.0.5=<iv, 预测效果：需考虑；
    len_f = feature_name_iv.shape[0]
    threshold_high = feature_name_iv[feature_name_iv.feature_iv<high_th].shape[0]
    threshold_low = feature_name_iv[feature_name_iv.feature_iv>=low_th].shape[0]
    if threshold_high==len_f and threshold_low==len_f:
        print('all feature...')
    else:
        print('some feature...')
    # 选取iv值在固定范围的特征,若整体iv值不高，可稍微适当调整
    feature_sel = feature_name_iv[feature_name_iv.feature_iv>=low_th]
    feature_sels = feature_sel[feature_sel.feature_iv<high_th]
    return feature_sels

# 根据相关性、iv值，综合考虑特征
def del_feature_sub(sn_df, numf_vi, th_c):
    del_feature_sub = []
    for i in range(sn_df.shape[1]):
        for j in range(sn_df.shape[1]):
            if i==j:
                continue
            else:
                if sn_df.iloc[i][j]>=th_c:
                    numf_vi_i = numf_vi.iloc[i, 1]
                    numf_vi_j = numf_vi.iloc[j, 1]
                    if numf_vi_i>=numf_vi_j:
                        del_feature_sub.append(numf_vi.iloc[j, 0])
    del_feature_subs = list(set(del_feature_sub)) # 可删除特征字段
    return del_feature_subs

# 柱状图
def plot_histogram(df):
    all_feature_iv_list = list(df.feature_name)
    all_feature_iv_df = df.feature_iv
    plt.bar(range(len(all_feature_iv_list)), all_feature_iv_df, color='rgby')
    plt.show()

# 热力图
def plot_seaborn(df):
    plt.figure(figsize=(12, 8))
    sn.heatmap(df, annot=True)
    plt.title("The feature's heatmap")
    plt.show()