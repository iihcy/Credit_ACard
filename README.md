信贷申请评分卡(Credit_ACard)
==== 
项目流程：数据获取、数据清洗(特征初筛)、特征工程、模型建立与评价等; 

数据介绍：
-------
1.信贷申请数据：
42535个样本，144个特征，其中该数据包含了借款后的信息，共18个特征(如代码所示)；

2.字段的基本描述：
id号，职位/职务、借款期限(36/60)、工作年限、房产权、用途、邮政编码、地址状态、贷款额度、年收入、过去两年逾期次数、贷款状态等；

项目基本介绍：
---------
1.原始数据共有144个特征，经过不符合A卡的字段、敏感特征信息、去重、缺失率与特征值的分析、脏数据的整理，以及特征工程技术(特征衍生、特征筛选等)，最终得到12个特征，作为入模特征；

2.模型建立：采用CatBoost、LightGBM与XGBoost建模，可获取各自的信贷申请评分，以便进行对比与分析等；

模型结果：
---------
1.CatBoost：

【Train】The model accuracy is 0.8798, AUC is 0.7438;

【Test】The model accuracy is 0.8789, AUC is 0.7109, KS is 0.3168;

【Score】Max(score)=694.858773, Min(score)=440.530273;

2.LightGBM：

【Train】The model accuracy is 0.8762, AUC is 0.7604;

【Test】The model accuracy is 0.8797, AUC is 0.7037, KS is 0.3044;

【Score】Max(score)=675.494371, Min(score)=503.161308;

1.XGBoost：

【Train】The model accuracy is 0.8822, AUC is 0.7800;

【Test】The model accuracy is 0.8787, AUC is 0.6952, KS is 0.2851;

【Score】Max(score)=558.257202, Min(score)=488.194519;

*模型其它效果以及数据，详见代码实验！ 

备注：本可以利用二维交叉矩阵(基于申请评分)进一步验证模型效果，但官方的标准数据比较难获取，因此无法展示业界的模型评价步骤！
