expriment: 实验过程，
    model为各个单一模型和集成模型的训练过程（LR,SVM,RF,XGBoost,Voting,avergae,Stacking,Stacking改），
        .pkl为训练好的模型
    dataProcess为数据处理过程,包含RF随机森林特征重要性筛选、MI-VAR互信息+方差联合筛选、AutoEncoder自编码筛选
    dataset为经过数据处理的训练集和测试集
expriment/NewExpriment: 更新实验过程，更换训练集和测试集
results为论文实验结果

hmp2019数据集：样本数1627，患病样本1201，健康样本426，（实际共132名受试者，每人多次采样）

prjeb1220数据集：样本数1312，患病样本871，健康样本441，（实际共396名受试者，每人多次采样）

NielsenHB_2014数据集：样本数396，患病样本148，健康样本248，

ijauz(ljazUZ_2017)数据集：样本数94，患病样本56，健康样本38

hab数据集（HallAB_2017）：样本数259，患病样本185，健康样本74

