from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

def main():
    print('LOADING RAW DATASET.....')
    trainset = pd.read_excel('SWaT_Dataset_Normal_v1.xlsx', engine='openpyxl', header=1)
    testset = pd.read_excel('SWaT_Dataset_Attack_v0.xlsx', engine='openpyxl', header=1)

    print('PREPROCESSING.....')
    rename_cols = {' AIT201': 'AIT201',
                   ' MV101': 'MV101',
                   ' MV201': 'MV201',
                   ' MV303': 'MV303',
                   ' P201': 'P201',
                   ' P202': 'P202',
                   ' P204': 'P204'}
    testset.rename(columns=rename_cols, inplace=True)
    testset.replace('A ttack', 'Attack', inplace=True)

    uniq_value_cols = (trainset.nunique() == 1) & (testset.nunique() == 1)
    trainset.drop(columns=trainset.columns[uniq_value_cols], inplace=True)
    testset.drop(columns=testset.columns[uniq_value_cols], inplace=True)

    test_labels = testset['Normal/Attack']

    trainset.drop(columns=[' Timestamp','Normal/Attack'], inplace=True)
    testset.drop(columns=[' Timestamp','Normal/Attack'], inplace=True)

    print('NORMALIZING.....')
    scaler = MinMaxScaler()
    trainset = scaler.fit_transform(trainset)
    testset = scaler.transform(testset)

    test_labels.replace('Normal', 0, inplace=True)
    test_labels.replace('Attack', 1, inplace=True)
    test_labels = test_labels.to_numpy()
    test_anomaly_index = np.where(test_labels == 1)[0]
    data_dict = {'training': trainset,
                 'test': testset,
                 'idx_anomaly_test': test_anomaly_index}
    print('SAVING TO NPZ FILE...')
    np.savez('swat.npz', **data_dict)
    print('DONE.')

if __name__ == '__main__':
    main()
