import pandas as pd

origin_path = 'dataset/house_sales.csv'
train_path ='dataset/train.csv'
test_path = 'dataset/test.csv'

origin_df = pd.read_csv(origin_path)
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

test_indexs = test_df.index.values
print(test_indexs)
test_gt = origin_df.iloc[test_indexs, :]

print(len(origin_df))
print(len(train_df) + len(test_df))

print(test_gt)
test_gt.to_csv('anser.csv', index_label=True)