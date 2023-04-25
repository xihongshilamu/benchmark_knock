import torch
import pandas as pd
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import time
from sklearn.model_selection import train_test_split
import numpy as np

time_start = time.time()


device = torch.device("cuda:5")

# device = torch.device('cpu')

class MLP(nn.Module):
    def __init__(self,n_feature, n_hidden,out):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_feature, n_hidden)
        self.fc2 = nn.Linear(n_feature, out)

    def forward(self, x):
        x = self.fc1(x)
        out = torch.relu(x)
        out = self.fc2(out)
        out = F.relu(out)

        return out

data_c = pd.read_csv('/data/share/data/data_c_del_rowcol.csv')
print(data_c)
data_c.set_index('Gene Name', drop=True, append=False, inplace=True)
data_c = np.log(data_c + 1)
data_t = pd.read_csv('/data/share/data/data_t_del_rowcol.csv')
data_t.set_index('Gene Name', drop=True, append=False, inplace=True)
data_t = np.log(data_t + 1)

columns_name = list(data_t.columns)
data_c.columns = columns_name
# 将要敲除的基因值置为 0
data_c_copy = data_c.copy()
for i in data_c.columns:
    ko_name = i.split('_')[0]
    ko_name = ko_name.split('/')[0]
    data_c.at[ko_name, i] = 0


data_test_c1 = pd.read_csv('./data_c_choose_3000.csv')

data_test_c1.set_index('Gene Name', drop=True, append=False, inplace=True)
name_test = list(data_test_c1.index)

data_c=data_c.loc[name_test]

data_t=data_t.loc[name_test]


name_drop = data_t.index
data_c = data_c.values.T
data_t = data_t.values.T


change1 = data_t -data_c
# import pickle
# import sys
# with open('./data_c.pkl', 'wb') as f:
#     pickle.dump(data_c, f)
    
# with open('./data_t.pkl', 'wb') as f:
#     pickle.dump(data_t, f)
    
# sys.exit()
input_node = len(name_test)

net = MLP(n_feature=input_node, n_hidden=input_node,out=input_node).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

def criterion2(a,b):
    loss2 = torch.nn.MSELoss(reduction='none').to(device)
    loss  =loss2(a,b)
    loss = torch.mean(loss)
    return loss



train_c, test_c, train_t, test_t = train_test_split(data_c, data_t)

data_c_train = torch.tensor(train_c, dtype=torch.float32)
data_t_train = torch.tensor(train_t, dtype=torch.float32)
data_c_test = torch.tensor(test_c, dtype=torch.float32)
data_t_test = torch.tensor(test_t, dtype=torch.float32)
# print(data_t_train)

torch_dataset = data.TensorDataset(data_c_train, data_t_train)
torch_dataset1 = data.TensorDataset(data_c_test, data_t_test)

# 把dataset放入DataLoader
loader = data.DataLoader(
    dataset=torch_dataset,
    batch_size=1024,             # 每批提取的数量
    shuffle=True,             # 要不要打乱数据（打乱比较好）
    num_workers=2           # 多少线程来读取数据
)
loader1 = data.DataLoader(
    dataset=torch_dataset1,
    batch_size=1024,             # 每批提取的数量
    shuffle=True,             # 要不要打乱数据（打乱比较好）
    num_workers=2           # 多少线程来读取数据
)



@torch.no_grad()
def test(loader1):
    net.eval()
    data_test_c11 = data_test_c1.values.T
    data_c_test1 = torch.tensor(data_test_c11, dtype=torch.float32).to(device)
    data_pre = net(data_c_test1)
    data_pre_num = data_pre.detach().cpu().numpy()
    data_pre_num = data_pre_num.T
    print(data_pre_num)
    data_test_res = pd.DataFrame(data_pre_num)
    data_test_res['Gene Name'] = name_test
    data_test_res.set_index('Gene Name', drop=True, append=False, inplace=True)
    print(data_test_res)
    data_test_res.to_csv('result100_2_3000_tf_ko_div0_312_ko_200_del0.csv', index=True)

if __name__ == '__main__':

    for epoch in range(100):    # 对整套数据训练3次
        for step, (batch_x, batch_y) in enumerate(loader):  # 每一步loader释放一小批数据用来学习
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            prediction = net(batch_x)  # 用网络预测一下
            loss = criterion2(prediction,batch_y)  # 计算损失
            optimizer.zero_grad()  # 清除上一步的梯度
            loss.backward()  # 反向传播, 计算梯度
            optimizer.step()  # 优化一步
            print(loss)
    test(loader1)
print(net.fc1.weight.data)
# 获取第一个全连接层的权重矩阵
weights = net.fc1.weight.data.cpu().numpy()
# 将权重矩阵保存到文件中
np.save('weights_dia0.npy', weights)
time_end = time.time()
print("Elapsed time: %.2f seconds" % (time_end - time_start))


