import pandas as pd
import numpy as np
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def tag_test(datac,datat):
    datac = datac.values
    datat = datat.values
    print(datac.shape)
    change = datat - datac
    # print(change)
    mask_up5x = change >= np.log(5)
    # print(mask_up5x)
    mask_up2x = (change >= np.log(2)) & (change < np.log(5))
    mask_none = (change > -np.log(2)) & (change < np.log(2))
    mask_down2x = (change <= -np.log(2)) & (change > -np.log(5))
    mask_down5x = change <= -np.log(5)
    # target = pd.DataFrame(np.zeros_like(data_c), index=data_c.index, columns=data_c.columns, dtype=int)
    target = pd.DataFrame(np.zeros_like(datac),dtype=int)
    target[mask_down5x] = 0
    target[mask_down2x] = 1
    target[mask_none] = 2
    target[mask_up2x] = 3
    target[mask_up5x] = 4
    target = target.values
    print(target)
    print(target.shape)

    return target

def heat_map(cm):
    # 绘制 heatmap
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges)

    # 设置颜色条
    plt.colorbar()

    # 设置坐标轴标签
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')

    # 设置坐标轴刻度
    tick_marks = np.arange(5)
    plt.xticks(tick_marks, ['down_5x', 'down_2x', 'none', 'up_2x', 'up_5x'])
    plt.yticks(tick_marks, ['down_5x', 'down_2x', 'none', 'up_2x', 'up_5x'])

    # 在格子中显示数字
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j],'.0f'),
                 ha='center', va='center',
                 color='black' if cm[i, j] > thresh else 'black')
    plt.savefig('heatmap.png')

    plt.show()
    plt.close()

def heat_map_acc(acc):
    # 绘制 heatmap
    # cmap = plt.cm.get_cmap('Blues')
    plt.imshow(acc, interpolation='nearest', cmap=plt.cm.Oranges)


    # 设置颜色条
    plt.colorbar()

    # 设置坐标轴标签
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')

    # 设置坐标轴刻度
    tick_marks = np.arange(5)
    plt.xticks(tick_marks, ['down_5x', 'down_2x', 'none', 'up_2x', 'up_5x'])
    plt.yticks(tick_marks,['down_5x', 'down_2x', 'none', 'up_2x', 'up_5x'])

    # 在格子中显示数字
    for i, j in np.ndindex(acc.shape):
        plt.text(j, i, format(acc[i, j], '.2f'),
                 ha='center', va='center',
                 color='black')
    plt.savefig('acc.png')
    plt.show()
    plt.close()
    
i_choose=0
res = pd.read_csv('result100_2_3000_tf_ko_div0_312_ko_200_del0.csv')
name = list(res['Gene Name'])
res.set_index('Gene Name', drop=True, append=False, inplace=True)
res=res.iloc[:,0:i_choose+6]
# res=res.iloc[:,i_choose:i_choose+1]
data_t  =pd.read_csv('data_c_choose_3000.csv')
print(data_t.head)
# data_t = pd.DataFrame(data_t, dtype=float)
data_t = np.log1p(data_t)
gene_name = pd.read_csv('/data/share/data/name_gene1.csv')
data_t['Gene Name'] = list(gene_name['name'])
data_t.set_index('Gene Name', drop=True, append=False, inplace=True)
data_t=data_t.iloc[:,i_choose:i_choose+6]
data_c = pd.read_csv('data_c_choose_3000.csv')
# data_c = pd.DataFrame(data_c, dtype=float)
data_c = np.log1p(data_c)
data_c['Gene Name'] = list(gene_name['name'])
data_c.set_index('Gene Name', drop=True, append=False, inplace=True)
data_c=data_c.iloc[:,i_choose:i_choose+6]

tf=name
data_t = data_t.loc[tf]
print(data_t)
data_c = data_c.loc[tf]
print(data_c)
# data_c.to_csv('data_c_choose.csv',index=True)
# data_t.to_csv('data_t_choose.csv',index=True)
print(data_c)
print(res)

res_data = res.values-data_t.values
print(res_data)

def count_percentage_in_range(arr, lower, upper):
    count = 0
    for num in arr:
        if lower <= num <= upper:
            count += 1
    return count / len(arr)

flat_arr = []
for row in res_data:
    for element in row:
        flat_arr.append(element)

c = count_percentage_in_range(flat_arr,-np.log(5),-np.log(2))
c1 = count_percentage_in_range(flat_arr,-np.log(2),np.log(2))
c2 = count_percentage_in_range(flat_arr,np.log(2),np.log(5))
c3 = count_percentage_in_range(flat_arr,-100000,-np.log(5))
c4 = count_percentage_in_range(flat_arr,np.log(5),1000000000)
print(c)
print(c1)
print(c2)
print(c3)
print(c4)


true = tag_test(data_c, data_t)
pred = tag_test(data_c, res)
y_true=true.flatten()
    # print(true.shape)
y_pred = pred.flatten()
# 计算混淆矩阵
# 定义一个包含多个元素的列表
lst_tr = list(y_true)
lst_pre=list(y_pred)

# 使用Counter类统计每个元素出现的次数
counts = Counter(lst_tr)
counts_p = Counter(lst_pre)
print(counts)
print(counts_p)
cm = confusion_matrix(y_true, y_pred)

# 定义类别标签

# 计算混淆矩阵
cm = np.zeros((5, 5))
for i in range(len(y_true)):
    cm[y_true[i]][y_pred[i]] += 1

print(cm)
# 绘制混淆矩阵

heat_map(cm)

acc = np.zeros((5, 5))


for i in range(5):
    for j in range(5):
        if cm[i, :].sum() == 0:
            acc[i] = 0
        else:
            acc[i, j] = cm[i, j] / cm[i, :].sum()

heat_map_acc(acc)



# 画回归直线
data_t=data_t.values
res=res.values
y_test=data_t.flatten()
    # print(true.shape)
y_pred = res.flatten()
plt.scatter(y_test, y_pred,s=0.5,color='red')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlim([0, 12])
plt.ylim([0, 12])
t1 = np.linspace(0,10,10)
t2 = np.linspace(0,10,10)
# plt.plot(y_test, y_pred, color='orange', label='Fitted')

# plt.plot(t1, 0.96*t1,color='orange', linestyle='--') #线性回归线，5.5是斜
plt.plot(t1, t1+0.7,color='green', linestyle='--') #线性回归线，5.5是斜
plt.plot(t1, t1+1.6,color='blue', linestyle='--') #线性回归线，5.5是斜
plt.plot(t2, t2-0.7,color='green', linestyle='--') #线性回归线，5.5是斜
plt.plot(t2, t2-1.6,color='blue', linestyle='--') #线性回归线，5.5是斜
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.savefig('p100_3000_2_zong_tf_1.png')
plt.show()

