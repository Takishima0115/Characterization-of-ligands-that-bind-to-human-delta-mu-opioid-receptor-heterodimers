#解析対象を絞ったコード
#分類IDから外れ値を削除済み

import collections
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline
import sklearn
from sklearn.decomposition import PCA
from IPython.display import display
from natsort import natsorted
import seaborn as sns
import csv
import time
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import axes3d
from io import BytesIO
from PIL import Image

PC_all = pd.DataFrame([])
df = pd.read_csv('best_descriptors.csv', index_col=0)
#print(df)

id = pd.read_csv('10396full.csv') 
b = id['id'].astype(str)

dfs = df.iloc[:, :].apply(lambda x: (x-x.mean())/x.std(), axis=0)
print(dfs)
pca = PCA()
pca.fit(dfs)
# データを主成分空間に写像
feature = pca.transform(dfs)
#print(feature)
PCpoint = pd.DataFrame(feature, columns=['PC{}'.format(x + 1) for x in range(len(dfs.columns))])
#print(PCpoint)
add_id = pd.concat([PCpoint,b], axis=1)
#print(add_id)
add_id_index = add_id.set_index('id')
#print(add_id_index['PC1'].describe())
#print(add_id_index['PC2'].describe())


#分類IDの読み込み
#リガンド名を書いた列の追加
uniD = pd.read_csv('../Ligand_CID/uniD.csv')
D = [str(d) for d in uniD['V1'].values.tolist()]
Dpd = add_id_index.loc[D]
#print(uniD['V1'].values.tolist())
Ddf = Dpd.assign(ligand='OPRD1')

uniM = pd.read_csv('../Ligand_CID/uniM.csv')
M = [str(m) for m in uniM['V1'].values.tolist()]
Mpd = add_id_index.loc[M]
#print(uniM['V1'].values.tolist())
Mdf = Mpd.assign(ligand='OPRM1')

MDMD = pd.read_csv('../Ligand_CID/M_DM_D_ID.csv')
M_DM_D = [str(mdmd) for mdmd in MDMD['V1'].values.tolist()]
MDMDpd = add_id_index.loc[M_DM_D]
#print(MDMDpd)
MDMDdf = MDMDpd.assign(ligand='OPRM1-OPRD1')

#リガンド名の列が入った新しいデータフレームの作成
df_PCpoint = pd.concat([Ddf, Mdf, MDMDdf])
#print(df_PCpoint)

#マーカー設定
colors = ["orangered", "hotpink", "deepskyblue"]
markers = ['o', 'v', 'D']

X = [Dpd, Mpd, MDMDpd]
y = df_PCpoint["ligand"]
y_items = y.unique()


#第三主成分までの累積寄与率
ev = pd.DataFrame(pca.explained_variance_ratio_, index=['PC{}'.format(x + 1) for x in range(len(dfs.columns))], columns=['寄与率']).T
#display(ev)
print(ev)
evPC5 = ev.iloc[:,[0,1,2,3]]
PC_all = PC_all.append(evPC5)
PC_all['ccr'] = PC_all['PC1'] + PC_all['PC2'] + PC_all['PC3']
print('第３主成分までの累積寄与率')
print(PC_all)


#各主成分の固有ベクトル
vector = pd.DataFrame(pca.components_, columns=df.columns[:], index=["PC{}".format(x + 1) for x in range(len(dfs.columns))])
print('各主成分の固有ベクトル')
print(vector)
#display(vector)


#各リガンドと成分の図
fig = plt.figure()
for i, each_ligand in enumerate(y_items):
    plt.scatter(X[i].PC1, X[i].PC2, alpha=0.8, s=10, c=colors[i], marker=markers[i], label=each_ligand)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig("plot_components.png", bbox_inches='tight', pad_inches=0.1)

#各リガンドと成分の図(単一)
fig = plt.figure()
for i, each_ligand in enumerate(y_items):
    plt.scatter(X[i].PC1, X[i].PC2, alpha=0.8, s=10, c=colors[i], marker=markers[i], label=each_ligand)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig("{}_plot.png".format(each_ligand), bbox_inches='tight', pad_inches=0.1)
    fig = plt.figure()


#第一、第二主成分の固有ベクトル図
plt.figure(figsize=(6, 6))
for x, y, name in zip(pca.components_[0], pca.components_[1], df.columns[1:]):
    plt.text(x, y, name)
plt.scatter(pca.components_[0], pca.components_[1], alpha=0.8)
plt.grid()
plt.xlabel("PC1")
plt.ylabel("PC2")
#plt.show()
plt.savefig("PC1_PC2.png")

"""
#第二、第三主成分の固有ベクトル図
plt.figure(figsize=(6, 6))
for x, y, name in zip(pca.components_[1], pca.components_[2], df.columns[1:]):
    plt.text(x, y, name)
plt.scatter(pca.components_[1], pca.components_[2], alpha=0.8)
plt.grid()
plt.xlabel("PC2")
plt.ylabel("PC3")
#plt.show()
plt.savefig("PC2_PC3.png")
"""


#各リガンドと成分の図(3次元)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i, each_ligand in enumerate(y_items):
    ax.scatter(X[i].PC1, X[i].PC2, X[i].PC3, alpha=0.8,  s=10, c=colors[i], marker=markers[i], label=each_ligand)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig("3D_components_plot.png", bbox_inches='tight', pad_inches=0.1)


#散布図行列(主成分軸)
sns.pairplot(df_PCpoint, hue='ligand',
            palette={'OPRD1': "orangered",
                     'OPRM1': "hotpink",
                     'OPRM1-OPRD1': "deepskyblue"},
            markers=['o', 'v', 'D'],
            vars=['PC{}'.format(x+1) for x in range(10)]).savefig('pairplot.png', bbox_inches='tight', pad_inches=0.1)


'''
#GIFアニメーションの作成
def render_frame(angle):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, each_ligand in enumerate(y_items):
        ax.scatter(X[i].PC1, X[i].PC2, X[i].PC3, alpha=0.8,  s=10, c=colors[i], marker=markers[i], label=each_ligand)
    ax.view_init(30, angle)
    plt.close()
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    buf = BytesIO()
    fig.savefig(buf, bbox_inches='tight', pad_inches=0.0)
    return Image.open(buf)

images = [render_frame(angle) for angle in range(360)]
images[0].save('3Dcomponents.gif', save_all=True, append_images=images[1:], duration=100, loop=0)


#3次元グラフを回転する
#主成分得点図
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i, each_ligand in enumerate(y_items):
    ax.scatter(X[i].PC1, X[i].PC2, X[i].PC3, alpha=0.8,  s=10, c=colors[i], marker=markers[i], label=each_ligand)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.view_init(elev=10, azim=50)
plt.savefig('3Dcomponents(roll).png', bbox_inches='tight', pad_inches=0.1)

'''


