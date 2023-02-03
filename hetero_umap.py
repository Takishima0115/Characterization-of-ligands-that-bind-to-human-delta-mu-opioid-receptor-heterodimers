#解析対象を限定したコード
#分類IDから外れ値を削除済み

import collections
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
#import os
import numpy as np
#import matplotlib.pyplot as plt
#matplotlib inline
import sklearn
from sklearn.decomposition import PCA
from IPython.display import display
from natsort import natsorted
import seaborn as sns
import csv
import time
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap
from scipy.sparse.csgraph import connected_components
from io import BytesIO
from PIL import Image
from mpl_toolkits.mplot3d import axes3d

df = pd.read_csv('no3d_add_id.csv', header=0, index_col=0)
#print(df)


#分類IDの読み込み
#リガンド名を書いた列の追加
uniD = pd.read_csv('../Ligand_CID/uniD.csv')
D = [int(d) for d in uniD['V1'].values.tolist()]
Dpd = df.loc[D]
Ddf = Dpd.assign(ligand='OPRD1')
#print(Ddf)

uniM = pd.read_csv('../Ligand_CID/uniM.csv')
M = [int(m) for m in uniM['V1'].values.tolist()]
Mpd = df.loc[M]
Mdf = Mpd.assign(ligand='OPRM1')

MDMD = pd.read_csv('../Ligand_CID/M_DM_D_ID.csv')
M_DM_D = [int(mdmd) for mdmd in MDMD['V1'].values.tolist()]
MDMDpd = df.loc[M_DM_D]
#MDMDdf = MDMDpd.assign(ligand='OPRD1&OPRM1&OPRM1-OPRD1')
MDMDdf = MDMDpd.assign(ligand='OPRM1-OPRD1')

#リガンド名の列が入った新しいデータフレームの作成
df_PCpoint = pd.concat([Ddf, Mdf, MDMDdf])

#マーカーの設定
colors = ["orangered", "hotpink", "deepskyblue"]
markers = ['o', 'v', 'D']

#umap
X, y = df_PCpoint.drop(["ligand"], axis=1), df_PCpoint["ligand"]
y_items = y.unique()
#print(X, y)
#print("y_items: ", y_items)
#print(df_PCpoint.drop(["ligand"], axis=1)) #"ligand"を消した。列なのでaxis = 1。行の場合はaxis = 0を指定。
#print("ligand: ", df_PCpoint["ligand"]) #IDに対応した"ligand"の列のみを出力。

"""
#2次元プロットの作成
def create_2d_umap_plots(target_X, y, y_labels, perplexity_list):
    fig, axes = plt.subplots(nrows=1, ncols=len(perplexity_list),figsize=(5*len(perplexity_list), 4))
    for i, (ax, perplexity) in enumerate(zip(axes.flatten(), perplexity_list)):
        start_time = time.time()
        #umap = umap.UMAP().fit_transform(target_X)
        Y = umap.UMAP().fit_transform(target_X)
        #print(Y)
        for n, each_label in enumerate(y_labels):
            c_plot_bool = y == each_label
            ax.scatter(Y[c_plot_bool, 0], Y[c_plot_bool, 1], c=colors[n], marker=markers[n], label="{}".format(each_label))
        end_time = time.time()
#        ax.legend()
        #ax.set_title("Perplexity: {}".format(perplexity))
        print("Time to plot perplexity {} is {:.2f} seconds.".format(perplexity,
 end_time - start_time))
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig('2D_umap_plot.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()
    return
#print(ncols)
"""
"""
#3次元プロットの作成
def create_3d_umap_plots(target_X, y, y_labels, perplexity_list):
    fig = plt.figure(figsize=(5*len(perplexity_list),4))
    for i, perplexity in enumerate(perplexity_list):
        ax = fig.add_subplot(1, len(perplexity_list), i+1, projection='3d')
        start_time = time.time()
        #tsne = TSNE(n_components=3, init='random', random_state=0, perplexity=perplexity)
        Y = umap.UMAP().fit_transform(target_X)
        for n, each_label in enumerate(y_labels):
            c_plot_bool = y == each_label
            ax.scatter(Y[c_plot_bool, 0], Y[c_plot_bool, 1], c=colors[n], marker=markers[n], label="{}".format(each_label))
        end_time = time.time()
#        ax.legend()
        ax.set_title("Perplexity: {}".format(perplexity))
        print("Time to plot perplexity {} is {:.2f} seconds.".format(perplexity, end_time - start_time))
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig('3Dumap_plot.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()
    return
"""

#標準化
scaler = StandardScaler() #標準化
scaled_X = scaler.fit_transform(X)

Y = umap.UMAP().fit_transform(scaled_X)

#全リガンド
for i, each_quality in enumerate(y_items):
    start_time = time.time()
    c_plot_bool = y == each_quality
    plt.scatter(Y[c_plot_bool, 0], Y[c_plot_bool, 1], c= colors[i], marker=markers[i], label="Ligand: {}".format(each_quality))
    each_df = pd.DataFrame(Y[c_plot_bool])   
end_time = time.time()
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0,), borderaxespad=0)
plt.savefig("2D_umap.png".format(each_quality) , bbox_inches='tight', pad_inches=0.1)
print("Time to plot is {:.2f} seconds.".format(end_time - start_time))
fig = plt.figure()


#それぞれのリガンド
#figu = plt.figure()
for i, each_quality in enumerate(y_items):
    start_time = time.time()
    #print(i)
    #print(i, each_quality)
    c_plot_bool = y == each_quality
    #print(y)
    plt.scatter(Y[c_plot_bool, 0], Y[c_plot_bool, 1], c= colors[i], marker=markers[i], label="Ligand: {}".format(each_quality))
    each_df = pd.DataFrame(Y[c_plot_bool])   
    #print("{}_data".format(each_quality))
    #print(Y)
    #name_list = ["OPRD1", "OPRM1", "OPRM1-OPRD1"]
    #print(each_df)
    each_df.to_csv("{}_UMAP.csv".format(each_quality))
    #each_df.to_csv("each_ligand_t-SNE.csv")
    #updated_df = pd.read_csv("work/each_ligand_t-SNE.csv")
    #print(each_df)
    end_time = time.time()
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0,), borderaxespad=0)
    #ax.legend()
    #each_df = pd.DataFrame(data={'x': Y[:, 0], 'y': Y[:, 1], 'label': markers})
    #print(df)
    #plt.title('perplexity:50')
    plt.savefig("{}_2D_perplexity50.png".format(each_quality) , bbox_inches='tight', pad_inches=0.1)
    #print("Time to plot is {:.2f} seconds.".format(end_time - start_time))
    fig = plt.figure()
    #each_df.to_csv("{}_t-SNE.csv".format(each_quality))

"""
#3次元
start_time = time.time()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#tsne = TSNE(n_components=3, init='random', random_state=0, perplexity=perplexity)
Y = umap.UMAP().fit_transform(scaled_X)
for i, each_quality in enumerate(y_items):
    c_plot_bool = y == each_quality
    ax.scatter(Y[c_plot_bool, 0], Y[c_plot_bool, 1], c=colors[i], marker=markers[i], label="Ligand: {}".format(each_quality))
end_time = time.time()
ax.view_init(elev=10, azim=50) #図の回転。不要な場合はコメントアウト
ax.legend()
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title('perplexity:50')
plt.savefig('3D_perplexity50.png', bbox_inches='tight', pad_inches=0.1)
print("Time to plot is {:.2f} seconds.".format(end_time - start_time))
plt.show()
"""