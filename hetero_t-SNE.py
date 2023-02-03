#解析対象を限定したコード
#分類IDから外れ値を削除済み

import collections
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly
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
from io import BytesIO
from PIL import Image

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
#print(MDMDdf)


#マーカーの設定
colors = ["orangered", "hotpink", "deepskyblue"]
markers = ['o', 'v', 'D']

#t-SNE
X, y = df_PCpoint.drop(["ligand"], axis=1), df_PCpoint["ligand"]
y_items = y.unique()
#print(y)

"""
#2次元プロットの作成
def create_2d_tsne_plots(target_X, y, y_labels, perplexity_list):
    fig, axes = plt.subplots(nrows=1, ncols=len(perplexity_list),figsize=(5*len(perplexity_list), 4))
    for i, (ax, perplexity) in enumerate(zip(axes.flatten(), perplexity_list)):
        start_time = time.time()
        tsne = TSNE(n_components=2, init='random', random_state=0, perplexity=perplexity)
        Y = tsne.fit_transform(target_X)
        for n, each_label in enumerate(y_labels):
            c_plot_bool = y == each_label
            ax.scatter(Y[c_plot_bool, 0], Y[c_plot_bool, 1], c=colors[n], marker=markers[n], label="{}".format(each_label))
        end_time = time.time()
#        ax.legend()
        ax.set_title("Perplexity: {}".format(perplexity))
        print("Time to plot perplexity {} is {:.2f} seconds.".format(perplexity, end_time - start_time))
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig('2Dt-SNE_plot.png', bbox_inches='tight', pad_inches=0.1)
#    plt.show()
    return
"""
"""
#3次元プロットの作成
def create_3d_tsne_plots(target_X, y, y_labels, perplexity_list):
    fig = plt.figure(figsize=(5*len(perplexity_list),4))
    for i, perplexity in enumerate(perplexity_list):
        ax = fig.add_subplot(1, len(perplexity_list), i+1, projection='3d')
        start_time = time.time()
        tsne = TSNE(n_components=3, init='random', random_state=0, perplexity=perplexity)
        Y = tsne.fit_transform(target_X)
        for n, each_label in enumerate(y_labels):
            c_plot_bool = y == each_label
            ax.scatter(Y[c_plot_bool, 0], Y[c_plot_bool, 1], c=colors[n], marker=markers[n], label="{}".format(each_label))
        end_time = time.time()
#        ax.legend()
        ax.set_title("Perplexity: {}".format(perplexity))
        print("Time to plot perplexity {} is {:.2f} seconds.".format(perplexity, end_time - start_time))
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig('3Dt-SNE_plot.png', bbox_inches='tight', pad_inches=0.1)
#    plt.show()
    return
"""

perplexity_list = [2, 5, 30, 50, 100]

#標準化
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)


#5種類のperplexityの図を作成
#create_2d_tsne_plots(scaled_X, y, y_items, perplexity_list)
#print("scaled_X:", scaled_X)
#print("scaled_X_len:", len(scaled_X))
#print("y:", y)
#print("y_len:", len(y))
#print("y_items:", y_items)
#print("y_items_len:", len(y_items))
#print(perplexity_list)
#create_3d_tsne_plots(scaled_X, y, y_items, perplexity_list)



#1つのperplexityのみでt-SNE
perplexity=50


#2次元
left_OPRD1_OPRM1 = []
leftt_OPRD1 = []
left_OPRM1 = []
right_hetero_OPRD1_OPRM1 = []

start_time = time.time()
fig, ax = plt.subplots()
tsne = TSNE(n_components=2, init='random', random_state=0, perplexity=perplexity)
Y = tsne.fit_transform(scaled_X)
#print(Y)
for i, each_quality in enumerate(y_items):
    c_plot_bool = y == each_quality
    ax.scatter(Y[c_plot_bool, 0], Y[c_plot_bool, 1], c= colors[i], marker=markers[i], label="Ligand: {}".format(each_quality))
end_time = time.time()
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax.legend()
plt.title('perplexity:50')
plt.savefig('2D_perplexity50.png', bbox_inches='tight', pad_inches=0.1)
print("Time to plot is {:.2f} seconds.".format(end_time - start_time))


list_ligand = []
list_TSNE = []
right_TSNE = []
start_time = time.time()
fig, ax = plt.subplots()
#fig = plt.figure()
tsne = TSNE(n_components=2, init='random', random_state=0, perplexity=perplexity)
Y = tsne.fit_transform(scaled_X)
#figu = plt.figure()
for i, each_quality in enumerate(y_items):
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
    each_df.to_csv("{}_t-SNE.csv".format(each_quality))
    each_df.to_csv("each_ligand_t-SNE.csv")
    #updated_df = pd.read_csv("work/each_ligand_t-SNE.csv")
    #print(each_df)
    end_time = time.time()
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0,), borderaxespad=0)
    #ax.legend()
    #each_df = pd.DataFrame(data={'x': Y[:, 0], 'y': Y[:, 1], 'label': markers})
    #print(df)
    plt.title('perplexity:50')
    plt.savefig("{}_2D_perplexity50.png".format(each_quality), bbox_inches='tight', pad_inches=0.1)
    print("Time to plot is {:.2f} seconds.".format(end_time - start_time))
    fig = plt.figure()
    each_df.to_csv("{}_t-SNE.csv".format(each_quality))

"""
#3次元
start_time = time.time()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
tsne = TSNE(n_components=3, init='random', random_state=0, perplexity=perplexity)
Y = tsne.fit_transform(scaled_X)
for i, each_quality in enumerate(y_items):
    c_plot_bool = y == each_quality
    ax.scatter(Y[c_plot_bool, 0], Y[c_plot_bool, 1], c=colors[i], marker=markers[i], label="Ligand: {}".format(each_quality))
end_time = time.time()
#ax.view_init(elev=10, azim=50) #図の回転。不要な場合はコメントアウト
#ax.legend()
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title('perplexity:50')
plt.savefig('3D_perplexity50.png', bbox_inches='tight', pad_inches=0.1)
print("Time to plot is {:.2f} seconds.".format(end_time - start_time))
"""

"""
#GIFアニメーションの作成
def render_frame(angle):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    tsne = TSNE(n_components=3, init='random', random_state=0, perplexity=perplexity)
    Y = tsne.fit_transform(scaled_X)
    for i, each_quality in enumerate(y_items):
        c_plot_bool = y == each_quality
        ax.scatter(Y[c_plot_bool, 0], Y[c_plot_bool, 1], c= colors[i], marker=markers[i], label="Ligand: {}".format(each_quality))
    ax.view_init(30, angle)
    plt.close()
    buf = BytesIO()
    fig.savefig(buf, bbox_inches='tight', pad_inches=0.0)
    return Image.open(buf)


#GIFアニメーション
images = [render_frame(angle) for angle in range(360)]
images[0].save('3D_perplexity50.gif', save_all=True, append_images=images[1:], duration=100
, loop=0)
"""
