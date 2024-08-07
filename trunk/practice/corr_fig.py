import pandas as pd
import numpy as np
import pickle

file_path = "/Users/taegyu.hwang/Documents/tghwang_git_repo/ipitems_analysis/data/colley/license_tag_list.csv"
license_name_list = [r["license_name"] for _, r in pd.read_csv(file_path).iterrows()]
file_path = "/Users/taegyu.hwang/Documents/tghwang_git_repo/ipitems_analysis/data/colley/category_tag_list.csv"
category_name_list = [r["category_name"] for _, r in pd.read_csv(file_path).iterrows()]

target_name_list = license_name_list + category_name_list
# target_tags_set = set(target_name_list)
file_path = "/Users/taegyu.hwang/Documents/tghwang_git_repo/ipitems_analysis/data/colley/tag_list.csv"
tags_name_list = [r["tag"] for _, r in pd.read_csv(file_path).iterrows()]

tags_count = len(tags_name_list)
file_path = "/Users/taegyu.hwang/Documents/tghwang_git_repo/ipitems_analysis/temp/colley/tags_score.bin"
item_corr: np.ndarray = None
with open(file=file_path, mode="rb") as fin:
    item_corr: np.ndarray = pickle.load(fin)
    fin.close()

corr_list_dict = list()
index_name_list = list()
for i in range(tags_count):
    if not tags_name_list[i] in target_name_list:
        continue
    corr_list_dict.append({tags_name_list[i]: list()})
    index_name_list.append(tags_name_list[i])
    # corr_list_dict.append({tags_name_list[i]:[item_corr[i][idx] for idx in range(tags_count)]})
df = pd.DataFrame(corr_list_dict, index=index_name_list)

for i in range(tags_count):
    corr_list = list()
    for idx in range(tags_count):
        if not tags_name_list[idx] in index_name_list:
            continue
        corr_list.append(item_corr[i][idx])
    df[tags_name_list[i]] = corr_list

import matplotlib.pyplot as plt

plt.pcolor(df)
plt.pcolor(df, cmap="Greys", vmin=-0.1, vmax=0.1)
# plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
# plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
plt.title("tags score", fontsize=12)
plt.xlabel("src tags", fontsize=10)
plt.ylabel("dest tags", fontsize=10)
plt.colorbar()
