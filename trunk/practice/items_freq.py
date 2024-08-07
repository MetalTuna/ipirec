import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

type_list = ["view", "like", "purchase"]

src_dir_path = (
    "/Users/taegyu.hwang/Documents/tghwang_git_repo/ipitems_analysis/data/colley"
)

items_id_set = set()
users_decision_dict = {
    "view": dict(),
    "like": dict(),
    "purchase": dict(),
}
"""
Key: "view", "like", "purchase" (str)
Value: (dict)
{
    Key: item_id (int)
    Value: user_ids set()
}
"""

for type_str in type_list:
    file_path = f"{src_dir_path}/{type_str}_list.csv"
    item_to_users_dict = dict()
    for _, r in pd.read_csv(file_path).iterrows():
        user_id = int(r["user_id"])
        item_id = int(r["item_id"])
        items_id_set.add(item_id)
        if not item_id in item_to_users_dict:
            item_to_users_dict.update({item_id: set()})
        item_to_users_dict[item_id].add(user_id)
    # end : for (actions)
    users_decision_dict[type_str] = item_to_users_dict
# end : for (DTypes)

# view_freq_list = list()
# like_freq_list = list()
# purchase_freq_list = list()
freq_list_dict = dict()
for type_str in type_list:
    freq_list = list()
    # freq_list_dict.update({type_str: list()})
    freq_dict: dict = users_decision_dict[type_str]
    for item_id in items_id_set:
        freq_list.append(len(freq_dict[item_id]) if item_id in freq_dict else 0)
    freq_list_dict.update({type_str: freq_list})
