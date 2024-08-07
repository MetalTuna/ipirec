## build-in
import os
import sys
import pickle

## 3rd Pty.
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


# [DEF] Environment variables
__dir_name__ = os.path.basename(os.path.dirname(__file__))
WORKSPACE_HOME = os.path.dirname(__file__).replace(f"/{__dir_name__}", "")
""".../ipirec"""
RESOURCE_DIR_PATH = f"{WORKSPACE_HOME}/resources"

# [SET] Env.append($WORKSPACE_HOME)
sys.path.append(WORKSPACE_HOME)


## Custom LIB.
from core import *
from colley import *
from ipirec import *


class ObservTagsScore:

    def __init__(self) -> None:
        self._model: BaseModel = None
        self._estimator: BaseEstimator = None

    # end : init()

    def read_estimator(
        self,
        set_no: int,
    ) -> AdjustedBiasedCorrelationEstimator:
        _file_path = self.get_estimator_pickle_path(set_no)
        estimator: AdjustedBiasedCorrelationEstimator = self._read_pickle_(_file_path)
        return estimator

    # end : public AdjustedBiasedCorrelationEstimator read_estimator()

    def read_model(
        self,
        set_no: int,
    ) -> CorrelationModel:
        _file_path = self.get_model_pickle_path(set_no)
        model: CorrelationModel = self._read_pickle_(_file_path)
        return model

    # end : public CorrelationModel read_model()

    def _read_pickle_(
        self,
        pickle_file_path: str,
    ):
        inst = None
        if not os.path.exists(pickle_file_path):
            raise FileNotFoundError()
        with open(
            file=pickle_file_path,
            mode="rb",
        ) as fin:
            inst = pickle.load(fin)
            fin.close()
        # end : StreamReader()
        return inst

    # end : protected Any read_pickle()

    @staticmethod
    def labeled_tags_dict() -> dict:
        _ifc_dict = dict()
        file_path = str.format(
            "{0}/tags_dictionary/IFC_tagging.csv",
            RESOURCE_DIR_PATH,
        )
        if not os.path.exists(file_path):
            raise FileNotFoundError()

        for _, r in pd.read_csv(file_path).iterrows():
            _name = r["토큰"]
            _label = r["tag_change"]

            _ifc_dict.update({_name: _label})
        # end : for (ifc_tags)
        return _ifc_dict

    # end : public static dict labeled_tags_dict()

    @staticmethod
    def get_model_pickle_path(
        set_no: int,
    ) -> str:
        return str.format(
            "{0}/{1}/{2}.bin",
            RESOURCE_DIR_PATH,
            CorrelationModel.__name__,
            set_no,
        )

    # end : public static str get_model_pickle_path()

    @staticmethod
    def get_estimator_pickle_path(
        set_no: int,
    ) -> str:
        return str.format(
            "{0}/{1}/{2}.bin",
            RESOURCE_DIR_PATH,
            AdjustedBiasedCorrelationEstimator.__name__,
            set_no,
        )

    # end : public static str get_estimator_pickle_path()


# end : class


if __name__ == "__main__":
    _extractor = ObservTagsScore()
    KFOLD_SET_NO = 4
    _model = _extractor.read_model(
        set_no=KFOLD_SET_NO,
    )
    _estimator = _extractor.read_estimator(
        set_no=KFOLD_SET_NO,
    )
    ifc_tags_dict: dict = _extractor.labeled_tags_dict()

    _tag_names_list = []
    lname_to_idx_dict = dict()
    fname_to_idx_dict = dict()
    cname_to_idx_dict = dict()

    for tag_name in _estimator.tag_name_to_idx.keys():
        _tag_names_list.append(tag_name)
        if not tag_name in ifc_tags_dict:
            continue
        _idx = _estimator.tag_name_to_idx[tag_name]
        _class = ifc_tags_dict[tag_name]
        if _class == "i":
            lname_to_idx_dict.update({tag_name: _idx})
        elif _class == "c":
            cname_to_idx_dict.update({tag_name: _idx})
        else:
            fname_to_idx_dict.update({tag_name: _idx})
    # end : for (tags)
    __T = len(_tag_names_list)
    __L = len(lname_to_idx_dict)
    __C = len(cname_to_idx_dict)
    __F = len(fname_to_idx_dict)

    """
    file_path = f"{RESOURCE_DIR_PATH}/tags_list.csv"
    with open(file=file_path, mode="wt", encoding="utf-8") as fout:
        for tag_name, tag_idx in _estimator.tag_name_to_idx.items():
            fout.write(f"{tag_idx},{tag_name}\n")
        # end : for (filtered_tags)
        fout.close()
    # end : StreamWriter()
    exit(0)
    """

    plt.rcParams["font.family"] = "AppleGothic"
    mpl.rcParams["axes.unicode_minus"] = False

    # plt.title(label="Tags scores (CorrelationModel)", fontsize=8.0)
    # _tags_score: np.ndarray = _model.arr_tags_score

    _tags_score: np.ndarray = _estimator.arr_tags_score
    plt.title(label="Tags scores (Estimator)", fontsize=8.0)

    lc_scores_arr = np.zeros(shape=(__L, __C))
    i = j = 0
    for x_name, x_idx in lname_to_idx_dict.items():
        for y_name, y_idx in cname_to_idx_dict.items():
            lc_scores_arr[i][j] = _tags_score[x_idx][y_idx]
            j += 1
        # end : for (|C|)
        i += 1
        j = 0
    # end : for (|L|)

    _min = np.min(lc_scores_arr)
    _max = np.max(lc_scores_arr)

    lname_list = list(lname_to_idx_dict.keys())
    cname_list = list(cname_to_idx_dict.keys())

    _file_path = str.format(
        "{0}/corr_{1}.csv",
        RESOURCE_DIR_PATH,
        KFOLD_SET_NO,
    )
    with open(file=_file_path, mode="wt") as fout:
        ## HEADER
        fout.write(",")
        for _cname in cname_list:
            fout.write(f"{_cname},")
        fout.write("\n")

        ## scores
        for x_idx in range(__L):
            fout.write(f"{lname_list[x_idx]},")
            for y_idx in range(__C):
                fout.write(f"{lc_scores_arr[x_idx][y_idx]},")
            # end : for (|C|)
            fout.write("\n")
        # end : for (|L|)
        fout.close()
    # end : StramWriter()

    # _max = np.max(lc_scores_arr[lc_scores_arr < 1.0])

    # plt.rc("legend", fontsize="small")
    # plt.rc("xtick", labelsize=3)
    # plt.rc("ytick", labelsize=3)
    # plt.rc("font.size")

    # """
    plt.xlabel(xlabel="Source tags name", fontsize=4.0)
    plt.ylabel(ylabel="Target tags name", fontsize=4.0)
    plt.xticks(fontsize=2.0)
    plt.yticks(fontsize=2.0)
    ax = sns.heatmap(
        np.transpose(lc_scores_arr),
        vmin=_min,
        vmax=_max,
        center=np.median(
            lc_scores_arr[(lc_scores_arr != 1.0) & (lc_scores_arr != 0.0)]
        ),
        cmap="Grays",
        # cmap="RdBu",
        xticklabels=lname_list,
        yticklabels=cname_list,
    )
    plt.show()
    # """

    """
    ax = sns.clustermap(
        lc_scores_arr,
        vmin=_min,
        vmax=_max,
        cmap="Grays",
        xticklabels=lname_list,
        yticklabels=cname_list,
        ## defaults
        # cbar_kws=dict(use_gridspec=False, location="top"),
        # cbar_pos=(0.02, 0.8, 0.05, 0.18),
        # cbar_kws=dict(use_gridspec=False, location="top"),
        # cbar_pos=(0.03, 0.85, 0.1, 0.01),
        ## (pos_x, pos_y, len_x, len_y)
    )
    ax.tick_params(axis="x", labelsize=2.0)
    ax.tick_params(axis="y", labelsize=2.0)
    plt.show()
    """
# end : main()
