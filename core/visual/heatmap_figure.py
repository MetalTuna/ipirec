from abc import *

from pandas import DataFrame
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


class HeatmapFigure:
    def __init__(self) -> None:
        pass

    @staticmethod
    def draw_heatmap(
        tag_idx_to_name_dict: dict,
        tags_corr: np.ndarray,
        fig_title: str = "",
        file_path: str = "",
        diag_score: float = 0.0,
    ) -> plt:
        plt.clf()
        plt.rcParams["font.family"] = "AppleGothic"
        # plt.rcParams["font.family"] = "Times"
        mpl.rcParams["axes.unicode_minus"] = False
        # plt.rcParams["figure.figsize"] = (200, 4)

        tags_count = len(tag_idx_to_name_dict)
        tags_list = list()
        corr_list_dict = list()
        for x_idx in range(tags_count):
            x_name = tag_idx_to_name_dict[x_idx]
            tags_list.append(x_name)
            corr_list_dict.append({x_name: list()})
        # end : for (tags)

        df = DataFrame(corr_list_dict, index=tags_list)
        for x_idx in range(tags_count):
            corr_list = list()
            for y_idx in range(tags_count):
                corr: float = (
                    diag_score if x_idx == y_idx else float(tags_corr[x_idx][y_idx])
                )
                corr_list.append(corr)
            # end : for (tag_y)
            df[tag_idx_to_name_dict[x_idx]] = corr_list
        # end : for (tag_x)

        # plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
        # plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
        # plt.pcolor(df, vmin=-1.0, vmax=1.0)
        # plt.pcolor(df)
        # plt.pcolor(df, cmap="Greys", vmin=-1.0, vmax=1.0)
        plt.pcolor(df, cmap="Greys")
        fig_title = "Tags score" if fig_title == "" else fig_title
        plt.title(fig_title, fontsize=12)
        plt.xlabel("Src. tags", fontsize=10)
        plt.ylabel("Dest. tags", fontsize=10)
        plt.xticks(
            np.arange(0, len(df.columns), 1),
            labels=df.columns,
            fontsize=1.5,
            rotation=45,
        )
        plt.yticks(
            np.arange(0, len(df.index), 1),
            labels=df.index,
            fontsize=1.5,
        )
        plt.colorbar()

        if file_path != "":
            plt.savefig(file_path)

    # end : public static matplotlib.pyplot draw_heatmap()
