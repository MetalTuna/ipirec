## build-in
import os
import sys


# [DEF] Environment variables
__dir_name__ = os.path.basename(os.path.dirname(__file__))
WORKSPACE_HOME = os.path.dirname(__file__).replace(f"/experiments/{__dir_name__}", "")
""".../ipirec"""

# [SET] Env.append($WORKSPACE_HOME)
sys.path.append(WORKSPACE_HOME)

## custom modules
# from .nmf_aggr import NMFResultAggregator
from nmf_aggr import NMFResultAggregator


class IPIRecResultAggregator(NMFResultAggregator):

    def __init__(self) -> None:
        super().__init__()
        self._model_name_ = "BiasedIPIRecA"

        # model_params dict만 변경하면 됨
        self.params_dict = {
            # data info
            "DataSet": set(),
            "Dtype": set(),
            # tags score
            "top_n_tags": set(),
            "co_occur_items": set(),
            "score_iterations": set(),
            "score_learning_rate": set(),
            "score_generanlization": set(),
            # estimator
            "weight_learning_rate": set(),
            "weight_generalization": set(),
            "weight_iterations": set(),
            "default_voting": set(),
            "FrobNorm": set(),
        }
        """
        - Summary: Conditions_dictionary
        - Elements:
            - Key: 
                - Data
                    - DataSet
                    - Dtype
                - Model
                    - top_n_tags
                    - co_occur_items
                    - score_iterations
                    - score_learning_rate
                - Estimator
                    - weight_learning_rate
                    - weight_generalization
                    - weight_iterations
                    - default_voting
                    - FrobNorm
            - Value: params_set (set)
        """

    # end : init()

    """
    def __dump_results__(self, file_path: str) -> None:
        # Worksheet에 model_params 출력만 변경하면 됨
        raise NotImplementedError()
        return super().__dump_results__(file_path)
    """

    def __parse_result__(
        self,
        lines: list,
        filtering: bool,
    ) -> list:
        parsed_list = list()

        # DATA
        split_ch = ","
        dataset = lines[1].split(split_ch)[1].strip()
        decision = lines[1].split(split_ch)[2].strip()
        set_id = int(lines[2].split(split_ch)[1])
        self.params_dict["DataSet"].add(dataset)
        self.params_dict["Dtype"].add(decision)

        # MODEL
        split_ch = ":"
        top_n_tags = int(lines[4].split(split_ch)[1].strip())
        co_occur_items = int(lines[5].split(split_ch)[1].strip())
        score_iterations = int(lines[6].split(split_ch)[1].strip())
        score_learning_rate = float(lines[7].split(split_ch)[1].strip())
        self.params_dict["top_n_tags"].add(top_n_tags)
        self.params_dict["co_occur_items"].add(co_occur_items)
        self.params_dict["score_iterations"].add(score_iterations)
        self.params_dict["score_learning_rate"].add(score_learning_rate)
        score_generalization = float(lines[14].split(split_ch)[1].strip())
        self.params_dict["score_generanlization"].add(score_generalization)

        # ESTIMATOR
        weight_learning_rate = float(lines[9].split(split_ch)[1].strip())
        weight_generalization = float(lines[10].split(split_ch)[1].strip())
        weight_iterations = int(lines[11].split(split_ch)[1].strip())
        frob_norm = int(lines[12].split(split_ch)[1].strip())
        default_voting = float(lines[13].split(split_ch)[1].strip())
        self.params_dict["weight_learning_rate"].add(weight_learning_rate)
        self.params_dict["weight_generalization"].add(weight_generalization)
        self.params_dict["weight_iterations"].add(weight_iterations)
        self.params_dict["default_voting"].add(default_voting)
        self.params_dict["FrobNorm"].add(frob_norm)

        ## [HEADER] ln. 14
        split_ch = ","
        # Sheet
        if len(self.properties_key_list) < 2:
            metrics_properties = [
                l.strip() for l in lines[15].split(split_ch) if l != ""
            ]
            self.properties_key_list.extend(metrics_properties)
            self.properties_key_list[1] = "Top-N"
            ## Fixed Prop. Seq.
            self.properties_key_list.append("DataSet")
            self.properties_key_list.append("Dtype")
            self.properties_key_list.append("top_n_tags")
            self.properties_key_list.append("co_occur_items")
            self.properties_key_list.append("score_iterations")
            self.properties_key_list.append("score_learning_rate")
            self.properties_key_list.append("score_generanlization")
            self.properties_key_list.append("weight_learning_rate")
            self.properties_key_list.append("weight_generalization")
            self.properties_key_list.append("weight_iterations")
            self.properties_key_list.append("default_voting")
            self.properties_key_list.append("FrobNorm")
        # end : if

        ## [Results] ln. 15 -- 32 >> 16 -- 33"
        # for idx in range(15, len(lines)):
        for idx in range(16, len(lines)):
            result_list = list()

            values = [float(v) for v in lines[idx].split(split_ch)]
            values.pop(0)
            if (float(values[3]) == 0.0) and filtering:
                continue
            # end : if (results_filtering)

            result_list.append(set_id)
            result_list.extend(values)
            result_list.append(dataset)
            result_list.append(decision)
            result_list.append(top_n_tags)
            result_list.append(co_occur_items)
            result_list.append(score_iterations)
            result_list.append(score_learning_rate)
            result_list.append(score_generalization)
            result_list.append(weight_learning_rate)
            result_list.append(weight_generalization)
            result_list.append(weight_iterations)
            result_list.append(default_voting)
            result_list.append(frob_norm)

            parsed_list.append(result_list)
        # end : for (results)

        return parsed_list

    # end : private list parse_result()


# end : class


if __name__ == "__main__":
    _datestr = "20240605"
    """YYYYMMDD"""
    cpy_time_str = "1717"
    """HHMM"""
    train_dtype_seq_str = "vlp"
    # machine_name_str = "3090"
    machine_name_str = "SubMachine"

    # module_name_str = "Pinpoint_Debug"
    # module_name_str = "IPIRecAB_Scores"
    # module_name_str = "IPIRecAB_TopNTags"
    # module_name_str = "IPIRecAB_Personalization"

    ## pinpoin
    # module_name_str = "Scores"
    # module_name_str = "TopNTags"
    module_name_str = "Personalization"

    results_home_path = str.format(
        "{0}/results/{1}/{2}",  # {3}_{4}/{5}",
        WORKSPACE_HOME,
        _datestr,
        cpy_time_str,
        # train_dtype_seq_str,
        # machine_name_str,
        # module_name_str,
    )

    RESULTS_DIR_PATH = str.format(
        "{0}/{1}_{2}",
        results_home_path,
        train_dtype_seq_str,
        machine_name_str,
    )
    result_files_dir_path = str.format(
        "{0}/{1}",
        RESULTS_DIR_PATH,
        module_name_str,
    )
    workbook_file_path = str.format(
        "{0}/{1}_summary_{2}.xlsx",
        RESULTS_DIR_PATH,
        module_name_str,
        machine_name_str,
    )
    inst = IPIRecResultAggregator()
    inst.aggregation(
        results_dir_path=result_files_dir_path,
        save_workbook_path=workbook_file_path,
        filtering=True,
    )

# end : main()
