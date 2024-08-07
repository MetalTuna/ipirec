# Built-in
import os
import sys
import gc
import copy
import pickle

# Custom Lib.
from core import *
from .ipirec_approx_estimator import IPIRecApproxEstimator


class IPIRecApproxEstimatorRev(IPIRecApproxEstimator):

    def train(
        self,
        fit_dtype_seq: list = [
            DecisionType.E_VIEW,
            DecisionType.E_LIKE,
            DecisionType.E_PURCHASE,
        ],
        post_reg_dtype_seq: list = [],
        BIN_DUMP_DIR_PATH: str = None,
    ) -> None:
        _oracle_estimator: IPIRecApproxEstimatorRev = None
        if BIN_DUMP_DIR_PATH != None:
            _proc_kwd = DecisionType.list_to_kwd_str(fit_dtype_seq)
            _post_kwd = DecisionType.list_to_kwd_str(post_reg_dtype_seq)
            _prev_file_path = str.format(
                "{0}/{1}/{2}_{3}",
                BIN_DUMP_DIR_PATH,
                IPIRecApproxEstimatorRev.__name__,
                self._KFOLD_NO,
                _proc_kwd,
            )
            if _post_kwd != "":
                _prev_file_path += f"_{_post_kwd}"
            _prev_file_path += ".bin"

            if os.path.exists(_prev_file_path):
                with open(_prev_file_path, "rb") as fin:
                    self: IPIRecApproxEstimatorRev = pickle.load(fin)
                    fin.close()
                # end : StreamReader()
                return
            if not DirectoryPathValidator.exist_dir(BIN_DUMP_DIR_PATH):
                DirectoryPathValidator.mkdir(BIN_DUMP_DIR_PATH)
        # end : if BIN_DUMP_DIR_PATH != None

        # harmonic loss
        _HL = list()

        # 전처리
        ## 태그점수 보정 (DM -> ML)
        _ls = self.__fit_scores__(DecisionType.E_VIEW)
        ## 태그 축 계산 (DENSE)
        self.__append_biases__()
        ## 태그점수 보정 (GEN)
        _ls = self.__fit_scores__(DecisionType.E_VIEW)

        if BIN_DUMP_DIR_PATH != None:
            file_path = str.format(
                "{0}/{1}/{2}.bin",
                BIN_DUMP_DIR_PATH,
                IPIRecApproxEstimatorRev.__name__,
                self._KFOLD_NO,
            )
            with open(file_path, "wb") as fout:
                pickle.dump(self, fout)
                fout.close()
            # end : StreamWriter()

        # 주처리
        _proc_dtype_seq_list = list()
        _post_dtype_seq_list = list()
        ## 의사결정 타입별 순차 보정
        for decision_type in fit_dtype_seq:
            _proc_dtype_seq_list.append(decision_type)
            _proc_kwd = DecisionType.list_to_kwd_str(_proc_dtype_seq_list)

            ## LOAD_BIN
            if BIN_DUMP_DIR_PATH != None:
                file_path = str.format(
                    "{0}/{1}/{2}_{3}.bin",
                    BIN_DUMP_DIR_PATH,
                    IPIRecApproxEstimatorRev.__name__,
                    self._KFOLD_NO,
                    _proc_kwd,
                )
                if os.path.exists(file_path):
                    with open(file_path, "rb") as fin:
                        self: IPIRecApproxEstimatorRev = pickle.load(fin)
                        fin.close()
                    # end : StreamReader()
                    continue
            # end : if

            _min_hl = sys.float_info.max
            _HL.append(_min_hl)

            ## training
            while True:
                ### 개인화 보정
                _lw = self.__fit_weights__(decision_type)
                ### 태그점수 보정
                _ls = self.__fit_scores__(decision_type)
                _ls = self.__append_biases__()
                _ls = self.__fit_scores__(decision_type)

                _hl = (2 * _ls * _lw) / (_ls + _lw)
                _min_hl = min(_HL)
                if _min_hl > _hl:
                    if _oracle_estimator != None:
                        self: IPIRecApproxEstimatorRev = _oracle_estimator
                    break
                if _min_hl < _hl:
                    _oracle_estimator: IPIRecApproxEstimatorRev = copy.deepcopy(self)
                _HL.append(_hl)
            # end : while (!is_fit)

            ## DUMP_BIN
            if BIN_DUMP_DIR_PATH != None:
                file_path = str.format(
                    "{0}/{1}/{2}_{3}.bin",
                    BIN_DUMP_DIR_PATH,
                    IPIRecApproxEstimatorRev.__name__,
                    self._KFOLD_NO,
                    _proc_kwd,
                )
                self._config_info.set("Estimator", "train_seq", _proc_kwd)
                with open(file_path, "wb") as fout:
                    pickle.dump(self, fout)
                    fout.close()
                # end : StreamWriter()
                with open(file_path.replace("bin", "ini"), "wt") as fout:
                    self._config_info.write(fout)
                    fout.close()
                # end : StreamWriter()
            # end : if
        # end : for (decision_types)
        _HL.clear()
        gc.collect()

        # 후처리
        for decision_type in post_reg_dtype_seq:
            _post_dtype_seq_list.append(decision_type)
            _post_kwd = DecisionType.list_to_kwd_str(_post_dtype_seq_list)
            ## LOAD_BIN
            if BIN_DUMP_DIR_PATH != None:
                file_path = str.format(
                    "{0}/{1}/{2}_{3}_{4}.bin",
                    BIN_DUMP_DIR_PATH,
                    IPIRecApproxEstimatorRev.__name__,
                    self._KFOLD_NO,
                    _proc_kwd,
                    _post_kwd,
                )
                if os.path.exists(file_path):
                    with open(file_path, "rb") as fin:
                        self: IPIRecApproxEstimatorRev = pickle.load(fin)
                        fin.close()
                    # end : StreamReader()
                    continue
            # end : if

            if decision_type != DecisionType.E_VIEW:
                self.__fit_scores__(DecisionType.E_VIEW)
            _ls = self.__fit_scores__(decision_type)

            ## DUMP_BIN
            if BIN_DUMP_DIR_PATH != None:
                file_path = str.format(
                    "{0}/{1}/{2}_{3}_{4}.bin",
                    BIN_DUMP_DIR_PATH,
                    IPIRecApproxEstimatorRev.__name__,
                    self._KFOLD_NO,
                    _proc_kwd,
                    _post_kwd,
                )
                self._config_info.set("Estimator", "train_seq", _proc_kwd)
                self._config_info.set("Estimator", "post_train_seq", _post_kwd)
                with open(file_path, "wb") as fout:
                    pickle.dump(self, fout)
                    fout.close()
                # end : StreamWriter()
                with open(file_path.replace("bin", "ini"), "wt") as fout:
                    self._config_info.write(fout)
                    fout.close()
                # end : StreamWriter()
            # end : if
        # end : for (post_decision_types)

    # public override void train()


# end : class
