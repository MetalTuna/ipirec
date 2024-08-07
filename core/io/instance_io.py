import os
import pickle
from configparser import ConfigParser
from .dir_path_validator import DirectoryPathValidator


class InstanceIO:

    @staticmethod
    def dump_inst(
        obj,
        file_path: str,
    ) -> None:
        """인스턴스 저장"""
        _dir_path = os.path.dirname(file_path)
        if not os.path.exists(_dir_path):
            DirectoryPathValidator.mkdir(_dir_path)
        with open(file_path, "wb") as fout:
            pickle.dump(obj, fout)
            fout.close()
        # end : StreamWriter()

    # end : public static void dump_inst()

    @staticmethod
    def dump_kfold_bin_inst(
        obj,
        dump_dir_path: str,
        config_info: ConfigParser,
    ) -> None:
        _file_path = str.format(
            "{0}/{1}/{2}.bin",
            dump_dir_path,
            type(obj).__name__,
            config_info["DataSet"].get("kfold_set_no", ""),
        )
        InstanceIO.dump_inst(obj, _file_path)

    # end : public static void dump_kfold_bin_inst()

    @staticmethod
    def load_bin_file_inst(
        _inst_obj,
        file_path: str,
    ):
        """인스턴스 적재"""
        _obj = None
        _inst_kwd = _inst_obj.__name__
        if not _inst_kwd in file_path:
            print(f"{file_path} is not {_inst_kwd} instance.")
            raise TypeError()
        if not os.path.exists(file_path):
            raise FileNotFoundError()
        with open(file_path, "rb") as fin:
            _obj = pickle.load(fin)
            fin.close()
        # end : StreamReader()
        return _obj

    # end : public void load()


# end : class
