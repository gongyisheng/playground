import re
import pandas as pd
from itertools import product
import torch
import datetime

from config import CFG
from pipeline import run_training_pipeline


class EXP:
    def __init__(self, cfg: CFG):
        self.cfg = cfg
        self.default_params = {
            k: v
            for k, v in vars(cfg).items()
            if (not "__" in k) and (not re.search("^_", k))
        }
        self.exp_param_names = set()  # names of params that are changed in exps
        self.df = None  # df (rows as exps) with all params
        self.grid = None  # df (rows are exps) with just params that are changed
        self.n_exp = 0

    def to_cfgs(self) -> list:
        """Convert each row in self.df to a cfg object"""
        cfg_list = []
        for config_dict in self.list_of_dicts(self.df.to_dict("list")):
            cfg_list.append(self.create_cfg(config_dict))
        return cfg_list

    def create_cfg(self, config_dict):
        """Convert a dict of params to a cfg object"""
        cfg = CFG()
        for param in config_dict.keys():
            setattr(cfg, param, config_dict[param])
        return cfg

    def add_exp(self, exp):
        """Add experiment(s) from dataframe, dict of param values, or dict of lists of param values. Clean up experiment param df and set experiment grid df"""
        if isinstance(exp, pd.DataFrame):
            exp = exp.to_dict("list")
            self.add_many_exp(exp)
        elif isinstance(exp, dict):
            if isinstance(list(exp.values())[0], list):
                self.add_many_exp(exp)
            else:
                self.add_one_exp(exp)
        else:
            raise ValueError("experiment must be a dict or a df")
        self.reduce_df()
        self.set_grid()

    def add_one_exp(self, exp: dict):
        """Add one experiment from a dict of param values to be changed"""
        self._validate_exp_params(exp)
        self.exp_param_names.update(exp.keys())
        exp = pd.DataFrame(
            {
                k: [self.default_params[k]] if k not in exp.keys() else [exp[k]]
                for k in self.default_params.keys()
            }
        )
        if self.n_exp == 0:
            self.df = exp.copy()
        else:
            self.df = pd.concat([self.df, exp], axis=0, ignore_index=True)
        self.n_exp += 1

    def add_many_exp(self, exp):
        """Convert dict of lists of param values into list of dicts of single param values.
        If len(value_list) is constant for all params, treat each index as individual exp.
        If not, take cartesian product of all param values.
        Then, iterate through list of experiment dicts."""
        if len(set([len(param_list) for param_list in exp.values()])) == 1:
            exp_list = self.list_of_dicts(exp)
        else:
            exp_list = self.product_of_dicts(exp)
        for exp in exp_list:
            self.add_one_exp(exp)

    def _validate_exp_params(self, exp: dict):
        """Validate dict of exp params by using validation logic in default cfg object"""
        for k, v in exp.items():
            if not self.cfg._validate(k, v):
                raise ValueError(f"Invalid parameter: {k}={v}")

    def reduce_df(self):
        """Drop duplicates from df, keeping first entry and ignoring index.
        This is necessary since drop_duplicates() method does not work w
        fields w list values. Then update n_exp."""
        self.df = self.df.loc[self.df.astype(str).drop_duplicates().index].reset_index(
            drop=True
        )
        self.n_exp = self.df.shape[0]

    def set_grid(self):
        """Create experiment grid df by subsetting experiment full df to fields that have been changed"""
        self.grid = self.df[list(self.exp_param_names)]

    def list_of_dicts(self, d):
        """Convert dict of lists to list of dicts"""
        keys = d.keys()
        vals = zip(*[d[k] for k in keys])
        return [dict(zip(keys, v)) for v in vals]

    def product_of_dicts(self, d):
        """Convert dict of lists to list of dicts resulting from taking cartesian product of list values"""
        return [dict(zip(d.keys(), values)) for values in product(*d.values())]

    def __repr__(self):
        return str(self.grid)


def run_training_experiments(df_or_df_list, cfg_or_exp) -> pd.DataFrame:
    results = []
    start_dt = datetime.datetime.utcnow()
    if isinstance(df_or_df_list, list) and isinstance(cfg_or_exp, CFG):
        for i, df in enumerate(df_or_df_list):
            print(f"============= EXP {i+1}/{len(df_or_df_list)} ==============")
            res = run_training_pipeline(df, cfg_or_exp)
            # only keeping track of logs here
            log = []
            for i in range(len(res)):
                log.append(res[i]["log"])
            log = pd.concat(log, axis=0, ignore_index=True)
            results.append(log)
            print("Emptying cache...")
            torch.cuda.empty_cache()
            print("====================================")
    elif isinstance(df_or_df_list, pd.DataFrame) and isinstance(cfg_or_exp, EXP):
        print_exp_grid(cfg_or_exp, start_dt)
        for i, cfg in enumerate(cfg_or_exp.to_cfgs()):
            print(f"============= EXP {i+1}/{cfg_or_exp.n_exp} ==============")
            res = run_training_pipeline(df_or_df_list, cfg)
            # only keeping track of logs here
            log = []
            for i in range(len(res)):
                log.append(res[i]["log"])
            log = pd.concat(log, axis=0, ignore_index=True)
            results.append(log)
            print("Emptying cache...")
            torch.cuda.empty_cache()
            print("====================================")
    elif (
        isinstance(df_or_df_list, list)
        and isinstance(cfg_or_exp, EXP)
        and len(df_or_df_list) == cfg_or_exp.n_exp
    ):
        print_exp_grid(cfg_or_exp, start_dt)
        for i, (df, cfg) in enumerate(list(zip(df_or_df_list, cfg_or_exp.to_cfgs()))):
            print(f"============= EXP {i+1}/{cfg_or_exp.n_exp} ==============")
            res = run_training_pipeline(df, cfg)
            # only keeping track of logs here
            log = []
            for i in range(len(res)):
                log.append(res[i]["log"])
            log = pd.concat(log, axis=0, ignore_index=True)
            results.append(log)
            print("Emptying cache...")
            torch.cuda.empty_cache()
            print("====================================")
    else:
        raise ValueError("Invalid inputs")

    # return a single logs df
    results = pd.concat(results, axis=0, ignore_index=True)
    print(
        f"Experiments complete. Time elapsed: {datetime.datetime.utcnow() - start_dt}"
    )
    return results


def print_exp_grid(exp, start_dt):
    print("************************************")
    print("CUSTOM EXPERIMENTATION GRID")
    print(f"dt: {str(start_dt)}")
    print()
    print(exp)
    print("************************************")
