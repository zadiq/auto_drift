import os
import numpy as np
from matplotlib import pyplot as plt
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from drift.commons import Sequence
from drift.robots import ETHModel
from drift.models.ddpg_tensor import Params


class Ensemble:
    """Class for ensembling model actions using different ensemble methods"""

    def __init__(self, base_path, file_names, method="mean", inf_scores=None):
        _ = os.path.join
        self.base_path = base_path
        self.file_names = file_names
        self.method = method
        self.opt_weights = None
        self.opt_count = 0
        self.opt_model = None

        assert method in self._get_methods(), (f"{method} is not a valid method."
                                               f"choose from {self._get_methods()}")

        sequence_paths = [_(base_path, f) for f in file_names]
        self.sequences = [Sequence.load(p) for p in sequence_paths]

        # extract actions
        self.actions = [np.array(s['1_delta']) for s in self.sequences]
        self.actions = np.array(self.actions)

        self.scores = inf_scores or self.extract_scores()

        self.ensemble_action = self.get_ensembled_action()
        # variable used to provide action on call
        self.ensemble_action_serve = self.ensemble_action.tolist()

    @staticmethod
    def _get_methods():
        return [
            'mean', 'weighted_mean'
        ]

    def __call__(self, *args, **kwargs):
        try:
            return self.ensemble_action_serve.pop(0)
        except IndexError:
            raise Exception("No more action to serve, reset to reserve")

    def reset(self):
        self.update()
        self.ensemble_action_serve = self.ensemble_action.tolist()

    def extract_scores(self):
        """
        Extract reward score from file name, negative sign is removed
        therefore lower score is better.
        """
        _ = int
        scores = []
        for f in self.file_names:
            scores.append(_(f.split("-")[3].split(".")[0]))
        return scores

    @property
    def _weighted_mean(self):
        if not self.opt_weights:
            scores = np.array(self.scores)
            re_score = scores.sum() - scores
            weights = re_score / re_score.sum()
        else:
            weights = self.opt_weights
        return np.average(self.actions, axis=0, weights=weights)

    @property
    def _mean(self):
        """ensemble by averaging across each sequence actions"""
        return self.actions.mean(axis=0)

    def get_ensembled_action(self):
        return getattr(self, f"_{self.method}")

    def get_model_names(self):
        names = []
        for n in range(len(self)):
            names.append(f"model-{n}")
        return names

    def plot(self, title=None):
        for i, (ac, sc) in enumerate(zip(self.actions, self.scores)):
            label = f"Model {i}: score={sc}"
            plt.plot(ac, label=label)

        plt.plot(self.ensemble_action, label="Ensemble action")
        plt.legend(loc="Best")
        title = title or f"{self.method} ensemble of {len(self)} model(s)"
        plt.title(title)

    def score(self, model, model_params):
        """Score the ensemble action on a dynamic model"""

        model.reset()
        self.reset()
        dt = 0
        mp = model_params
        inf_reward = []

        while dt < mp.drive_dur:
            action = self()
            _, _, reward, _, _ = model(action, duty_cycle=1)
            inf_reward.append(reward)

            dt += model.variables.dt
            model.sequence(model.variables.group)

        return np.array(inf_reward).sum()

    def update(self):
        """Update ensemble actions"""
        self.ensemble_action = self.get_ensembled_action()

    def optimise_weights(self, model, model_params):
        """
        Find the optimal blending weights for best reward score.
        This only works for weighted means method
        """

        if self.method != "weighted_mean":
            raise AttributeError("optimise_weights method is only available when "
                                 "`weighted_mean method.")
        space = [Real(0, 1, name=n) for n in self.get_model_names()]
        self.opt_count = 0

        @use_named_args(space)
        def optimise(**params):
            self.opt_count += 1
            self.opt_weights = [params[k] for k in self.get_model_names()]
            cost = -1 * self.score(model, model_params)
            print(f"#({self.opt_count}): {cost}, {self.opt_weights}")
            self.update()

            return cost

        self.opt_model = gp_minimize(
            optimise,
            space, n_calls=100, n_jobs=2,
            random_state=43
        )

        # update ensemble actions after optimising weights
        self.opt_weights = self.opt_model.x
        self.update()

    def __len__(self):
        return len(self.sequences)

    def __str__(self):
        return f"Ensemble{len(self), self.method}"

    def __repr__(self):
        return str(self)


def similar_model_analysis(use_inference=False):
    """
    Analysing models with trained with similar control algorithm
    hyper-parameters.
    """
    _ = os.path.join
    base_path = "/home/zadiq/dev/@p/drift_project/data/26-02-2019/ddpg_run_3"
    inf_path = "/home/zadiq/dev/@p/drift_project/data/inference"
    file_names = [
        "eps-38--171.sequence",
        "eps-93--234.sequence",
        "eps-54--247.sequence",
        "eps-42--259.sequence",
        "eps-90--268.sequence"
    ]
    inf_scores = None

    m_params = Params()
    m_params.from_json(_(base_path, 'params.json'))

    if use_inference:
        date = os.path.basename(os.path.dirname(base_path))
        run_num = os.path.basename(base_path)[-1]
        infer_files = []
        for f in file_names:
            f = f.rstrip(".sequence")
            infer_files.append(f"{date}-{run_num}-{f}-inference.sequence")

        base_path = inf_path
        file_names = infer_files

    eth = ETHModel(
        imp="drifty", ds="three",
        reward_version="v1",
        reward_params={'sigma': .9}
    )

    esm = Ensemble(base_path, file_names, method="mean", inf_scores=inf_scores)
    mean_score = esm.score(eth, m_params)
    mean_actions = esm.get_ensembled_action()

    esm = Ensemble(base_path, file_names, method="weighted_mean", inf_scores=inf_scores)
    weighted_mean_score = esm.score(eth, m_params)
    weighted_actions = esm.get_ensembled_action()

    esm.optimise_weights(eth, m_params)
    opt_w_score = esm.score(eth, m_params)
    opt_w_actions = esm.get_ensembled_action()

    print({
        "Mean Score: ": mean_score,
        "Weighted Mean Score: ": weighted_mean_score,
        "Opt Weighted Score: ": opt_w_score,
    })
    plt.title("Plot of blended actions using Mean, Weighted Mean and Optimised Weighted Mean")
    plt.plot(mean_actions, label="Mean")
    plt.plot(weighted_actions, label="Weighted Mean")
    plt.legend()
    plt.plot(opt_w_actions, label="Optimised Weighted Mean")

    return esm


def diverse_model_analysis():
    """Perform model analysis for diverse models"""
    model_sequence_name = [  # list of model inference sequence
        "26-02-2019-3-eps-38--171-inference.sequence",  # 1
        "02-03-2019-1-eps-27--244-inference.sequence",  # 2
        "02-03-2019-2-eps-11--252-inference.sequence",  # 3
        # "02-03-2019-2-eps-11--252-inference.sequence",  # 4
        "02-03-2019-4-eps-108--233-inference.sequence",  # 5
        "02-03-2019-5-eps-27--294-inference.sequence",  # 6
        "02-03-2019-6-eps-17--222-inference.sequence",  # 7
        "02-03-2019-7-eps-42--238-inference.sequence",  # 8
    ]
    inf_scores = [
        222, 184, 238, 200, 287, 175, 182
    ]
    base_path = "/home/zadiq/dev/@p/drift_project/data"
    _ = os.path.join

    def extract_params(sequence_name: str):
        """Extract model params from training directory"""
        info = sequence_name.rstrip("-inference.sequence").rsplit("-", 4)[0]
        m_date, m_run = info.rsplit('-', 1)
        param_path = _(base_path, m_date, f"ddpg_run_{m_run}", "params.json")
        params = Params()
        params.from_json(param_path)
        return params

    scores = {
        "Mean": [],
        "Weighted Mean": [],
        "Opt Weight Mean": [],
    }

    esm = {
        "Mean": [],
        "Weighted Mean": [],
        "Opt Weight Mean": [],
    }

    inf_path = "/home/zadiq/dev/@p/drift_project/data/inference"
    esm_mean = Ensemble(inf_path, model_sequence_name, method="mean", inf_scores=inf_scores)
    esm_weight = Ensemble(inf_path, model_sequence_name, method="weighted_mean", inf_scores=inf_scores)

    for seq in model_sequence_name:
        esm_opt_weight = Ensemble(inf_path, model_sequence_name, method="weighted_mean", inf_scores=inf_scores)
        par = extract_params(seq)
        eth = ETHModel(
            imp=par.dynamics,
            reward_version=par.reward_version,
            ds=par.ds,
            reward_params=par.reward_params
        )
        scores["Mean"].append(str(int(esm_mean.score(eth, par))))
        scores["Weighted Mean"].append(str(int(esm_weight.score(eth, par))))
        esm_opt_weight.optimise_weights(eth, par)
        scores["Opt Weight Mean"].append(str(int(esm_opt_weight.score(eth, par))))
        esm["Opt Weight Mean"].append(esm_opt_weight)

    marker = "\t\t|\t\t"
    title = [str(num) for num in range(len(model_sequence_name))]
    title = "Blend\t\t|\t\t" + "\t\t\t|\t\t".join(title)
    mean_scores = "Mean\t\t|\t\t" + marker.join(scores["Mean"])
    w_mean_scores = "W. Mean\t\t|\t\t" + marker.join(scores["Weighted Mean"])
    o_mean_scores = "O. Mean\t\t|\t\t" + marker.join(scores["Opt Weight Mean"])

    esm["Mean"] = esm_mean
    esm["Weighted Mean"] = esm_weight

    print(title)
    print(mean_scores)
    print(w_mean_scores)
    print(o_mean_scores)

    return esm


if __name__ == "__main__":
    """Run an analysis function here"""

    # ensembler = similar_model_analysis(True)
    d_esm = diverse_model_analysis()
