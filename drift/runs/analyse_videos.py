import pickle
from drift.commons import Animation


if __name__ == '__main__':

    with open('/home/zadiq/dev/@p/auto_drift/drift/notebooks/data/eth_analysis.pkl', 'rb') as f:
        analysis = pickle.load(f)

        model = analysis['02-03-2019/ddpg_run_7'][1][41]
        ani = Animation(model.sequence, show_line=True)
        ani.show()
