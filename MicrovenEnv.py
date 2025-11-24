import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def load_actions_from_npz(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    if isinstance(d, np.lib.npyio.NpzFile):
        actions = d['data'] if 'data' in d else d[list(d.keys())[0]]
        names = list(d['filenames']) if 'filenames' in d else None
        grid_N = int(d['grid_N']) if 'grid_N' in d else None
    else:
        actions = d
        names = None
        grid_N = None
    return actions, names, grid_N   # actions: ndarray shape (M, N, N)

def get_action_array(actions, idx):# actions: ndarray (M, N, N); idx: 0..M-1
    return actions[int(idx)]    # 返回 2D ndarray (N, N)

class MicrowaveEnv:
    def __init__(self, ncol, nrow, actions_path=None):
        self.ncol = ncol
        self.nrow = nrow
        self.temperature = [0.0] * (ncol * nrow)
        self.border_vals = self.border_list()
        self.actions = None
        self.step_count = 0
        self.episoide_count = 0

        beijing_now = datetime.utcnow() + timedelta(hours=8)
        folder_name = beijing_now.strftime(str(self.nrow)+"x"+str(self.ncol)+"_" + str(self.episoide_count) + "_%m%d_%H%M%S")
        self.rec_child_dir = beijing_now.strftime("demo_%m-%d-%H:%M")
        self.record_dir = os.path.join('record', self.rec_child_dir, folder_name)
        # os.makedirs(self.record_dir, exist_ok=True)

        if actions_path and os.path.exists(actions_path):
            self.actions, self.action_names, gridN = load_actions_from_npz(actions_path)
            if self.actions.ndim == 3 and (self.actions.shape[1], self.actions.shape[2]) != (self.nrow, self.ncol):
                print(f'注意:actions shape与环境不符')

    def reset(self):
        self.temperature = [0 for _ in range(self.ncol * self.nrow)]
        self.step_count = 0
        self.episoide_count += 1
        self.border_vals = self.border_list()

        if(self.episoide_count % 50 == 0):
            beijing_now = datetime.utcnow() + timedelta(hours=8)
            folder_name = beijing_now.strftime(str(self.nrow)+"x"+str(self.ncol)+"_" + str(self.episoide_count) + "_%m%d_%H%M%S")
            self.record_dir = os.path.join('record', self.rec_child_dir, folder_name)
            os.makedirs(self.record_dir, exist_ok=True)
        
        # return self.temperature
        return np.array(self.temperature, dtype=np.float32).reshape(self.nrow, self.ncol)

    def step(self, action):
        if self.actions is None:
            raise RuntimeError("actions not loaded; pass actions_path when creating env")
        self.step_count += 1
        a = int(action)
        delta = get_action_array(self.actions, a) 
        T = np.array(self.temperature, dtype=float).reshape((self.nrow, self.ncol))
        T += delta
        self.temperature = T.ravel().tolist()
        self.border_vals = self.border_list()
        if self.episoide_count % 50 == 0 and self.step_count % 20 == 0:
            self.render()
        # 用平均绝对偏差系数 (mean absolute deviation / mean) 作为标准化不一致性度量
        reward = -self.full_grid_cv()
        done = (self.step_count >= 200)
        self.last_action = a
        self.last_reward = float(reward)
        return np.array(self.temperature, dtype=np.float32).reshape(self.nrow, self.ncol), float(reward), bool(done), {}
        
    def render(self):
        T = np.array(self.temperature, dtype=float).reshape((self.nrow, self.ncol))
        plt.figure(figsize=(4, 4))
        plt.imshow(T, origin='lower', cmap='viridis', interpolation='nearest')
        plt.colorbar(fraction=0.046)
        # 标题包含 step、action、reward（若存在）
        title = f'step {self.step_count}'
        if hasattr(self, 'last_action'):
            title += f' | action={self.last_action}'
        if hasattr(self, 'last_reward'):
            title += f' | reward={self.last_reward:.4f}'
        plt.title(title)
        plt.tight_layout()
        os.makedirs(self.record_dir, exist_ok=True)
        fname = f"{int(self.step_count):04d}.png"
        outpath = os.path.join(self.record_dir, fname)
        plt.savefig(outpath, dpi=150)
        plt.close()

    def border_list(self):
        T = np.array(self.temperature, dtype=float).reshape((self.nrow, self.ncol))
        vals = []
        for r in range(0, self.nrow):
            vals.append(float(T[r, 0]))
        for c in range(1, self.ncol):
            vals.append(float(T[self.nrow - 1, c]))
        for r in range(self.nrow - 2, -1, -1):
            vals.append(float(T[r, self.ncol - 1]))
        for c in range(self.ncol - 2, 0, -1):
            vals.append(float(T[0, c]))
        return vals
    def border_array(self):
        return np.asarray(self.border_list(), dtype=float)
    def border_variance(self, ddof=0):
        arr = self.border_array()
        if arr.size == 0:
            return float('nan')
        return float(np.var(arr, ddof=ddof))
    def border_normalized_std(self, ddof=0):
        arr = self.border_array()
        if arr.size == 0:
            return float('nan')
        mean = float(np.mean(arr))
        if mean == 0.0:
            return float('nan')
        std = float(np.std(arr, ddof=ddof))
        return std / mean
    def border_mad_coeff(self):
        arr = self.border_array()
        if arr.size == 0:
            return float('nan')
        mean = float(np.mean(arr))
        if mean == 0.0:
            return float('nan')
        mad = float(np.mean(np.abs(arr - mean)))
        return mad / abs(mean)    
    def border_quantize(self, levels=16, method='linear'):
        arr = self.border_array()
        if arr.size == 0:
            return np.array([], dtype=int)

        if method == 'linear':
            mn, mx = float(arr.min()), float(arr.max())
            if mx == mn:
                return np.zeros_like(arr, dtype=int)
            scaled = (arr - mn) / (mx - mn) * (levels - 1)
            return np.rint(scaled).astype(int)

        if method == 'equalfreq':
            cuts = np.percentile(arr, np.linspace(0, 100, levels + 1))
            bins = cuts[1:-1] if cuts.size > 2 else []
            inds = np.digitize(arr, bins, right=True)
            return inds.astype(int)

        if method == 'kmeans':
            try:
                from sklearn.cluster import KMeans
            except Exception:
                raise RuntimeError("kmeans 方法需要安装 sklearn（pip install scikit-learn）")
            X = arr.reshape(-1, 1)
            kmeans = KMeans(n_clusters=levels, random_state=0, n_init=10).fit(X)
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_.flatten()
            order = np.argsort(centers)
            mapped = np.empty_like(labels)
            for rank, cls in enumerate(order):
                mapped[labels == cls] = rank
            return mapped.astype(int)
        raise ValueError("unknown method: " + str(method))

    def full_grid_cv(self): # (Coefficient of Variation, CV)变异系数
        T_arr = np.array(self.temperature, dtype=float) 
        if T_arr.size == 0:
            return float('nan')
        mean = float(np.mean(T_arr))
        if mean == 0.0: 
            return 0.0 
        std = float(np.std(T_arr, ddof=0))
        return std / abs(mean)

if __name__ == "__main__":
    env = MicrowaveEnv(8, 8, actions_path='data_process/NewDivideResult/8x8/aggregated_8x8.npz')
    env.step(0)    
    env.render()  