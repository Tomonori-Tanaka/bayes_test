from matplotlib import pyplot as plt
import numpy as np

x_train = np.array([])
y_train = np.array([])
with open("ofm_eform_summary.txt", encoding="utf-8") as f:
    for line in f:
        print(line)
        x_train = np.append(x_train, float(line.split()[0]))
        y_train = np.append(y_train, float(line.split()[1]))


import GPy.kern as gp_kern
import GPy
# kern = gp_kern.PeriodicMatern32(input_dim=1) * gp_kern.RBF(input_dim=1)
kern = gp_kern.RBF(input_dim=1) + gp_kern.Bias(input_dim=1)
# kern = gp_kern.PeriodicExponential(input_dim=1)
gpy_model = GPy.models.GPRegression(X=x_train.reshape(-1, 1), Y=y_train.reshape(-1, 1), kernel=kern, normalizer=None)
fig = plt.figure(figsize=(6,8))
ax1 = fig.add_subplot(211)
gpy_model.plot(ax=ax1)  # 最適化前の予測
gpy_model.optimize()

ax2 = fig.add_subplot(212, sharex=ax1)
gpy_model.plot(ax=ax2)  # カーネル最適化後の予測

ax1.set_ylim(ax2.set_ylim(0, 0.6))
ax1.set_title("GPy effect of kernel optimization")
ax1.set_ylabel("Before")
ax2.set_ylabel("After")
fig.tight_layout()
fig.savefig("GPy_kernel_optimization.png", dpi=150)
