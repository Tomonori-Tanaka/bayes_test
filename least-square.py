import numpy as np
from matplotlib import pyplot as plt

x_train = np.array([])
y_train = np.array([])
with open("ofm_eform_summary.txt", encoding="utf-8") as f:
    for line in f:
        x_train = np.append(x_train, float(line.split()[0]))
        y_train = np.append(y_train, float(line.split()[1]))

coe = np.polyfit(x_train, y_train, 2)
print(coe)

x = np.arange(0, 5, 0.1)
y_fit = coe[0] * x**2 + coe[1] * x + coe[2]
# y_fit = coe[0] * x ** 3 + coe[1] * x ** 2 + coe[2] * x + coe[3]
# y_fit = coe[0] * x ** 4 + coe[1] * x ** 3 + coe[2] * x ** 2 + coe[3] * x + coe[4]
# y_fit = coe[0] * x**5 + coe[1] * x**4 + coe[2] *x**3 + coe[3] * x**2 + coe[4] * x + coe[5]

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(x_train, y_train, label='sample', lw=1, marker="o")
ax1.plot(x, y_fit, label='fitted curve', lw=1)

plt.xlim(2.5, 5)
plt.ylim(0, 0.6)
plt.show()
plt.close()
