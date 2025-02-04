import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

rng = np.random.default_rng()

coef = 0.2, 0.5, -1., -0.4, 0.3, 0.1
coefl = len(coef)
num = 50
batch_size = 25
learning_rate = 0.001


def cost(a, y):
    return 0.5 * np.mean((a - y) ** 2)


def dcost(a, y):
    return (a - y)


def forward(a, coef):
    return sum(k * a ** i for i, k in enumerate(coef))


def backward(da, *, a_prev, coef, m, v):
    dcoef = np.array([np.mean(da * a_prev ** i) for i in range(coefl)])

    m *= 0.9; m += (1. - 0.9) * dcoef
    v *= 0.9; v += (1. - 0.9) * dcoef ** 2

    coef -= learning_rate * m / (np.sqrt(v) + 1e-7)


def frames():
    coef, m, v = rng.random(coefl) - 0.5, np.zeros(coefl), np.zeros(coefl)
    a = np.zeros(num)

    while cost(a, y) > 1e-7:
        for i in range(0, num, batch_size):
            bx = x[i : i + batch_size]
            by = y[i : i + batch_size]
            ba = a[i : i + batch_size] = forward(bx, coef)
            backward(dcost(ba, by), a_prev=bx, coef=coef, m=m, v=v)
        yield a


def func(frame):
    line.set_ydata(frame)
    return lines


x = np.linspace(-2, 2, num=num)
y = forward(x, coef)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.set(ylim=(-2, 2), xmargin=0)

ax.plot(x, y, lw=1, ls='--', color='indianred')
line, = lines = ax.plot(x, y)

ani = animation.FuncAnimation(fig, func, frames, cache_frame_data=False, interval=10, blit=True)

plt.tight_layout()
plt.show()
