import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np

v = 1.7
noise = 0.15
dt = 0.01
n_steps=10000

bins = np.arange(0, 1.5, 0.05)
rt_yes = []
rt_no = []

fig, (ax_hist_yes, ax_plot, ax_hist_no) = plt.subplots(nrows=3, figsize=(6, 6), sharex=True, height_ratios=[0.1, 1, 0.1], layout="constrained")
ax_plot.set_xlim(0, 1.4)
ax_plot.set_ylim(-1.2, 1.2)
ax_plot.axhline(1, color="black")
ax_plot.axhline(-1, color="black")
line = ax_plot.plot([0], [0], color="black")[0]
hist_vals_yes, _, patches_yes = ax_hist_yes.hist(rt_yes, bins=bins, color="C0")
hist_vals_no, _, patches_no = ax_hist_no.hist(rt_no, bins=bins, color="C1")
ax_hist_yes.set_ylim(0, 10)
ax_hist_no.set_ylim(0, 10)
ax_hist_no.set_xlabel("reaction time (seconds)")

ax_hist_yes.set_ylabel("YES")
ax_hist_no.set_ylabel("NO")
last_i = 0
to_return = list(patches_yes) + list(patches_no) + [line]


def anim(i):
    global line
    global last_i
    time = (i - last_i) * dt
    xdata, ydata = line.get_data()
    last_y = ydata[-1]
    if last_y > 1 or last_y < -1 or time > 1.4:
        if last_y > 1:
            rt_yes.append(time)
            color = "C0"
        elif last_y < -1:
            rt_no.append(time)
            color = "C1"
        else:
            color = "C2"
        line.set_color(color)
        line.set_alpha(0.2)
        line = ax_plot.plot([0], [0], color="black")[0]
        xdata, ydata = [0], [0]
        last_y = ydata[-1]
        last_i = i
        time = 0
        to_return.append(line)

        new_hist_vals_yes, _ = np.histogram(rt_yes, bins)
        new_hist_vals_no, _ = np.histogram(rt_no, bins)
        for patch, new_height in zip(patches_yes, new_hist_vals_yes):
            patch.set_height(new_height)
        for patch, new_height in zip(patches_no, new_hist_vals_no):
            patch.set_height(new_height)

    xdata = np.append(xdata, [time])
    if time < 0.2:
        ydata = np.append(ydata, last_y + 0.1 * noise * np.random.randn(1))
    else:
        ydata = np.append(ydata, last_y + dt * v + noise * np.random.randn(1))
    line.set_data(xdata, ydata)
    if (i + 1) % 1000 == 0:
        print(f"{i}/10000")

    return to_return

anim = matplotlib.animation.FuncAnimation(
    fig, anim, frames=n_steps, interval=1, blit=True
)
anim.save('histogram_animation.mp4', writer='ffmpeg', fps=60)
