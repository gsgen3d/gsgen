# learning rate schedulers
import torch
import numpy as np


def exp_decay(tot_steps, lr_start, lr_end, warmup_steps=0, warmup_type="linear"):
    def _decay(step):
        if step < warmup_steps:
            if warmup_type == "linear":
                return lr_start * (step / warmup_steps)
        else:
            t = np.clip(
                (step - warmup_steps) / (tot_steps - warmup_steps),
                0,
                1,
            )
            return np.exp(np.log(lr_start) * (1 - t) + np.log(lr_end) * t)

    return _decay


def cosine_decay(tot_steps, lr_start, lr_end, warmup_steps=0, warmup_type="linear"):
    def _decay(step):
        if step < warmup_steps:
            return lr_start * (step / warmup_steps)
        else:
            progress = (step - warmup_steps) / (tot_steps - warmup_steps)
            return lr_end + (lr_start - lr_end) * (1 + np.cos(np.pi * progress)) / 2

    return _decay


def no_decay(tot_steps, lr_start, lr_end, warmup_steps=0, warmup_type="linear"):
    return lambda x: lr_start


lr_schedulers = dict(
    nothing=no_decay,
    cosine=cosine_decay,
    exp=exp_decay,
)
if __name__ == "__main__":
    # test code
    import matplotlib.pyplot as plt

    tot_steps = 10000
    warmup_steps = 1000
    lr_start = 1e-4
    lr_end = 1e-6
    fig, ax = plt.subplots()
    for key in lr_schedulers:
        fn = lr_schedulers[key](tot_steps, lr_start, lr_end, warmup_steps=warmup_steps)
        xs = []
        ys = []
        for i in range(tot_steps):
            xs.append(i)
            ys.append(fn(i))

        ax.plot(xs, ys, label=key)

    ax.legend()

    fig.savefig("./tmp/debug/lr_schedulers.png", bbox_inches="tight")
