import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d


def plot_cpp_p300_analysis(
    c,
    session,
    ID,
    p300_pre,
    p300_post,
    nop300_pre,
    nop300_post,
    reaction_time_pre,
    reaction_time_post,
    start_time_pre,
    start_time_post,
    fs=512,
):
    if session == "tACS":
        pz, cz, c1, c2 = 11, 15, 19, 21
    elif session == "Relaxation":
        pz, cz, cpz, c1, c2 = 26, 15, 20, 19, 21
    else:
        raise ValueError("Session must be 'tACS' or 'Relaxation'")

    time_ms = np.arange(p300_pre.shape[0]) * (1000 / fs)

    if c == 1:
        cpp_pre = np.mean(
            np.mean(
                np.concatenate([nop300_pre[:, c1, :], nop300_pre[:, c2, :]], axis=1),
                axis=1,
            ),
            axis=1,
        )
        stdpre = (
            np.std(
                np.concatenate([nop300_pre[:, c1, :], nop300_pre[:, c2, :]], axis=1),
                axis=1,
            )
            / nop300_pre.shape[2]
        )

        cpp_post = np.mean(
            np.mean(
                np.concatenate([nop300_post[:, c1, :], nop300_post[:, c2, :]], axis=1),
                axis=1,
            ),
            axis=1,
        )
        stdpost = (
            np.std(
                np.concatenate([nop300_post[:, c1, :], nop300_post[:, c2, :]], axis=1),
                axis=1,
            )
            / nop300_post.shape[2]
        )

        os.makedirs("Experiment/CPP", exist_ok=True)
        plt.figure()
        plt.plot(time_ms, uniform_filter1d(cpp_pre, 5), "k", linewidth=2)
        plt.plot(time_ms, uniform_filter1d(cpp_post, 5), "g", linewidth=2)
        plt.plot(time_ms, uniform_filter1d(cpp_pre + stdpre, 5), "k", linewidth=0.5)
        plt.plot(time_ms, uniform_filter1d(cpp_pre - stdpre, 5), "k", linewidth=0.5)
        plt.plot(time_ms, uniform_filter1d(cpp_post + stdpost, 5), "g", linewidth=0.5)
        plt.plot(time_ms, uniform_filter1d(cpp_post - stdpost, 5), "g", linewidth=0.5)
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude (µV)")
        plt.grid(True)
        plt.title("CPP Analysis - Avg CP1/CP2")
        plt.legend(["Pre", "Post"])
        plt.savefig(f"Experiment/CPP/CPP_avg_{ID}.png")
        plt.show()

    elif c == 2:
        pz_pre = np.mean(p300_pre[:, pz, :], axis=1)
        stdpre = np.std(p300_pre[:, pz, :], axis=1) / p300_pre.shape[2]
        pz_post = np.mean(p300_post[:, pz, :], axis=1)
        stdpost = np.std(p300_post[:, pz, :], axis=1) / p300_post.shape[2]

        os.makedirs("Experiment/P300", exist_ok=True)
        plt.figure()
        plt.plot(time_ms, pz_pre, "k", linewidth=2)
        plt.plot(time_ms, pz_post, "g", linewidth=2)
        plt.plot(time_ms, pz_pre + stdpre, "k", linewidth=0.5)
        plt.plot(time_ms, pz_pre - stdpre, "k", linewidth=0.5)
        plt.plot(time_ms, pz_post + stdpost, "g", linewidth=0.5)
        plt.plot(time_ms, pz_post - stdpost, "g", linewidth=0.5)
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude (µV)")
        plt.title("P300 Analysis with Standard Error")
        plt.grid(True)
        plt.legend(["Pre", "Post"])
        plt.savefig(f"Experiment/P300/P300_avg_{ID}.png")
        plt.show()

    elif c == 3:
        os.makedirs("Experiment/Target vs Non-Target Avg", exist_ok=True)

        pz_pre_target = np.mean(p300_pre[:, pz, :], axis=1)
        std_target_pre = np.std(p300_pre[:, pz, :], axis=1) / p300_pre.shape[2]

        pz_pre_nontarget = np.mean(nop300_pre[:, pz, :], axis=1)
        std_nontarget_pre = np.std(nop300_pre[:, pz, :], axis=1) / nop300_pre.shape[2]

        plt.figure()
        plt.plot(time_ms, pz_pre_target, "b", linewidth=2)
        plt.plot(time_ms, pz_pre_nontarget, "r", linewidth=2)
        plt.plot(time_ms, pz_pre_target + std_target_pre, "b", linewidth=0.5)
        plt.plot(time_ms, pz_pre_target - std_target_pre, "b", linewidth=0.5)
        plt.plot(time_ms, pz_pre_nontarget + std_nontarget_pre, "r", linewidth=0.5)
        plt.plot(time_ms, pz_pre_nontarget - std_nontarget_pre, "r", linewidth=0.5)
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude (µV)")
        plt.title("P300 Target vs Non-Target (Pre)")
        plt.grid(True)
        plt.legend(["Pre Target", "Pre Non-Target"])
        plt.show()

        pz_post_target = np.mean(p300_post[:, pz, :], axis=1)
        std_target_post = np.std(p300_post[:, pz, :], axis=1) / p300_post.shape[2]

        pz_post_nontarget = np.mean(nop300_post[:, pz, :], axis=1)
        std_nontarget_post = (
            np.std(nop300_post[:, pz, :], axis=1) / nop300_post.shape[2]
        )

        plt.figure()
        plt.plot(time_ms, pz_post_target, "b", linewidth=2)
        plt.plot(time_ms, pz_post_nontarget, "r", linewidth=2)
        plt.plot(time_ms, pz_post_target + std_target_post, "b", linewidth=0.5)
        plt.plot(time_ms, pz_post_target - std_target_post, "b", linewidth=0.5)
        plt.plot(time_ms, pz_post_nontarget + std_nontarget_post, "r", linewidth=0.5)
        plt.plot(time_ms, pz_post_nontarget - std_nontarget_post, "r", linewidth=0.5)
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude (µV)")
        plt.title("P300 Target vs Non-Target (Post)")
        plt.grid(True)
        plt.legend(["Post Target", "Post Non-Target"])
        plt.show()
