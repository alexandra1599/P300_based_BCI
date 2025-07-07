def run_analysis(
    ID,
    session,
    labels,
    p300_pre,
    p300_post,
    nop300_pre,
    nop300_post,
    fs=512,
):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d

    # === Channel indices (adjust based on session) ===
    if session == "tACS":
        pz, cz, c1, c2 = 11, 15, 19, 21
    elif session == "Relaxation":
        pz, cz, cpz, c1, c2 = 26, 15, 20, 19, 21

    prompt = "CPP[1] or P300[2] or Target[3] analysis: "
    c = int(input(prompt))

    time_ms = np.arange(p300_pre.shape[0]) * 1000 / fs

    if c == 1:  # === CPP Analysis ===
        cpp_pre = np.mean(
            np.mean(
                np.concatenate((nop300_pre[:, c1, :], nop300_pre[:, c2, :]), axis=1),
                axis=1,
            )
        )
        cpp_post = np.mean(
            np.mean(
                np.concatenate((nop300_post[:, c1, :], nop300_post[:, c2, :]), axis=1),
                axis=1,
            )
        )

        stdpre = np.std(
            np.concatenate((nop300_pre[:, c1, :], nop300_pre[:, c2, :]), axis=1), axis=1
        ) / np.sqrt(nop300_pre.shape[2])
        stdpost = np.std(
            np.concatenate((nop300_post[:, c1, :], nop300_post[:, c2, :]), axis=1),
            axis=1,
        ) / np.sqrt(nop300_post.shape[2])

        plt.figure()
        plt.plot(
            time_ms, gaussian_filter1d(cpp_pre, sigma=2), label="CPP Pre", color="black"
        )
        plt.plot(
            time_ms,
            gaussian_filter1d(cpp_post, sigma=2),
            label="CPP Post",
            color="green",
        )
        plt.fill_between(
            time_ms, cpp_pre - stdpre, cpp_pre + stdpre, color="black", alpha=0.2
        )
        plt.fill_between(
            time_ms, cpp_post - stdpost, cpp_post + stdpost, color="green", alpha=0.2
        )
        plt.title("CPP Analysis (CP1 + CP2)")
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude (µV)")
        plt.legend()
        plt.grid(True)

        plt.show()

    elif c == 2:  # === P300 Target Analysis ===
        pz_pre_target = np.mean(p300_pre[:, pz, :], axis=1)
        pz_post_target = np.mean(p300_post[:, pz, :], axis=1)

        stdpre = np.std(p300_pre[:, pz, :], axis=1) / np.sqrt(p300_pre.shape[2])
        stdpost = np.std(p300_post[:, pz, :], axis=1) / np.sqrt(p300_post.shape[2])

        plt.figure()
        plt.plot(
            time_ms,
            gaussian_filter1d(pz_pre_target, sigma=2),
            label="P300 Pre",
            color="black",
        )
        plt.plot(
            time_ms,
            gaussian_filter1d(pz_post_target, sigma=2),
            label="P300 Post",
            color="green",
        )
        plt.fill_between(
            time_ms,
            pz_pre_target - stdpre,
            pz_pre_target + stdpre,
            color="black",
            alpha=0.2,
        )
        plt.fill_between(
            time_ms,
            pz_post_target - stdpost,
            pz_post_target + stdpost,
            color="green",
            alpha=0.2,
        )
        plt.title("P300 Analysis at Pz")
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude (µV)")
        plt.legend()
        plt.grid(True)
        plt.show()

    elif c == 3:  # === Target vs Non-Target (Pre and Post) ===
        for label, targ, nontarg in zip(
            ["Pre", "Post"], [p300_pre, p300_post], [nop300_pre, nop300_post]
        ):
            target = np.mean(targ[:, pz, :], axis=1)
            nontarget = np.mean(nontarg[:, pz, :], axis=1)

            std_target = np.std(targ[:, pz, :], axis=1) / np.sqrt(targ.shape[2])
            std_nontarget = np.std(nontarg[:, pz, :], axis=1) / np.sqrt(
                nontarg.shape[2]
            )

            plt.figure()
            plt.plot(
                time_ms,
                gaussian_filter1d(target, sigma=2),
                label=f"{label} Target",
                color="blue",
            )
            plt.plot(
                time_ms,
                gaussian_filter1d(nontarget, sigma=2),
                label=f"{label} Non-Target",
                color="red",
            )
            plt.fill_between(
                time_ms,
                target - std_target,
                target + std_target,
                color="blue",
                alpha=0.2,
            )
            plt.fill_between(
                time_ms,
                nontarget - std_nontarget,
                nontarget + std_nontarget,
                color="red",
                alpha=0.2,
            )
            plt.title(f"{label} P300: Target vs Non-Target")
            plt.xlabel("Time (ms)")
            plt.ylabel("Amplitude (µV)")
            plt.legend()
            plt.grid(True)

            plt.show()
