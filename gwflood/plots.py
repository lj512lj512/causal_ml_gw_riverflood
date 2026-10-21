"""Figures based on explicit data inputs; no module-level I/O."""
def plot_effect_distribution(effects, value="effect", ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.hist(effects[value].dropna(), bins=30)
    ax.set_xlabel("Estimated effect (outcome model scale)")
    ax.set_ylabel("Count")
    return ax
