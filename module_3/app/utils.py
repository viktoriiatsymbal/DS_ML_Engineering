from pathlib import Path
import matplotlib.pyplot as plt

def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_confusion_matrix_plot(cm, labels, out_path, title):
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def save_results_plot(df, out_path):
    out_path = Path(out_path)
    top = df.sort_values("f1_macro", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(top["run_name"], top["f1_macro"])
    ax.set_title("Top 10 runs by macro F1")
    ax.set_ylabel("macro F1")
    ax.set_xlabel("run")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
