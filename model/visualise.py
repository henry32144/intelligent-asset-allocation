import config
import matplotlib.pyplot as plt


def plot_history(history):
    plt.figure(figsize=(15, 6))

    plt.subplot(2, 2, 1)
    plt.plot([int(x + 1) for x in range(config.EPOCHS)], history["train_mcc"], label="train mcc")
    plt.plot([int(x + 1) for x in range(config.EPOCHS)], history["val_mcc"], label="validation mcc")
    plt.title("Training MCC History")
    plt.ylabel("MCC")
    plt.xlabel("Epoch")
    plt.ylim([-1, 1])
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot([int(x + 1) for x in range(config.EPOCHS)], history["train_f1"], label="train f1")
    plt.plot([int(x + 1) for x in range(config.EPOCHS)], history["val_f1"], label="validation f1")
    plt.title("Training F1 History")
    plt.ylabel("F1")
    plt.xlabel("Epoch")
    plt.ylim([0, 1])
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.plot([int(x + 1) for x in range(config.EPOCHS)], history["train_acc"], label="train acc")
    plt.plot([int(x + 1) for x in range(config.EPOCHS)], history["val_acc"], label="validation acc")
    plt.title("Training Accuracy History")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.ylim([0, 1])
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot([int(x + 1) for x in range(config.EPOCHS)], history["train_loss"], label="train loss")
    plt.plot([int(x + 1) for x in range(config.EPOCHS)], history["val_loss"], label="validation loss")
    plt.title("Training Loss History")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.ylim([0, 1])
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()