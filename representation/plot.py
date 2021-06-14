import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def plot_dataset_images(model, dataset, num_col, target=None, row_max=500, is_showed=True, save_path=None):
    dataset_size = len(dataset)

    if row_max:
        num_row = row_max
        residual = row_max % num_col
    else:
        num_row = dataset_size // num_col
        residual = dataset_size % num_col

    if residual != 0:
        num_row += 1

    num_row = num_row * 2

    fig, ax = plt.subplots(nrows=num_row, ncols=num_col, figsize=(num_col * 3, num_row * 3))
    data_idx = 0

    for row in range(0, num_row, 2):
        for col in range(num_col):
            # Odd : original
            odd_row = row
            print(odd_row, col)
            current_ax = ax[odd_row][col]

            x = dataset[data_idx][0].squeeze()
            target = dataset[data_idx][1]

            current_ax.imshow(x, cmap='gray')
            current_ax.set_xticks([])
            current_ax.set_yticks([])
            current_ax.set_title(f"\nTarget : {target}")

            # Even : predict
            even_row = row + 1
            print(even_row, col)

            current_ax = ax[even_row][col]

            x = dataset[data_idx][0]
            input_shape = x.shape

            x = x.view(x.size(0), -1)
            x_hat, z = model(x)

            x_hat = x_hat.view(input_shape).detach().numpy().squeeze()
            target = dataset[data_idx][1]

            current_ax.imshow(x_hat.squeeze(), cmap='gray')
            current_ax.set_xticks([])
            current_ax.set_yticks([])
            current_ax.set_title(f"Reconstructed")

            data_idx += 1

            if data_idx == dataset_size:
                break

    if save_path:
        plt.savefig(save_path)

    if is_showed:
        plt.show()


def get_metrics_data_dict(train_result_dict: dict):
    train_loss_list = list()
    train_accuracy_list = list()
    train_precision_list = list()
    train_recall_list = list()
    train_f1_list = list()

    val_loss_list = list()
    val_accuracy_list = list()
    val_precision_list = list()
    val_recall_list = list()
    val_f1_list = list()

    for train_history_dict, val_history_dict in zip(train_result_dict["train_history"], train_result_dict["val_history"]):
        train_loss_list.append(train_history_dict["loss"])
        train_accuracy_list.append(train_history_dict["accuracy"])
        train_precision_list.append(train_history_dict["precision"])
        train_recall_list.append(train_history_dict["recall"])
        train_f1_list.append(train_history_dict["f1"])

        val_loss_list.append(val_history_dict["loss"])
        val_accuracy_list.append(val_history_dict["accuracy"])
        val_precision_list.append(val_history_dict["precision"])
        val_recall_list.append(val_history_dict["recall"])
        val_f1_list.append(val_history_dict["f1"])

    return dict(
        train_loss_list=train_loss_list,
        train_accuracy_list=train_accuracy_list,
        train_precision_list=train_precision_list,
        train_recall_list=train_recall_list,
        train_f1_list=train_f1_list,
        val_loss_list=val_loss_list,
        val_accuracy_list=val_accuracy_list,
        val_precision_list=val_precision_list,
        val_recall_list=val_recall_list,
        val_f1_list=val_f1_list
    )


def plot_loss(train_data_list, val_data_list, early_stopping_epoch, ax=None, y_lim=(-0.2, 10), figsize=(15, 7)):
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    ax.plot(range(len(train_data_list)), train_data_list, label="Train Loss", color="blue")
    ax.plot(range(len(val_data_list)), val_data_list, label="Validation Loss", color="red")
    ax.axvline(x=early_stopping_epoch, linestyle="--", label=f"Early Stopping (Epoch : {early_stopping_epoch})",
               color="black")
    ax.set_ylim(y_lim)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Train-Validation Loss")
    ax.legend()

    return ax


def plot_accuracy(train_data_list, val_data_list, early_stopping_epoch, ax=None, y_lim=(-0.02, 1.02), figsize=(15, 7)):
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    ax.plot(range(len(train_data_list)), train_data_list, label="Train Accuracy", color="blue")
    ax.plot(range(len(val_data_list)), val_data_list, label="Validation Accuracy", color="red")
    ax.axvline(x=early_stopping_epoch, linestyle="--", label=f"Early Stopping (Epoch : {early_stopping_epoch})",
               color="black")
    ax.set_ylim(y_lim)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.set_title("Train-Validation Accuracy")
    ax.legend()

    return ax


def plot_precision(train_data_list, val_data_list, early_stopping_epoch, ax=None, y_lim=(-0.02, 1.02), figsize=(15, 7)):
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    ax.plot(range(len(train_data_list)), train_data_list, label="Train Precision", color="blue")
    ax.plot(range(len(val_data_list)), val_data_list, label="Validation Precision", color="red")
    ax.axvline(x=early_stopping_epoch, linestyle="--", label=f"Early Stopping (Epoch : {early_stopping_epoch})",
               color="black")
    ax.set_ylim(y_lim)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Precision")
    ax.set_title("Train-Validation Precision")
    ax.legend()

    return ax


def plot_recall(train_data_list, val_data_list, early_stopping_epoch, ax=None, y_lim=(-0.02, 1.02), figsize=(15, 7)):
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    ax.plot(range(len(train_data_list)), train_data_list, label="Train Recall", color="blue")
    ax.plot(range(len(val_data_list)), val_data_list, label="Validation Recall", color="red")
    ax.axvline(x=early_stopping_epoch, linestyle="--", label=f"Early Stopping (Epoch : {early_stopping_epoch})",
               color="black")
    ax.set_ylim(y_lim)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Recall")
    ax.set_title("Train-Validation Recall")
    ax.legend()

    return ax


def plot_f1(train_data_list, val_data_list, early_stopping_epoch, ax=None, y_lim=(-0.02, 1.02), figsize=(15, 7)):
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    ax.plot(range(len(train_data_list)), train_data_list, label="Train F1 Score", color="blue")
    ax.plot(range(len(val_data_list)), val_data_list, label="Validation F1 Score", color="red")
    ax.axvline(x=early_stopping_epoch, linestyle="--", label=f"Early Stopping (Epoch : {early_stopping_epoch})",
               color="black")
    ax.set_ylim(y_lim)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("F1")
    ax.set_title("Train-Validation F1 Score")
    ax.legend()

    return ax


def plot_roc_curve(fpr, tpr, auc, ax=None, y_lim=(-0.02, 1.02), figsize=(8, 8)):
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    ax.plot(fpr, tpr, label="ROC", color="red")
    ax.plot([0, 1], [0, 1], linestyle="--", color="black")
    ax.fill_between(fpr, 0, tpr, alpha=0.5, label=f"AUC : {auc}")
    ax.set_ylim(y_lim)
    ax.set_xlabel("Fall-Out")
    ax.set_ylabel("Recall")
    ax.set_title("ROC Curve")
    ax.legend()

    return ax


def plot_metrics(train_result_dict, is_showed=True, is_saved=False, save_file_path=None):
    early_stopping_epoch = train_result_dict["early_stopping_epoch"]
    fpr = train_result_dict["roc"]["fpr"]
    tpr = train_result_dict["roc"]["tpr"]
    auc = train_result_dict["auc"]
    metrics_data_dict = get_metrics_data_dict(train_result_dict=train_result_dict)

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(15, 22))
    plot_loss(
        train_data_list=metrics_data_dict["train_loss_list"],
        val_data_list=metrics_data_dict["val_loss_list"],
        early_stopping_epoch=early_stopping_epoch,
        ax=ax[0][0]
    )
    plot_accuracy(
        train_data_list=metrics_data_dict["train_accuracy_list"],
        val_data_list=metrics_data_dict["val_accuracy_list"],
        early_stopping_epoch=early_stopping_epoch,
        ax=ax[0][1]
    )
    plot_precision(
        train_data_list=metrics_data_dict["train_precision_list"],
        val_data_list=metrics_data_dict["val_precision_list"],
        early_stopping_epoch=early_stopping_epoch,
        ax=ax[1][0]
    )
    plot_recall(
        train_data_list=metrics_data_dict["train_recall_list"],
        val_data_list=metrics_data_dict["val_recall_list"],
        early_stopping_epoch=early_stopping_epoch,
        ax=ax[1][1]
    )
    plot_f1(
        train_data_list=metrics_data_dict["train_f1_list"],
        val_data_list=metrics_data_dict["val_f1_list"],
        early_stopping_epoch=early_stopping_epoch,
        ax=ax[2][0]
    )
    plot_roc_curve(
        fpr=fpr,
        tpr=tpr,
        auc=auc,
        ax=ax[2][1]
    )

    # Save plot
    if is_saved:
        plt.savefig(save_file_path, dpi=300)

    # Show plot
    if is_showed:
        plt.show()
