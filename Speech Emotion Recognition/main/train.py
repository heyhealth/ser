import sys
import os

sys.path.append(os.path.dirname((os.path.dirname(os.path.realpath(__file__)))))
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import torch
import d2l.torch
from d2l import torch as d2l
from utils.train_visualization import Animator
from d2l.torch import Accumulator, Timer
from opts import ARGS
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from torch import distributed
from utils.aam_softmax import AngularAdditiveMarginSoftMaxLoss


def compute_the_average_results(all_folds_results, display_labels):
    """
    :param all_folds_results:(f1,recall,accuracy,precision,y_true,y_pred) array-like
    :return:
    """
    print(
        "ALL FOLDS AVERAGE RESULTS", '\n',
        '-' * 30, '\n',
        f"average Unweighted accuracy(UA) :{np.mean([metric[0] for metric in all_folds_results]):>10.3f}\n",
        f"average Weighted accuracy(WA) :{np.mean([metric[1] for metric in all_folds_results]):>10.3f}\n",
        f"average f1(macro) :{np.mean([metric[2] for metric in all_folds_results]):>10.3f}\n",
        f"average recall(macro) :{np.mean([metric[3] for metric in all_folds_results]):>10.3f}\n",
        f"average precision(macro) :{np.mean([metric[4] for metric in all_folds_results]):>10.3f}\n",
        '-' * 30, '\n',
    )
    print("EVERY FOLD RESULTS")
    all_y_true, all_y_pred = [], []
    for fold, results_tuple in enumerate(all_folds_results):
        all_y_true.extend(results_tuple[5])
        all_y_pred.extend(results_tuple[6])
        print(
            f"FOLD {fold + 1}'s RESULTS",
            '-' * 30, '\n',
            f"Unweighted accuracy(UA) :{results_tuple[0]:>10.3f}\n",
            f"Weighted accuracy(WA) :{results_tuple[1]:>10.3f}\n",
            f"f1(macro) :{results_tuple[2]:>10.3f}\n",
            f"recall(macro) :{results_tuple[3]:>10.3f}\n",
            f"precision(macro) :{results_tuple[4]:>10.3f}\n",
            '-' * 30, '\n',
        )
        print(
            classification_report(results_tuple[5], results_tuple[6])
        )
    #  take in all fold predict and corresponding target
    ConfusionMatrixDisplay.from_predictions(all_y_true, all_y_pred, display_labels=display_labels)
    plt.savefig(os.path.join(ARGS.PROJECTION_PATH, 'save', 'ConfusionMatrix', 'cm_loso_all_session_sum.png'))


def train_model(net, train_iter, test_iter, epochs, lr, weight_decay, device, animator_name, class_weight):
    """
    train the model on datasets
    :param net:
    :param train_iter:
    :param test_iter:
    :param epochs:
    :param lr:
    :param weight_decay:
    :param device:
    :return:
    """
    net.to(device)
    optimizer = torch.optim.Adam(net.trainable_parameters(), lr, weight_decay=weight_decay)
    if class_weight is not None:
        loss_fun = nn.CrossEntropyLoss(weight=class_weight.to(device))
    else:
        loss_fun = nn.CrossEntropyLoss()
    # loss_fun = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', legend=['train loss', 'train acc', 'test acc(WA)', 'UA'], xlim=[1, epochs])
    batches = len(train_iter)
    print(f'{batches} batches training on {device}')
    timer = Timer()
    for epoch in range(epochs):
        net.train()
        accumulator = Accumulator(3)
        for i, (X, length, y) in enumerate(train_iter):
            timer.start()
            X, length, y = X.to(device), length.to(device), y.to(device)
            y_hat = net(X, length)
            optimizer.zero_grad()
            loss = loss_fun(y_hat, y)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                accumulator.add(loss * X.shape[0], calculate_the_correct_pred(y_hat, y), X.shape[0])
            timer.stop()
            train_loss = accumulator[0] / accumulator[2]
            train_acc = accumulator[1] / accumulator[2]
            if i == batches - 1 or i % (batches // 5) == 0:
                animator.add(epoch + (i + 1) / batches, (train_loss, train_acc, None, None))

        if test_iter is not None:
            test_acc, UA = evaluate_model_accuracy_in_training(net, test_iter, device)
            animator.add(epoch + 1, (None, None, test_acc, UA))
            print(
                f'epoch:[{epoch + 1}/{epochs}],train loss:{train_loss:.3f},train acc:{train_acc:.3f},test acc(WA):{test_acc:.3f},UA:{UA:.3f}')
        else:
            print(
                f'epoch:[{epoch + 1}/{epochs}],train loss:{train_loss:.3f},train acc:{train_acc:.3f}')

        # Add Early Stopping Mechanism
        if (train_acc >= 0.950 and train_loss <= 0.150) or (test_acc >= 0.9 if test_iter is not None else False):
            # if (train_acc >= 0.950 and train_loss <= 0.150):
            if test_iter is not None:
                animator.add(epoch + 1, (None, None, test_acc, UA))

            plt.savefig(os.path.join(ARGS.PROJECTION_PATH, 'save', 'train', f'{animator_name}.png'))
            print(f"{accumulator[2] * (epoch + 1) / timer.sum():.1f} samples/sec on {str(device)}")
            print("*************** Early Stopping ***************\n")
            # return the net which be trained on total epochs
            return net

        if epoch + 1 == epochs:
            plt.savefig(os.path.join(ARGS.PROJECTION_PATH, 'save', 'train', f'{animator_name}.png'))

    print(f"{accumulator[2] * epochs / timer.sum():.1f} samples/sec on {str(device)}")
    # return the net which be trained on total epochs
    return net


def train_model_for_aamsoftmax(net, train_iter, test_iter, epochs, lr, weight_decay, device, animator_name,
                               class_weight):
    """
    train the model on datasets
    :param net:
    :param train_iter:
    :param test_iter:
    :param epochs:
    :param lr:
    :param weight_decay:
    :param device:
    :return:
    """
    optimizer = torch.optim.Adam(net.trainable_parameters(), lr, weight_decay=weight_decay)
    animator = Animator(xlabel='epoch', legend=['train loss', 'train acc', 'test acc(WA)', 'UA'], xlim=[1, epochs])
    net.to(device)
    batches = len(train_iter)
    print(f'{batches} batches training on {device}')
    timer = Timer()
    for epoch in range(epochs):
        net.train()
        accumulator = Accumulator(3)
        for i, (X, length, y) in enumerate(train_iter):
            timer.start()
            X, length, y = X.to(device), length.to(device), y.to(device)
            loss, y_hat = net(X, length, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                accumulator.add(loss * X.shape[0], calculate_the_correct_pred(y_hat, y), X.shape[0])
            timer.stop()
            train_loss = accumulator[0] / accumulator[2]
            train_acc = accumulator[1] / accumulator[2]
            if i == batches - 1 or i % (batches // 5) == 0:
                animator.add(epoch + (i + 1) / batches, (train_loss, train_acc, None, None))

        if test_iter is not None:
            test_acc, UA = evaluate_model_accuracy_in_training(net, test_iter, device)
            animator.add(epoch + 1, (None, None, test_acc, UA))
            print(
                f'epoch:[{epoch + 1}/{epochs}],train loss:{train_loss:.3f},train acc:{train_acc:.3f},test acc(WA):{test_acc:.3f},UA:{UA:.3f}')
        else:
            print(
                f'epoch:[{epoch + 1}/{epochs}],train loss:{train_loss:.3f},train acc:{train_acc:.3f}')

        # Add Early Stopping Mechanism
        if train_acc >= 0.900:
            if test_iter is not None:
                animator.add(epoch + 1, (None, None, test_acc, UA))

            plt.savefig(os.path.join(ARGS.PROJECTION_PATH, 'save', 'train', f'{animator_name}.png'))
            print(f"{accumulator[2] * (epoch + 1) / timer.sum():.1f} samples/sec on {str(device)}")
            print("*************** Early Stopping ***************\n")
            # return the net which be trained on total epochs
            return net

        if epoch + 1 == epochs:
            plt.savefig(os.path.join(ARGS.PROJECTION_PATH, 'save', 'train', f'{animator_name}.png'))

    print(f"{accumulator[2] * epochs / timer.sum():.1f} samples/sec on {str(device)}")
    # return the net which be trained on total epochs
    return net


def train_model_for_multi_task(net, train_iter, test_iter, epochs, lr_asr, lr_ser, weight_decay, device, animator_name,
                               class_weight):
    """
    train the model on datasets
    :param net:
    :param train_iter:
    :param test_iter:
    :param epochs:
    :param lr:
    :param weight_decay:
    :param device:
    :return:
    """
    optimizer_asr = torch.optim.Adam(net.asr_task_trainable_parameters(), lr_asr, weight_decay=weight_decay)
    optimizer_ser = torch.optim.Adam(net.ser_task_trainable_parameters(), lr_ser, weight_decay=weight_decay)
    if class_weight is not None:
        loss_fun = nn.CrossEntropyLoss(weight=class_weight.to(device))
    else:
        loss_fun = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', legend=['train loss', 'train acc', 'test acc(WA)', 'UA'], xlim=[1, epochs])
    net.to(device)
    batches = len(train_iter)
    print(f'{batches} batches training on {device}')
    timer = Timer()
    for epoch in range(epochs):
        net.train()
        accumulator = Accumulator(3)
        for i, (X, length, y, token_ids) in enumerate(train_iter):
            timer.start()
            X, length, y, token_ids = X.to(device), length.to(device), y.to(device), token_ids.to(device)
            y_hat, asr_ctc_loss = net(X, length, token_ids)
            optimizer_asr.zero_grad()
            optimizer_ser.zero_grad()
            loss = loss_fun(y_hat, y)
            loss = 0.5 * loss * asr_ctc_loss / (loss + asr_ctc_loss) + 0.5 * loss
            loss.backward()
            optimizer_asr.step()
            optimizer_ser.step()
            with torch.no_grad():
                accumulator.add(loss * X.shape[0], calculate_the_correct_pred(y_hat, y), X.shape[0])
            timer.stop()
            train_loss = accumulator[0] / accumulator[2]
            train_acc = accumulator[1] / accumulator[2]
            if i == batches - 1 or i % (batches // 5) == 0:
                animator.add(epoch + (i + 1) / batches, (train_loss, train_acc, None, None))

        if test_iter is not None:
            test_acc, UA = evaluate_model_accuracy_in_training(net, test_iter, device)
            animator.add(epoch + 1, (None, None, test_acc, UA))
            print(
                f'epoch:[{epoch + 1}/{epochs}],train loss:{train_loss:.3f},train acc:{train_acc:.3f},test acc(WA):{test_acc:.3f},UA:{UA:.3f}')
        else:
            print(
                f'epoch:[{epoch + 1}/{epochs}],train loss:{train_loss:.3f},train acc:{train_acc:.3f}')

        # Add Early Stopping Mechanism
        if train_acc >= 0.950 and train_loss <= 0.150:
            if test_iter is not None:
                animator.add(epoch + 1, (None, None, test_acc, UA))

            plt.savefig(os.path.join(ARGS.PROJECTION_PATH, 'save', 'train', f'{animator_name}.png'))
            print(f"{accumulator[2] * (epoch + 1) / timer.sum():.1f} samples/sec on {str(device)}")
            print("*************** Early Stopping ***************\n")
            # return the net which be trained on total epochs
            return net

        if epoch + 1 == epochs:
            plt.savefig(os.path.join(ARGS.PROJECTION_PATH, 'save', 'train', f'{animator_name}.png'))

    print(f"{accumulator[2] * epochs / timer.sum():.1f} samples/sec on {str(device)}")
    # return the net which be trained on total epochs
    return net


def calculate_the_correct_pred(input, target):
    return sum(input.argmax(axis=1) == target).item()


def evaluate_model_accuracy_in_training(net, data_iter, device):
    """
    compute the model's accuracy on data iter
    :param net:
    :param data_iter:
    :param device:
    :return:
    """
    net.eval()
    net.to(device)
    accumulator = Accumulator(2)
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, length, y in data_iter:
            X, length, y = X.to(device), length.to(device), y.to(device)
            y_hat = net(X, length)
            y_hat_ = y_hat.argmax(axis=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(y_hat_.cpu().numpy())
            accumulator.add(calculate_the_correct_pred(y_hat, y), X.shape[0])
    UA = recall_score(y_true, y_pred, average='macro')
    return accumulator[0] / accumulator[1], UA


def evaluate_model_when_finish_training(net, data_iter, device, cm_name, display_labels):
    """
    compute the model's accuracy on data iter
    :param net:
    :param data_iter:
    :param device:
    :param cm_name:
    :param display_labels:
    :return:
    """
    net.eval()
    net.to(device)
    print(f"{len(data_iter)} batches testing on {device}")
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, length, y in data_iter:
            X, length = X.to(device), length.to(device)
            y_hat_ = net(X, length)
            y_hat = y_hat_.argmax(axis=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(y_hat.cpu().numpy())
    unweight_accuracy, weight_accuracy, f1, recall, precision = print_classification_metrics(y_true, y_pred, cm_name,
                                                                                             display_labels)
    return (unweight_accuracy, weight_accuracy, f1, recall, precision, y_true, y_pred)


def evaluate_model_when_finish_training_for_aamsoftmax(net, data_iter, device, cm_name, display_labels):
    """
    compute the model's accuracy on data iter
    :param net:
    :param data_iter:
    :param device:
    :param cm_name:
    :param display_labels:
    :return:
    """
    net.eval()
    net.to(device)
    print(f"{len(data_iter)} batches testing on {device}")
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, length, y in data_iter:
            X, length, y = X.to(device), length.to(device), y.to(device)
            _, y_hat_ = net(X, length, y)
            y_hat = y_hat_.argmax(axis=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(y_hat.cpu().numpy())
    unweight_accuracy, weight_accuracy, f1, recall, precision = print_classification_metrics(y_true, y_pred, cm_name,
                                                                                             display_labels)
    return (unweight_accuracy, weight_accuracy, f1, recall, precision, y_true, y_pred)


def evaluate_model_transfer_learning(net, data_iter, device, cm_name, display_labels):
    """
    compute the model's accuracy on data iter
    :param net:
    :param data_iter:
    :param device:
    :param cm_name:
    :param display_labels:
    :return:
    """
    net.eval()
    net.to(device)
    print(f"{len(data_iter)} batches testing on {device}")
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, length, y in data_iter:
            X, length = X.to(device), length.to(device)
            y_hat_ = net(X, length)
            y_hat = y_hat_.argmax(axis=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(y_hat.cpu().numpy())
    unweight_accuracy, weight_accuracy, f1, recall, precision = print_classification_metrics(y_true, y_pred, cm_name,
                                                                                             display_labels)
    return (unweight_accuracy, weight_accuracy, f1, recall, precision, y_true, y_pred)


def print_classification_metrics(y_true, y_pred, cm_name, display_labels):
    """
     print the classification metrics and plot the confusion matrix
    :param y_true:
    :param y_pred:
    :param cm_name:
    :return:
    """
    f1 = f1_score(y_true, y_pred, average='macro')
    # unweight_accuracy = accuracy_score(y_true, y_pred)
    weight_accuracy = accuracy_score(y_true, y_pred)
    """Weighted accuracy.

                Accuracy of the entire dataset, as defined in
                Han, Yu, and Tashev, "Speech Emotion Recognition Using Deep Neural
                Network and Extreme Learning Machine."

                """
    # weight_accuracy = balanced_accuracy_score(y_true, y_pred)
    unweight_accuracy = recall_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    """Unweighted accuracy.

               Average of recall of every class, as defined in
               Han, Yu, and Tashev, "Speech Emotion Recognition Using Deep Neural
               Network and Extreme Learning Machine."

               """
    precision = precision_score(y_true, y_pred, average='macro')
    print(
        '-' * 30, '\n',
        f"Weighted accuracy(WA) :{weight_accuracy:>10.3f}\n",
        f"Unweighted accuracy(UA) :{unweight_accuracy:>10.3f}\n",
        f"f1(macro) :{f1:>10.3f}\n",
        f"recall(macro) :{recall:>10.3f}\n",
        f"precision(macro) :{precision:>10.3f}\n",
        '-' * 30, '\n',
    )

    print(
        '-' * 70, '\n',
        classification_report(y_true, y_pred), '\n',
        '-' * 70, '\n',
    )
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=display_labels)
    plt.savefig(os.path.join(ARGS.PROJECTION_PATH, 'save', 'ConfusionMatrix', f'{cm_name}.png'))
    # plt.show()

    return unweight_accuracy, weight_accuracy, f1, recall, precision


def train_model_TAPT(net, data_iter, max_training_steps, warm_up_steps, lr, device, writer):
    optimizer = torch.optim.Adam(net.trainable_parameters(), lr=lr)
    # Learning rate scheduler
    num_training_steps = max_training_steps
    num_warmup_steps = warm_up_steps
    num_flat_steps = int(0.05 * num_training_steps)

    def lambda_lr(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step < (num_warmup_steps + num_flat_steps):
            return 1.0
        return max(
            0.0, float(num_training_steps - current_step) / float(
                max(1, num_training_steps - (num_warmup_steps + num_flat_steps)))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr)

    num_batches = len(data_iter)
    print(
        f"{num_batches} batches pretraining on {device}"
    )
    net.to(device)
    for epoch in range(num_training_steps):
        net.train()
        accumulator = Accumulator(2)
        for i, (X, length, _) in enumerate(data_iter):
            optimizer.zero_grad()
            X, length = X.to(device), length.to(device)
            outputs = net(X, length)
            pretrain_loss = outputs.loss
            pretrain_loss.backward()
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                accumulator.add(pretrain_loss, X.shape[0])
            pretrain_loss_ = accumulator[0] / accumulator[1]
            writer.add_scalar(tag='Loss(TAPT)',
                              scalar_value=pretrain_loss_,
                              global_step=epoch
                              )
            if i % (num_batches // 5) == 0 or i == num_batches - 1:
                print(
                    f"loss:{pretrain_loss_:.5f}"
                )
        print(
            f"------ [{epoch + 1}]/[{num_training_steps}] ------"
        )

    return net


# Modified from d2l:单机多卡(DP)(不推荐使用)
def train_model_TAPT_on_multi_gpu(net, data_iter, max_training_steps, warm_up_steps, lr, num_gpus, writer):
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    optimizer = torch.optim.Adam(net.trainable_parameters(), lr=lr)
    # Learning rate scheduler
    num_training_steps = max_training_steps
    num_warmup_steps = warm_up_steps
    num_flat_steps = int(0.05 * num_training_steps)

    def lambda_lr(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step < (num_warmup_steps + num_flat_steps):
            return 1.0
        return max(
            0.0, float(num_training_steps - current_step) / float(
                max(1, num_training_steps - (num_warmup_steps + num_flat_steps)))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr)

    num_batches = len(data_iter)
    print(
        f"{num_batches} batches pretraining on {devices}"
    )
    net = nn.DataParallel(net, device_ids=devices)
    net.to(devices[0])
    for epoch in range(num_training_steps):
        net.train()
        accumulator = Accumulator(2)
        for i, (X, length, _) in enumerate(data_iter):
            optimizer.zero_grad()
            X, length = X.to(devices[0]), length.to(devices[0])
            outputs = net(X, length)
            pretrain_loss = outputs.loss.sum()
            pretrain_loss.backward()
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                accumulator.add(pretrain_loss, X.shape[0])
            pretrain_loss_ = accumulator[0] / accumulator[1]
            writer.add_scalar(tag='Loss(TAPT)',
                              scalar_value=pretrain_loss_,
                              global_step=epoch
                              )
            if i % (num_batches // 5) == 0 or i == num_batches - 1:
                print(
                    f"loss:{pretrain_loss_:.5f}"
                )
        print(
            f"------ [{epoch + 1}]/[{num_training_steps}] ------"
        )

    return net


# Modified from d2l:单机多卡(DP)(不推荐使用)
def train_model_on_multi_gpu(net, train_iter, test_iter, epochs, lr, weight_decay, num_gpus, animator_name,
                             class_weight):
    """
    train the model on datasets
    :param net:
    :param train_iter:
    :param test_iter:
    :param epochs:
    :param lr:
    :param weight_decay:
    :param device: n
    :return:
    """
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    optimizer = torch.optim.Adam(net.trainable_parameters(), lr, weight_decay=weight_decay)
    loss_fun = nn.CrossEntropyLoss(weight=class_weight.to(devices[0]))
    animator = Animator(xlabel='epoch', legend=['train loss', 'train acc', 'test acc(WA)', 'UA'], xlim=[1, epochs])
    net = nn.DataParallel(net, device_ids=devices)
    net.to(devices[0])
    batches = len(train_iter)
    print(f'{batches} batches training on {devices}')
    timer = Timer()
    for epoch in range(epochs):
        net.train()
        accumulator = Accumulator(3)
        timer.start()
        for i, (X, length, y) in enumerate(train_iter):
            X, length, y = X.to(devices[0]), length.to(devices[0]), y.to(devices[0])
            y_hat = net(X, length)
            optimizer.zero_grad()
            loss = loss_fun(y_hat, y)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                accumulator.add(loss, calculate_the_correct_pred(y_hat, y), X.shape[0])
            timer.stop()
            train_loss = accumulator[0] / accumulator[2]
            train_acc = accumulator[1] / accumulator[2]
            if i == batches - 1 or i % (batches // 5) == 0:
                animator.add(epoch + (i + 1) / batches, (train_loss, train_acc, None, None))

        test_acc, UA = evaluate_model_accuracy_in_training(net, test_iter, devices[0])
        print(
            f'epoch:[{epoch + 1}/{epochs}],train loss:{train_loss:.3f},train acc:{train_acc:.3f},test acc(WA):{test_acc:.3f},UA:{UA:.3f}')
        animator.add(epoch + 1, (None, None, test_acc, UA))
        if epoch + 1 == epochs:
            plt.savefig(os.path.join(ARGS.PROJECTION_PATH, 'save', 'train', f'{animator_name}.png'))

    print(f"{accumulator[2] * epochs / timer.sum():.1f} samples/sec on {str(devices)}")
    # return the net which be trained on total epochs
    return net


# Modified from d2l:多机多卡(DDP)(推荐使用)
# TODO: evalute on test data (未完善)
def train_model_on_distributed_mechine(net, train_iter, test_iter, epochs, lr, weight_decay, local_rank, device,
                                       animator_name, class_weight):
    """
    train the model on datasets
    :param net:
    :param train_iter:
    :param test_iter:
    :param epochs:
    :param lr:
    :param weight_decay:
    :param local_rank:
    :param device:
    :param animator_name:
    :param class_weight
    :return:
    """

    def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
        rt = tensor.clone()
        distributed.all_reduce(rt, op=distributed.ReduceOp.SUM)
        rt /= distributed.get_world_size()  # 总进程数
        return rt

    optimizer = torch.optim.Adam(net.trainable_parameters(), lr, weight_decay=weight_decay)
    loss_fun = nn.CrossEntropyLoss(weight=class_weight.to(device))
    animator = Animator(xlabel='epoch', legend=['train loss', 'train acc', 'test acc(WA)', 'UA'], xlim=[1, epochs])
    net.to(
        device
    )
    net = nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank,
                                              broadcast_buffers=False, find_unused_parameters=True)
    batches = len(train_iter)
    print(f'{batches} batches training on cuda {local_rank}', end='\n')
    for epoch in range(epochs):
        net.train()
        accumulator = Accumulator(3)
        for i, (X, length, y) in enumerate(train_iter):
            X, length, y = X.to(device), length.to(device), y.to(device)
            y_hat = net(X, length)
            optimizer.zero_grad()
            loss = loss_fun(y_hat, y)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                accumulator.add(loss * X.shape[0], calculate_the_correct_pred(y_hat, y), X.shape[0])
            train_loss = accumulator[0] / accumulator[2]
            train_acc = accumulator[1] / accumulator[2]
            train_loss = torch.as_tensor(train_loss, dtype=torch.float32, device=device)
            train_acc = torch.as_tensor(train_acc, dtype=torch.float32, device=device)
            train_loss_avg = reduce_tensor(train_loss)
            train_acc_avg = reduce_tensor(train_acc)
            if local_rank == 0:
                if i == batches - 1 or i % (batches // 5) == 0:
                    animator.add(epoch + (i + 1) / batches,
                                 (train_loss_avg.cpu().numpy(), train_acc_avg.cpu().numpy(), None, None))
        with torch.no_grad():
            test_acc, UA = evaluate_model_accuracy_in_training(net, test_iter, device)
            test_acc = torch.as_tensor(test_acc, dtype=torch.float32, device=device)
            UA = torch.as_tensor(UA, dtype=torch.float32, device=device)
            UA_avg = reduce_tensor(UA)
            test_acc_avg = reduce_tensor(test_acc)
        if local_rank == 0:
            print(
                f'epoch:[{epoch + 1}/{epochs}],train loss:{train_loss_avg:.3f},train acc:{train_acc_avg:.3f},test acc(WA):{test_acc_avg:.3f},UA:{UA_avg:.3f}')
            animator.add(epoch + 1, (None, None, test_acc_avg.cpu().numpy(), UA_avg.cpu().numpy()))
            if epoch + 1 == epochs:
                plt.savefig(os.path.join(ARGS.PROJECTION_PATH, 'save', 'train', f'{animator_name}.png'))

    # return the net which be trained on total epochs
    return net


# Modified from d2l:多机多卡(DDP)(推荐使用)
# TODO: (未完善)
def train_model_TAPT_on_distributed_mechine(net, data_iter, max_training_steps, warm_up_steps, lr, device, local_rank,
                                            writer):
    optimizer = torch.optim.Adam(net.trainable_parameters(), lr=lr)
    # Learning rate scheduler
    num_training_steps = max_training_steps
    num_warmup_steps = warm_up_steps
    num_flat_steps = int(0.05 * num_training_steps)

    def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
        rt = tensor.clone()
        distributed.all_reduce(rt, op=distributed.ReduceOp.SUM)
        rt /= distributed.get_world_size()  # 总进程数
        return rt

    def lambda_lr(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step < (num_warmup_steps + num_flat_steps):
            return 1.0
        return max(
            0.0, float(num_training_steps - current_step) / float(
                max(1, num_training_steps - (num_warmup_steps + num_flat_steps)))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr)

    num_batches = len(data_iter)
    print(
        f"{num_batches} batches pretraining on {device}"
    )
    net.to(device)
    net = nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank,
                                              find_unused_parameters=True)
    for epoch in range(num_training_steps):
        net.train()
        accumulator = Accumulator(2)
        for i, (X, length, _) in enumerate(data_iter):
            optimizer.zero_grad()
            X, length = X.to(device), length.to(device)
            outputs = net(X, length)
            pretrain_loss = outputs.loss
            pretrain_loss.backward()
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                accumulator.add(pretrain_loss * X.shape[0], X.shape[0])
                pretrain_loss_ = accumulator[0] / accumulator[1]
                pretrain_loss_ = torch.as_tensor(pretrain_loss_, dtype=torch.float32, device=X.device)
                pretrain_loss_avg = reduce_tensor(pretrain_loss_)
            if local_rank == 0:
                writer.add_scalar(tag='Loss(TAPT)',
                                  scalar_value=pretrain_loss_avg.data,
                                  global_step=epoch
                                  )

                if i % (num_batches // 5) == 0 or i == num_batches - 1:
                    print(
                        f"loss:{pretrain_loss_avg:.5f}"
                    )

        if local_rank == 0:
            print(
                f"------ [{epoch + 1}]/[{num_training_steps}] ------"
            )

    return net


if __name__ == '__main__':
    # y_true = [0, 1, 2, 3, 4, 1, 2, 3, 4, 1, 1, 1, 1]
    # y_pred = [0, 1, 1, 3, 2, 1, 4, 3, 4, 1, 1, 2, 1]
    # y_true = [0, 1, 0, 0, 1, 0]
    # y_pred = [0, 1, 0, 0, 0, 1]
    # print_classification_metrics(y_true, y_pred, cm_name='none', display_labels=None)
    print(
        os.path.dirname((os.path.dirname(os.path.realpath(__file__))))
    )
