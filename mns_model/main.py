import logging

import torch
from sklearn.metrics import accuracy_score
from torch import nn, optim
from torch.utils.data import DataLoader

from mns_model.data_process import get_LogFillterBank, load_data_fromdat
from mns_model.dataset import SpeechEmoDataset
from mns_model.models import FusionModel, CNN, DNN, SeperateModel
from mns_model.utils import model_output, compute_metrics, set_seed
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/separate-5dnn+fusion')

logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",
    level=logging.INFO)

MODELCLASS = [CNN, DNN]


def train_separate(model, train_loader, optimizer, criterion, device, alpha=None):
    model.train()
    total_loss = 0
    cnn_loss = 0
    dnn_loss = 0
    cnn_all_preds, cnn_all_labels = [], []
    dnn_all_preds, dnn_all_labels = [], []

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        dnn_out, cnn_out = model(inputs)
        loss_cnn = criterion(cnn_out, labels)
        loss_dnn = criterion(dnn_out, labels)
        if alpha is None:
            loss = loss_cnn + loss_dnn
        else:
            loss = alpha * loss_cnn + (1-alpha) * loss_dnn

        loss.backward()
        optimizer.step()

        cnn_loss += loss_cnn.item()
        dnn_loss += loss_dnn.item()
        total_loss += loss.item()
        _, preds = torch.max(cnn_out, 1)
        cnn_all_preds.extend(preds.cpu().numpy())
        cnn_all_labels.extend(labels.cpu().numpy())
        _, preds = torch.max(dnn_out, 1)
        dnn_all_preds.extend(preds.cpu().numpy())
        dnn_all_labels.extend(labels.cpu().numpy())

    cnn_accuracy = accuracy_score(cnn_all_labels, cnn_all_preds)
    dnn_accuracy = accuracy_score(dnn_all_labels, dnn_all_preds)

    return (cnn_loss / len(train_loader), dnn_loss / len(train_loader),
            total_loss / len(train_loader), cnn_accuracy, dnn_accuracy)


# 训练和验证函数
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        if any(isinstance(model, model_class) for model_class in MODELCLASS):
            out, _ = model(inputs)
        else:
            out, _, _ = model(inputs)
        loss = criterion(out, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(out, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def evaluate_separate(model, test_loader, criterion, device, alpha=None):
    model.eval()
    total_loss = 0
    cnn_loss = 0
    dnn_loss = 0
    cnn_all_preds, cnn_all_labels = [], []
    dnn_all_preds, dnn_all_labels = [], []

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        dnn_out, cnn_out = model(inputs)
        loss_cnn = criterion(cnn_out, labels)
        loss_dnn = criterion(dnn_out, labels)
        if alpha is None:
            loss = loss_cnn + loss_dnn
        else:
            loss = alpha * loss_cnn + (1-alpha) * loss_dnn

        cnn_loss += loss_cnn.item()
        dnn_loss += loss_dnn.item()
        total_loss += loss.item()
        _, preds = torch.max(cnn_out, 1)
        cnn_all_preds.extend(preds.cpu().numpy())
        cnn_all_labels.extend(labels.cpu().numpy())
        _, preds = torch.max(dnn_out, 1)
        dnn_all_preds.extend(preds.cpu().numpy())
        dnn_all_labels.extend(labels.cpu().numpy())

    cnn_accuracy = accuracy_score(cnn_all_labels, cnn_all_preds)
    dnn_accuracy = accuracy_score(dnn_all_labels, dnn_all_preds)

    return (cnn_loss / len(test_loader), dnn_loss / len(test_loader),
            total_loss / len(test_loader), cnn_accuracy, dnn_accuracy)


def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if any(isinstance(model, model_class) for model_class in MODELCLASS):
                outputs, _ = model(inputs)
            else:
                outputs, _, _ = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def main_separate(dim=80, alpha=None, num_labels=18):
    # 数据准备
    logging.info("data process")

    train_dataset = SpeechEmoDataset(split="train", nfilt=dim)
    test_dataset = SpeechEmoDataset(split="test", nfilt=dim)

    batch_size = 6400
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 定义模型
    logging.info("prepare model")
    input_shape = (dim, 1)
    model = SeperateModel(input_shape, num_classes=num_labels)
    parameters = sum([torch.numel(params) for params in model.parameters()])
    logging.info("model params: {}".format(parameters))

    optimizer = optim.AdamW(model.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1e-08)
    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 设置训练参数
    num_epochs = 150
    max_acc_cnn = 0
    max_acc_dnn = 0

    # 训练和验证循环
    logging.info("start train")
    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch + 1}/{num_epochs}")

        # train_loss, train_accuracy = train(model, train_loader, optimizer, criterion, device)
        train_cnn_loss, train_dnn_loss, train_loss, train_cnn_acc, train_dnn_acc = (
            train_separate(model, train_loader, optimizer, criterion, device, alpha=alpha))
        val_cnn_loss, val_dnn_loss, val_loss, val_cnn_acc, val_dnn_acc = (
            evaluate_separate(model, test_loader, criterion, device, alpha=alpha))

        logging.info(
            f"Train CNN Loss: {train_cnn_loss:.4f}, Train DNN Loss: {train_dnn_loss:.4f}, Train Loss: {train_loss:.4f},"
            f" Train CNN Accuracy: {train_cnn_acc:.4f}, Train DNN Accuracy: {train_dnn_acc:.4f}")
        logging.info(
            f"Validation CNN Loss: {val_cnn_loss:.4f}, Validation DNN Loss: {val_dnn_loss:.4f}, Validation Loss: "
            f"{val_loss:.4f}, Validation CNN Accuracy: {val_cnn_acc:.4f}, Validation DNN Accuracy: {val_dnn_acc:.4f}")

        writer.add_scalar('Loss/Train/CNN', train_cnn_loss, epoch)
        writer.add_scalar('Loss/Train/DNN', train_dnn_loss, epoch)
        writer.add_scalar('Loss/Train/Total', train_loss, epoch)
        writer.add_scalar('Accuracy/Train/CNN', train_cnn_acc, epoch)
        writer.add_scalar('Accuracy/Train/DNN', train_dnn_acc, epoch)

        writer.add_scalar('Loss/Validation/CNN', val_cnn_loss, epoch)
        writer.add_scalar('Loss/Validation/DNN', val_dnn_loss, epoch)
        writer.add_scalar('Loss/Validation/Total', val_loss, epoch)
        writer.add_scalar('Accuracy/Validation/CNN', val_cnn_acc, epoch)
        writer.add_scalar('Accuracy/Validation/DNN', val_dnn_acc, epoch)

        # 保存最佳模型
        if val_cnn_acc > max_acc_cnn:  # 这里只是一个示例，您需要保存实际的最佳模型
            torch.save(model.cnn.state_dict(), "model/cnn_model_best.pth")
            max_acc_cnn = val_cnn_acc
        if val_dnn_acc > max_acc_dnn:  # 这里只是一个示例，您需要保存实际的最佳模型
            torch.save(model.dnn.state_dict(), "model/dnn_model_best.pth")
            max_acc_dnn = val_dnn_acc

    # 保存最终模型
    torch.save(model.state_dict(), "model/model_5dnn_separate.pth")


def main(dim=80, num_labels=18):
    # 数据准备
    logging.info("data process")

    train_dataset = SpeechEmoDataset(split="train", nfilt=dim)
    test_dataset = SpeechEmoDataset(split="test", nfilt=dim)

    batch_size = 6400
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 定义模型
    logging.info("prepare model")
    input_shape = (dim, 1)
    model = FusionModel(input_shape=input_shape, mode="train", num_classes=num_labels)
    parameters = sum([torch.numel(params) for params in model.parameters()])
    logging.info("model params: {}".format(parameters))

    optimizer = optim.AdamW(model.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1e-08)
    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 设置训练参数
    num_epochs = 400
    max_acc = 0

    # 训练和验证循环
    logging.info("start train")
    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch + 1}/{num_epochs}")

        train_loss, train_accuracy = train(model, train_loader, optimizer, criterion, device)
        valid_loss, valid_accuracy = evaluate(model, test_loader, criterion, device)

        logging.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        logging.info(f"Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.4f}")

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_accuracy, epoch)

        writer.add_scalar('Loss/Valid', valid_loss, epoch)
        writer.add_scalar('Accuracy/Valid', valid_accuracy, epoch)

        # 保存最佳模型
        if valid_accuracy > max_acc:
            torch.save(model.state_dict(), "model/model_sf_best.pth")
            max_acc = valid_accuracy

    # 保存最终模型
    torch.save(model.state_dict(), "model/model_sf.pth")


def valid(dim=26, num_labels=18):
    logging.info("data process")
    test_data, test_label = load_data_fromdat(path=None, split="test")
    test_lfb = get_LogFillterBank(test_data, sample_rate=16000, nfilt=dim)

    input_shape = (dim, 1)
    model = FusionModel(input_shape, mode="test", num_classes=num_labels)
    model.load_state_dict(torch.load("model_iemocap/model_sf_5dnn_best.pth"))

    true_labels, score_labels, answers, scores = model_output(test_lfb, test_label, model)     # test
    acc_score, eer_score, dcf = compute_metrics(true_labels, score_labels, answers, scores)
    logging.info("acc_score: {}, eer_score: {}, dcf: {}".format(acc_score, eer_score, dcf))


if __name__ == '__main__':
    set_seed(42)
    # main_separate(dim=26, alpha=None, num_labels=10)
    # main(dim=26, num_labels=10)
    valid(dim=26, num_labels=10)
