import cv2
import matplotlib.pyplot as plt

from model import *
from predict import *
from show import *
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import copy

cudnn.benchmark = True
nets = ["UNet", "UNet11", "UNet16", "LinkNet34", "AlbuNet"]
use_net = nets[0]

params = {
    "model": use_net,
    "device": "cuda",
    "lr": 0.001,
    "batch_size": 8,
    "num_workers": 0,
    "epochs": 30,
}
# batch_size 过大会导致pytorch内存不足，16的batch_size=8，11的batch_size=16
weight_path = "save/" + use_net + ".pth"
loss_path = "save/loss/" + use_net + ".txt"
loss_train_item = []
loss_validate_item = []


def train(train_loader, model, criterion, optimizer, epoch, params):
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)
    sum_loss = 0
    size = 0
    for i, (images, target) in enumerate(stream, start=1):
        images = images.to(params["device"], non_blocking=True)
        target = target.to(params["device"], non_blocking=True)
        output = model(images).squeeze(1)
        loss = criterion(output, target)
        metric_monitor.update("Loss", loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # loss_item.append()
        sum_loss += round(metric_monitor.metrics["Loss"]["avg"] + 0.005, 2)
        stream.set_description(
            "Epoch: {epoch}. Train.      {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
        )
        size = i
    loss_train_item.append(round(sum_loss / size, 2))


def validate(val_loader, model, criterion, epoch, params):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    sum_loss = 0
    size = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(params["device"], non_blocking=True)
            target = target.to(params["device"], non_blocking=True)
            output = model(images).squeeze(1)
            loss = criterion(output, target)
            metric_monitor.update("Loss", loss.item())

            sum_loss += round(metric_monitor.metrics["Loss"]["avg"] + 0.005, 2)

            stream.set_description(
                "Epoch: {epoch}. Validation. {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
            )
            size = i
    loss_validate_item.append(round(sum_loss / size, 2))


def train_and_validate(model, train_dataset, val_dataset, params):
    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=params["num_workers"],
        pin_memory=True,
    )
    criterion = nn.BCEWithLogitsLoss().to(params["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    for epoch in range(1, params["epochs"] + 1):
        train(train_loader, model, criterion, optimizer, epoch, params)
        validate(val_loader, model, criterion, epoch, params)
        if epoch % 10 == 0:
            torch.save(model, weight_path)
    return model


def visualize_augmentations(dataset, idx=0, samples=5):  # 可视化
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    figure, ax = plt.subplots(nrows=samples, ncols=2, figsize=(10, 24))
    for i in range(samples):
        image, mask = dataset[idx]
        ax[i, 0].imshow(image)
        ax[i, 1].imshow(mask, interpolation="nearest")
        ax[i, 0].set_title("Augmented image")
        ax[i, 1].set_title("Augmented mask")
        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()
    plt.tight_layout()
    plt.show()


def define_train_Params(train_images_filenames, val_images_filenames):
    train_transform = A.Compose(
        [
            A.Resize(256, 256),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    train_dataset = OxfordPetDataset(train_images_filenames, images_directory, masks_directory,
                                     transform=train_transform, )

    val_transform = A.Compose(
        [A.Resize(256, 256), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
    )
    val_dataset = OxfordPetDataset(val_images_filenames, images_directory, masks_directory, transform=val_transform)
    return train_dataset, val_dataset


def train_model(train_images_filenames, val_images_filenames, test_images_filenames):
    train_dataset, val_dataset = define_train_Params(train_images_filenames, val_images_filenames)

    # 修改train_transform
    random.seed(42)
    # visualize_augmentations(train_dataset, idx=55)

    model = create_model(params)
    model = train_and_validate(model, train_dataset, val_dataset, params)

    f = open(loss_path, "a")
    # a->追加读写

    f.write("train : " + str(loss_train_item) + "\n")
    f.write("validate : " + str(loss_validate_item) + "\n")

    f.close()

    return model


# 没用
def test_predict(train_images_filenames, val_images_filenames, test_images_filenames):
    # 仅用来测试predict
    model = train_model(train_images_filenames, val_images_filenames, test_images_filenames)
    test_transform = A.Compose(
        [A.Resize(256, 256), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
    )
    test_dataset = OxfordPetInferenceDataset(test_images_filenames, images_directory, transform=test_transform, )
    # 预测训练结果
    predictions = predict(model, params, test_dataset, batch_size=16)
    predicted_masks = []
    for predicted_256x256_mask, original_height, original_width in predictions:
        full_sized_mask = A.resize(
            predicted_256x256_mask, height=original_height, width=original_width, interpolation=cv2.INTER_NEAREST
        )
        predicted_masks.append(full_sized_mask)
    print(len(predicted_masks))

    display_image_grid(test_images_filenames, images_directory, masks_directory, predicted_masks=predicted_masks)


def image_segmentation():
    train_images_filenames, val_images_filenames, test_images_filenames = div_dataset()
    model = None
    if os.path.exists(weight_path):
        model = torch.load(weight_path)
        print("successfully")
    else:
        model = train_model(train_images_filenames, val_images_filenames, test_images_filenames)
    test_transform = A.Compose(
        [A.Resize(256, 256), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
    )
    test_dataset = OxfordPetInferenceDataset(test_images_filenames, images_directory, transform=test_transform, )
    # 预测训练结果
    predictions = predict(model, params, test_dataset, batch_size=16)
    predicted_masks = []
    for predicted_256x256_mask, original_height, original_width in predictions:
        full_sized_mask = A.resize(
            predicted_256x256_mask, height=original_height, width=original_width, interpolation=cv2.INTER_NEAREST
        )
        predicted_masks.append(full_sized_mask)

    image_operation(test_images_filenames, images_directory, masks_directory, predicted_masks)


# 统计iou使用的
def get_binary_white(image):
    sum = 0
    h, w = image.shape[0], image.shape[1]
    for i in range(h):
        for j in range(w):
            if image[i][j] == 1:
                sum += 1
    return sum


def image_operation(images_filenames, images_directory, masks_directory, predicted_masks):
    import shutil
    # 清空文件指定文件夹里的文件
    path = "resImg/" + use_net

    # if not os.path.exists(path):
    #     os.mkdir(path)
    # else:
    #     shutil.rmtree(path)
    #     os.mkdir(path)


    # Ious = []
    # Dice = []
    res_big_image = []
    for i, image_filename in enumerate(images_filenames):
        image = cv2.imread(os.path.join(images_directory, image_filename))

        mask = cv2.imread(os.path.join(masks_directory, image_filename.replace(".jpg", ".png")), cv2.IMREAD_UNCHANGED)
        mask = preprocess_mask(mask)

        predicted_mask = predicted_masks[i]

        # newImage = cv2.bitwise_and(image, predicted_mask)
        # cv2.imshow("newImage", newImage)

        b, g, r = cv2.split(image)
        for row in range(len(predicted_mask)):
            for col in range(len(predicted_mask[0])):
                if predicted_mask[row][col] == 0:
                    b[row][col] = 0
                    g[row][col] = 0
                    r[row][col] = 0

        res = cv2.merge([b, g, r])

        # cv2.imshow("image", image)
        # cv2.imshow("mask", mask)
        # cv2.imshow("predicted_mask", predicted_mask)
        # cv2.imshow("res", res)

        res_big_image.append(image)
        res_big_image.append(mask)
        res_big_image.append(predicted_mask)
        res_big_image.append(res)

        figure, ax = plt.subplots(nrows=1, ncols=4, figsize=(18, 10))
        ax.ravel()[0].imshow(image[:, :, [2, 1, 0]])
        ax.ravel()[0].set_title("(a)")
        ax.ravel()[1].imshow(mask)
        ax.ravel()[1].set_title("(b)")
        ax.ravel()[2].imshow(predicted_mask)
        ax.ravel()[2].set_title("(c)")
        ax.ravel()[3].imshow(res[:, :, [2, 1, 0]])
        ax.ravel()[3].set_title("(d)")
        plt.tight_layout()
        plt.savefig("./"+path+"/"+ image_filename)
        plt.show()

        # print(predicted_mask/mask)

        # intersection = cv2.bitwise_and(mask, predicted_mask)
        # print(len(intersection))
        # cv2.imshow("intersection", intersection)

        # union = cv2.bitwise_or(mask, predicted_mask)
        # print(len(union))
        # cv2.imshow("union", union)

        # Ious.append(get_binary_white(intersection) / get_binary_white(union))

        # Dice.append(2 * get_binary_white(intersection) / (get_binary_white(predicted_mask) + get_binary_white(mask)))

        # res = res[:, :, [0, 1, 2]]

        # cv2.imwrite("resImg/" + image_filename, res)

    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # avg_iou = 0
    # avg_dice = 0
    # for i in Ious:
    #     avg_iou += i
    # for i in Dice:
    #     avg_dice += i
    # avg_iou /= len(Ious)
    # avg_dice /= len(Dice)
    # # print(avg_iou)
    # print(avg_dice)


if __name__ == '__main__':
    image_segmentation()
