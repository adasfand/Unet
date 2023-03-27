import cv2

from model import *
from predict import *
from show import *
cudnn.benchmark = True

# method1
params = {
    "model": "UNet11",
    "device": "cuda",
    "lr": 0.001,
    "batch_size": 8,
    "num_workers": 0,
    "epochs": 1,
}

def train(train_loader, model, criterion, optimizer, epoch, params):
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)
    for i, (images, target) in enumerate(stream, start=1):
        images = images.to(params["device"], non_blocking=True)
        target = target.to(params["device"], non_blocking=True)
        output = model(images).squeeze(1)
        loss = criterion(output, target)
        metric_monitor.update("Loss", loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        stream.set_description(
            "Epoch: {epoch}. Train.      {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
        )

def validate(val_loader, model, criterion, epoch, params):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(params["device"], non_blocking=True)
            target = target.to(params["device"], non_blocking=True)
            output = model(images).squeeze(1)
            loss = criterion(output, target)
            metric_monitor.update("Loss", loss.item())
            stream.set_description(
                "Epoch: {epoch}. Validation. {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
            )

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
    return model

def visualize_augmentations(dataset, idx=0, samples=5):# 可视化
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
    val_dataset = OxfordPetDataset(val_images_filenames, images_directory, masks_directory, transform=val_transform, )
    return train_dataset, val_dataset

def train_model():
    train_images_filenames, val_images_filenames, test_images_filenames = div_dataset()
    train_dataset, val_dataset = define_train_Params(train_images_filenames, val_images_filenames)
    # 修改train_transform
    random.seed(42)
    # visualize_augmentations(train_dataset, idx=55)

    model = create_model(params)
    model = train_and_validate(model, train_dataset, val_dataset, params)
    return model, train_images_filenames, val_images_filenames, test_images_filenames

def test_predict():
    model, train_images_filenames, val_images_filenames, test_images_filenames = train_model()
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
    # print(len(predicted_masks))

    display_image_grid(test_images_filenames, images_directory, masks_directory, predicted_masks=predicted_masks)

def image_segmentation():
    model, train_images_filenames, val_images_filenames, test_images_filenames = train_model()
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

def image_operation(images_filenames, images_directory, masks_directory, predicted_masks):
    res_image_seg = []
    for i, image_filename in enumerate(images_filenames):
        image = cv2.imread(os.path.join(images_directory, image_filename))

        mask = cv2.imread(os.path.join(masks_directory, image_filename.replace(".jpg", ".png")), cv2.IMREAD_UNCHANGED, )
        mask = preprocess_mask(mask)

        predicted_mask = predicted_masks[i]
        # predicted_mask = cv2.cvtColor(predicted_masks[i], cv2.)

        # print(type(image), type(predicted_mask))
        # print(image.shape, predicted_mask.shape)
        # print(len(image), len(predicted_mask))

        # newImage = cv2.bitwise_and(image, predicted_mask)
        # cv2.imshow("newImage", newImage)

        newImage = np.zeros(image.shape[:2], dtype='uint8')

        b, g, r = cv2.split(image)
        for row in range(len(predicted_mask)):
            for col in range(len(predicted_mask[0])):
                if predicted_mask[row][col] == 0:
                    b[row][col] = 0
                    g[row][col] = 0
                    r[row][col] = 0

        res = cv2.merge([b, g, r, newImage])

        res_image_seg.append(res)
        cv2.imshow("image", image)
        cv2.imshow("mask", mask)
        cv2.imshow("predicted_mask", predicted_mask)
        cv2.imshow("res", res)

        print(predicted_masks[i])
        cv2.waitKey(0)
    cv2.destroyAllWindows()

    return res_image_seg

if __name__ == '__main__':
    image_segmentation()
