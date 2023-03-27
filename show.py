from dataset import *

# 该函数是展示测试集里面的图片以及对应的mask
def display_image_grid(images_filenames, images_directory, masks_directory, predicted_masks=None):
    cols = 3 if predicted_masks else 2
    rows = len(images_filenames)
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 24))
    for i, image_filename in enumerate(images_filenames):
        image = cv2.imread(os.path.join(images_directory, image_filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        mask = cv2.imread(os.path.join(masks_directory, image_filename.replace(".jpg", ".png")), cv2.IMREAD_UNCHANGED, )
        mask = preprocess_mask(mask)
        ax[i, 0].imshow(image)
        ax[i, 1].imshow(mask, interpolation="nearest")

        ax[i, 0].set_title("Image")
        ax[i, 1].set_title("Ground truth mask")

        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()

        if predicted_masks:
            predicted_mask = predicted_masks[i]
            ax[i, 2].imshow(predicted_mask, interpolation="nearest")
            ax[i, 2].set_title("Predicted mask")
            ax[i, 2].set_axis_off()

    plt.tight_layout()
    plt.show()

# 显示不同类型的图像的外观
def show_diff_img():
    example_image_filename = correct_images_filenames[0]
    image = cv2.imread(os.path.join(images_directory, example_image_filename))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    resized_image = A.resize(image, height=256, width=256)
    padded_image = A.pad(image, min_height=512, min_width=512)
    padded_constant_image = A.pad(image, min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT)
    cropped_image = A.center_crop(image, crop_height=256, crop_width=256)

    figure, ax = plt.subplots(nrows=1, ncols=5, figsize=(18, 10))
    ax.ravel()[0].imshow(image)
    ax.ravel()[0].set_title("Original image")
    ax.ravel()[1].imshow(resized_image)
    ax.ravel()[1].set_title("Resized image")
    ax.ravel()[2].imshow(cropped_image)
    ax.ravel()[2].set_title("Cropped image")
    ax.ravel()[3].imshow(padded_image)
    ax.ravel()[3].set_title("Image padded with reflection")
    ax.ravel()[4].imshow(padded_constant_image)
    ax.ravel()[4].set_title("Image padded with constant padding")
    plt.tight_layout()
    # 保存save即可
    plt.show()

if __name__ == '__main__':
    train_images_filenames, val_images_filenames, test_images_filenames = div_dataset()
    display_image_grid(test_images_filenames, images_directory, masks_directory)
    show_diff_img()