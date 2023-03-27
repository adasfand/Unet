from train import *

if __name__ == '__main__':
    image = cv2.imread("image.jpg")
    predicted_mask = cv2.imread("predict.jpg")
    cv2.cvtColor(predicted_mask, cv2.COLOR_BGR2RGB)

    cv2.imshow("image", image)
    cv2.imshow("predicted_mask", predicted_mask)

    newImage = cv2.bitwise_and(image, predicted_mask)
    cv2.imshow("newImage", newImage)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    改进：修改使用自己写的Unet网络,写一个read_file函数，
    最后最重要的是将predict函数得到的预测二值图与原图进行交运算得到分割结果
    """