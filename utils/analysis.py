import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def Dice(pred, label):
    pred = pred.flatten().astype(np.float32)
    label = label.flatten().astype(np.float32)
    intersection = np.sum(pred * label)

    eps = 0.0001
    return (2. * intersection + eps) / (np.sum(pred) + np.sum(label) + eps)


def Error_Visualization(pred, label):
    False_Negative = ((pred == 0) & (label == 1)).astype(np.uint8)  # 마스크 안 에러 (1이어야 하는데 0인 값)
    False_Positive = ((pred == 1) & (label == 0)).astype(np.uint8)  # 마스크 바깥 에러 (0이어야 하는데 1인 값)

    Confusion_Mat_Norm = confusion_matrix(
        y_true=label.flatten(),
        y_pred=pred.flatten(),
        normalize='true'
    )
    
    Confusion_Mat = confusion_matrix(
        y_true=label.flatten(),
        y_pred=pred.flatten()
    )

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(False_Negative, cmap='gray')
    ax[0].set_title("pixels that should be 1 (FN)")

    ax[1].imshow(False_Positive, cmap='gray')
    ax[1].set_title("pixels that should be 0 (FP)")

    print("Confusion Matrix (Normalized):")
    print(Confusion_Mat_Norm)
    
    print("Confusion Matrix:")
    print(Confusion_Mat)