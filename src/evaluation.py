from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

def save_confusion_matrix(y_true, y_pred):
    os.makedirs("../results/plots", exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    
    plt.savefig("../results/plots/confusion_matrix.png")
    print(" Confusion matrix saved")