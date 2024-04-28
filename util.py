import seaborn as sns
import matplotlib.pyplot as plt

def show_train_validation_loss(train_losses, valid_losses, filename):
    plt.figure(figsize=(10, 8))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("images/" + filename)



def show_confusion_matrix(cm, y_true, y_pred, filename):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=set(y_true), yticklabels=set(y_true))
    plt.title('Confusion Matrix')
    plt.xlabel('True Labels')
    plt.ylabel('Predicted Labels')
    plt.savefig("images" + filename)
