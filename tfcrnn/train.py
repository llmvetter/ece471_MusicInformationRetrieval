from tfcrnn.runners import FMARunner
from tfcrnn.config import Config
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def main():
  config = Config()
  config.parse_cli()
  config.print()
  
  runner = FMARunner(config)
  
  print(runner.model)
  print(f'\n=> Num params: {sum([p.numel() for p in runner.model.parameters()]):,}')
  # Adding history for plotting
  history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
  
  epoch = 0
  for epoch in range(epoch + 1, config.num_max_epochs + 1):
    print(f'Epoch {epoch:2}')
    loss_train, scores_train = runner.train()
    loss_valid, scores_valid = runner.validate()

    history["train_loss"].append(loss_train)
    history["val_loss"].append(loss_valid)
    history["train_acc"].append(scores_train["score"])
    history["val_acc"].append(scores_valid["score"])
    
    print(f'Train loss: {loss_train}, Train Accuracy: {scores_train}')
    print(f'Eval loss: {loss_valid}, Train Accuracy: {scores_valid}')
  
  print('-' * 80)
  print(f'Training finished after {config.num_max_epochs} epochs.')
  print('-' * 80)

  plt.figure(figsize=(10, 4))

  # 1. Loss plot
  plt.subplot(1, 2, 1)
  plt.plot(history["train_loss"], label="Train Loss", marker="o")
  plt.plot(history["val_loss"], label="Validation Loss", marker="o")
  plt.title("Training & Validation Loss")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend()
  plt.grid(True, linestyle="--", alpha=0.6)

  # 2. Accuracy plot
  plt.subplot(1, 2, 2)
  plt.plot(history["train_acc"], label="Train Accuracy", marker="o")
  plt.plot(history["val_acc"], label="Validation Accuracy", marker="o")
  plt.title("Training & Validation Accuracy")
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy")
  plt.legend()
  plt.grid(True, linestyle="--", alpha=0.6)

  plt.tight_layout()
  plt.savefig("training_curves.png", dpi=150)
  plt.show()

  print('Running final evaluation on the Test set...')
  loss_test, scores_test = runner.eval(runner.loader_test)
  
  final_scores = {
      'final_loss_test': loss_test,
      **{f'final_{k}_test': v for k, v in scores_test.items()}
  }
  print('\n=> Final Test Results:')
  for k, v in final_scores.items():
      print(f'=> {k:12}: {v:.4f}')
  print('=> Done.')

  print("\nAnalyzing test predictions...")
  
  # Get model outputs + filenames
  model = runner.model.eval().to(runner.device)
  correct_examples, incorrect_examples = [], []
  all_preds, all_labels = [], []
  
  with torch.no_grad():
      for i, (x, y) in enumerate(runner.loader_test):
          x = x.to(runner.device)
          y = y.to(runner.device)
          outputs = model(x)
        
          if isinstance(outputs, tuple):
              outputs = outputs[0]

          if outputs.ndim == 3:
              outputs = outputs.mean(dim=1)
          
          preds = torch.argmax(outputs, dim=1)
        
          all_preds.extend(preds.cpu().numpy())
          all_labels.extend(y.cpu().numpy())
  
          batch_start = i * runner.config.batch_size
          for j in range(len(y)):
              file_idx = batch_start + j
              if file_idx >= len(runner.dataset_test.paths):
                  break  # prevent overflow
              file_path = runner.dataset_test.paths[file_idx]
              true_label = y[j].item()
              pred_label = preds[j].item()

              if true_label == pred_label:
                  correct_examples.append((file_path, true_label, pred_label))
              else:
                  incorrect_examples.append((file_path, true_label, pred_label))
  
  # Print a few examples
  print("\nCorrect predictions:")
  for path, t, p in correct_examples[:5]:
      print(f"  {path.split('/')[-1]} | True: {t}, Pred: {p}")
  
  print("\nIncorrect predictions:")
  for path, t, p in incorrect_examples[:5]:
      print(f"  {path.split('/')[-1]} | True: {t}, Pred: {p}")
  print("\nGenerating confusion matrix...")

  # Compute confusion matrix
  cm = confusion_matrix(all_labels, all_preds)
  cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # normalize by row

  # Get label names if available
  if hasattr(runner, 'name2idx'):
      labels = list(runner.name2idx.keys())
  else:
      labels = [f'Class {i}' for i in range(cm.shape[0])]

  # Plot confusion matrix
  fig, ax = plt.subplots(figsize=(8, 6))
  disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels)
  disp.plot(cmap='Blues', ax=ax, colorbar=False)
  plt.title("Normalized Confusion Matrix")
  plt.xticks(rotation=45, ha='right')
  plt.tight_layout()
  plt.savefig("confusion_matrix.png", dpi=150)
  plt.show()

if __name__ == '__main__':
  main()
