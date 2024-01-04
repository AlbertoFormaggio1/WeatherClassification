import torch
from torch.utils.data import DataLoader
from typing import Dict, List
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          metrics: dict,
          writer: SummaryWriter,
          model_name: str,
          logdir: str,
          lr_schedule = None) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.


    :param model: A PyTorch model to be trained and tested.
    :param train_dataloader: A DataLoader instance for the model to be trained on.
    :param test_dataloader: A DataLoader instance for the model to be tested on.
    :param optimizer: A PyTorch optimizer to help minimize the loss function.
    :param loss_fn: A PyTorch loss function to calculate loss on both datasets.
    :param epochs: An integer indicating how many epochs to train for.
    :param device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for
    each epoch.
    In the form: {train_loss: [...],
                  train_acc: [...],
                  test_loss: [...],
                  test_acc: [...]}
    For example if training for epochs=2:
                 {train_loss: [2.0616, 1.0537],
                  train_acc: [0.3945, 0.3945],
                  test_loss: [1.2641, 1.5706],
                  test_acc: [0.3400, 0.2973]}
  """
    # Create empty results dictionary
    results = {"train_metrics": [],
               "test_metrics": [],
               }

    best_val_loss = float('inf')

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):

        # Freeze the first part of the network for the first 15 epochs and then train the full model
        if epoch == 0:
            for child in model.children():
                for param in child.parameters():
                    param.requires_grad = False
                break
        elif epoch == 15:
            for child in model.children():
                for param in child.parameters():
                    param.requires_grad = True
                break

        train_res = train_step(model=model,
                               dataloader=train_dataloader,
                               loss_fn=loss_fn,
                               optimizer=optimizer,
                               device=device,
                               metrics=metrics)
        test_res = test_step(model=model,
                             dataloader=test_dataloader,
                             loss_fn=loss_fn,
                             device=device,
                             metrics=metrics)

        # If the evaluation accuracy improved, save the model
        if test_res['loss'] < best_val_loss:
            best_val_loss = test_res['loss']
            torch.save(model.state_dict(), f'{logdir}/model.pth')

        # Take the sceduler to the next step
        if lr_schedule is not None:
            lr_schedule.step(test_res['accuracy'])

        # Print out what's happening
        print(f"Epoch: {epoch + 1} | Train: {train_res} | Test: {test_res}")
        for k, v in train_res.items():
            writer.add_scalar(f'{model_name}/{k}/train', v, epoch)
        for k, v in test_res.items():
            writer.add_scalar(f'{model_name}/{k}/eval', v, epoch)

        # Update results dictionary
        results["train_metrics"].append(train_res)
        results["test_metrics"].append(test_res)

    # Return the filled results at the end of the epochs
    return results


def test_single_ds(model: torch.nn.Module, dataloader: DataLoader, device: torch.device,
         metrics: Dict, label_mapping: Dict, class_num):
    """
    Tests the model over a single dataset and returns the value for the metrics given in input
    :param model: The model usable for inference
    :param dataloader: The dataloader to use for computing the predictions and to get the true labels
    :param device: A target device to compute on (e.g. "cuda" or "cpu").
    :param metrics: A dictionary of metrics to evaluate the test set
    :param label_mapping: The label mapping for the test set (used for converting from the model labels to the dataset labels)
    :param class_num: number of classes
    :return: a dictionary with the computed metrics
    """

    # get the predictions of the model over the dataloader provided in input
    predictions = do_inference(model, dataloader, device, label_mapping, class_num)

    # Convert the true labels from the model version to the dataset one
    truths = torch.cat([y for (X, y) in dataloader], dim=0)
    truths_copy = torch.clone(truths)
    for (old_l, new_l) in label_mapping.items():
        truths_copy[truths == old_l] = new_l
    truths = truths_copy

    # Create a dictionary counting the errors for each true class
    errors_class = {}
    for c in label_mapping.values():
        errors_class[c] = 0

    # For each sample, check if there was a mistake in the prediction. If there was, increase the counter of the corresponding class
    for idx, (y_pred, y_true) in enumerate(zip(predictions, truths)):
        if y_pred.item() != y_true.item():
            errors_class[y_true.item()] += 1

    # Compute the metrics
    res = {}

    for k, v in metrics.items():
        if k != 'accuracy':
            score = v.compute(predictions=predictions, references=truths, average='macro')
        else:
            score = v.compute(predictions=predictions, references=truths)
        res[k] = score[k]

    # Compute the confusion matrix
    conf_mat = confusion_matrix(truths, predictions)
    conf_mat_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)

    return res, errors_class, conf_mat_disp

def do_inference(model: torch.nn.Module, dataloader: DataLoader, device: torch.device, label_mapping, class_num):
    """
    Performs the inference on the given model by running on the dataloader provided in input
    :param model: The model to perform inference on
    :param dataloader: The dataloader to perform inference on
    :param device: The device 'cpu' or 'cuda'
    :param label_mapping: the mapping from model labels to dataset labels
    :return: the inference in the label setting of the model
    """
    # Extract the indices of the classes that are not used for this dataset: they will be removed from the possible
    # predictions that the model can do.
    label_ids_to_remove = torch.tensor([l for l in range(0, class_num) if l not in label_mapping.keys()])

    model.eval() # Set the model to evaluation stage
    model = model.to(device)
    full_preds = torch.tensor([])

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            logits = model(X)

            # Transformers model do not return tensors in output, logits are inside model(X).logits
            if not torch.is_tensor(logits):
                logits = logits.logits

            # Setting the logits to small value so that the softmax doesn't predict it
            if label_ids_to_remove.shape[0] != 0:
                logits[:, label_ids_to_remove] = -1000

            # Run the softmax only over the classes belonging to the current dataset
            preds = torch.argmax(torch.softmax(logits, dim=1), dim=1).to('cpu')
            final_preds = torch.clone(preds)

            # convert the predictions to the labeling used for the dataset
            for idx, p in enumerate(preds):
                final_preds[idx] = label_mapping[p.item()]

            # Concatenate the predictions of all the batches in a unique vector
            full_preds = torch.cat([full_preds, final_preds])

    return full_preds.to('cpu')


def test_step(model: torch.nn.Module,
              dataloader: DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device,
              metrics: dict):
    """Tests a PyTorch model for a single epoch.

        Turns a target PyTorch model to "eval" mode and then performs
        a forward pass on a testing dataset.

        :param model: A PyTorch model to be tested.
        :param dataloader: A DataLoader instance for the model to be tested on.
        :param loss_fn: A PyTorch loss function to calculate loss on the test data.
        :param device: A target device to compute on (e.g. "cuda" or "cpu").

        Returns:
        A tuple of testing loss and testing accuracy metrics.
        In the form (test_loss, test_accuracy). For example:

        (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    res = {'loss': 0, 'accuracy': 0, 'f1': 0, 'precision': 0, 'recall': 0}

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)
            if not torch.is_tensor(test_pred_logits):
                test_pred_logits = test_pred_logits.logits

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            res['loss'] += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = torch.argmax(torch.softmax(test_pred_logits, dim=1), dim=1)
            for k, v in metrics.items():
                if k != 'accuracy':
                    score = v.compute(predictions=test_pred_labels, references=y, average='macro')
                else:
                    score = v.compute(predictions=test_pred_labels, references=y)
                res[k] += score[k]

    # Adjust metrics to get average loss and accuracy per batch
    for k, v in res.items():
        res[k] = res[k] / len(dataloader)

    return res


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               metrics: dict):
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    :param model: A PyTorch model to be trained.
    :param dataloader: A DataLoader instance for the model to be trained on.
    :param loss_fn: A PyTorch loss function to minimize.
    :param optimizer: A PyTorch optimizer to help minimize the loss function.
    :param device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    res = {'loss': 0, 'accuracy': 0, 'f1': 0, 'precision': 0, 'recall': 0}

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)
        if not torch.is_tensor(y_pred):
            y_pred = y_pred.logits

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        res['loss'] += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        for k, v in metrics.items():
            if k != 'accuracy':
                score = v.compute(predictions=y_pred_class, references=y, average='macro')
            else:
                score = v.compute(predictions=y_pred_class, references=y)
            res[k] += score[k]

    # Adjust metrics to get average loss and accuracy per batch
    for k, v in res.items():
        res[k] = res[k] / len(dataloader)

    return res
