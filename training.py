import tqdm
import torch
import time

def evaluate_model(model, loader, training_step_fn):
    model.eval()
    
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0
        
    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm(loader, position=0, leave=True)):
            num_samples, num_correct_samples, summed_loss = training_step_fn(model, batch)
            
            total_loss += float(summed_loss)
            total_correct += float(num_correct_samples)
            total_samples += float(num_samples)
            
    return total_loss/total_samples, total_correct/total_samples

def train_model(model,
    training_step_fn,
    trainloader,
    validloader=None,
    epochs=2,
    optimizer=None,
    print_every=1,
    on_epoch_end=lambda x: x,
    save_best_to=False,
    eval_every=None
    ):
    
    alpha = 0.9

    if optimizer is None:
        assert hasattr(model, 'optimizer'), 'Please provide an optimizer either as argument or by setting model.optimizer = ...'
        optimizer = model.optimizer
        
    total_steps = 0

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):

        running_loss = 0.0
        running_accuracy = 0.0
        running_batch_time = 0.0
        running_loader_time = 0.0
        epoch_loss = 0.0
        epoch_correct = 0.0
        epoch_samples = 0
        
        model.train()
        
        t0 = time.time()
        for i, batch in enumerate(trainloader):
            
            if eval_every is not None and (i+1)%eval_every == 0:
                valid_loss, valid_accuracy = evaluate_model(model, validloader, training_step_fn)
                print(f'Validation accuracy {valid_accuracy:.4f}, loss {valid_loss:.4f}\n')
                val_losses.append(valid_loss)
                val_accuracies.append(valid_accuracy)
            
            if i == 0:
                running_loader_time = time.time() - t0
            else:
                running_loader_time = running_loader_time * alpha + (1-alpha) * (time.time() - t0)
            
            t0 = time.time()
            
            optimizer.zero_grad()
            
            num_samples, num_correct_samples, summed_loss = training_step_fn(model, batch)
            
            mean_loss = summed_loss/num_samples
            mean_loss.backward()
            optimizer.step()
            
            epoch_loss += summed_loss.item()
            epoch_correct += num_correct_samples.item()
            epoch_samples += num_samples
            
            accuracy = num_correct_samples/num_samples

            train_losses.append(mean_loss.item())
            train_accuracies.append(accuracy.item())

            if i == 0:
                running_loss = mean_loss.item()
                running_accuracy = accuracy.item()
                running_batch_time = time.time() - t0
            else:
                running_loss = running_loss * alpha + (1-alpha) * mean_loss.item()
                running_accuracy = running_accuracy * alpha + (1-alpha) * accuracy.item()
                running_batch_time = running_batch_time * alpha + (1-alpha) * (time.time() - t0)
                
            if i%print_every == print_every-1:
                print(f'Epoch {epoch}, iteration {i+1}/{len(trainloader)}: running loss {running_loss:.4f}, running acc {running_accuracy:.4f}, batch loss: {mean_loss.item():.4f}, batch acc: {accuracy.item():.4f}, running time per batch: {running_batch_time:.4f}s (model) {running_loader_time:.4f}s (loader)')
        
            t0 = time.time()
        
        train_loss = epoch_loss/epoch_samples
        train_accuracy = epoch_correct/epoch_samples
        print(f'Epoch {epoch}: training loss {train_loss:.4f}, training accuracy {train_accuracy:.4f}')
        
        if validloader is not None and eval_every is None:
            valid_loss, valid_accuracy = evaluate_model(model, validloader, training_step_fn)
            print(f'Validation accuracy {valid_accuracy:.4f}, loss {valid_loss:.4f}\n')
            val_losses.append(valid_loss)
            val_accuracies.append(valid_accuracy)

        on_epoch_end({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'valid_loss': None if validloader is None else valid_loss,
            'valid_accuracy': None if validloader is None else valid_accuracy
        })

    print('Finished Training')

    return train_losses, train_accuracies, val_losses, val_accuracies




