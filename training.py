import tqdm

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

def train_model(model, training_step_fn, trainloader, validloader=None, epochs=2, optimizer=None, print_every=1, on_epoch_end=lambda x: x):
    
    alpha = 0.9

    if optimizer is None:
        assert hasattr(model, 'optimizer'), 'Please provide an optimizer either as argument or by setting model.optimizer = ...'

        optimizer = model.optimizer

    for epoch in range(epochs):

        running_loss = 0.0
        running_accuracy = 0.0
        epoch_loss = 0.0
        epoch_correct = 0.0
        epoch_samples = 0
        
        model.train()
        
        for i, batch in enumerate(trainloader):
            optimizer.zero_grad()
            
            num_samples, num_correct_samples, summed_loss = training_step_fn(model, batch)
            
            mean_loss = summed_loss/num_samples
            mean_loss.backward()
            optimizer.step()
            
            epoch_loss += summed_loss.item()
            epoch_correct += num_correct_samples.item()
            epoch_samples += num_samples
            
            accuracy = num_correct_samples/num_samples

            if i == 0:
                running_loss = mean_loss.item()
                running_accuracy = accuracy.item()
            else:
                running_loss = running_loss * alpha + (1-alpha) * mean_loss.item()
                running_accuracy = running_accuracy * alpha + (1-alpha) * accuracy.item()
                
            if i%print_every == print_every-1:
                print(f'Epoch {epoch}, iteration {i}/{len(trainloader)}: running loss {running_loss:.4f}, running accuracy {running_accuracy:.4f}')
        
        print(f'Epoch {epoch}: training loss {epoch_loss/epoch_samples:.4f}, training accuracy {epoch_correct/epoch_samples:.4f}')
        
        if validloader is not None:
            val_loss, val_accuracy = evaluate_model(model, validloader, training_step_fn)
            print(f'Validation accuracy {val_accuracy:.4f}, loss {val_loss:.4f}\n')
        else:
            print()

        on_epoch_end(epoch)

    print('Finished Training')