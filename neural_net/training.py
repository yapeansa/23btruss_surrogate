import torch
import time as time
from neural_net.loss_functions import fem_residual_loss

def train_with_loader(model, dataloader_train, dataloader_test, l_rate, epochs=20000, early_stop=True):
    begin = time.time()
    print("training begins")
    optimizer = torch.optim.Adam(model.parameters(), lr=l_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.5)
    
    sum_loss_train, sum_loss_test = 0.0, 0.0

    for epoch in range(epochs):
        model.train()
        sum_loss_train = 0.0
        for d, k, f in dataloader_train:
            prediction = model(d)
            loss = fem_residual_loss(prediction, k, f)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss_train += loss.item()

        avg_train_loss = sum_loss_train / len(dataloader_train)
        
        model.eval()
        sum_loss_test = 0.0
        with torch.inference_mode(): # Desativa gradientes para poupar memória
            for d, k, f in dataloader_test:
                prediction = model(d)
                loss_t = fem_residual_loss(prediction, k, f)
                sum_loss_test += loss_t.item()

        avg_test_loss = sum_loss_test / len(dataloader_test)
        scheduler.step(avg_test_loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.5E} | Test Loss = {avg_test_loss:.5E}")
        
        # earlystop logic
        if early_stop:
            if epoch == 0:
                    best_test_loss = float('inf')
                    patience_counter = 0
                
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                patience_counter = 0
                torch.save(model.state_dict(), 'models/best_model.pt')
            else:
                patience_counter += 1
            
            if patience_counter >= 100:  # patience threshold
                print(f"Early stopping at epoch {epoch}")
                break

    end = time.time()

    print("training ends")
    print("Total time for training is", end-begin)

    return sum_loss_train/len(dataloader_train), sum_loss_test/len(dataloader_test)
