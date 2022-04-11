import pandas as pd
import numpy as np
import torch
import utils
import optuna

DEVICE = 'cuda'
EPOCHS = 100


def run_training(fold, params, save_model=False):
    df = pd.read_csv('../input/train_features.csv')
    df = df.drop(['cp_time', 'cp_type', 'cp_dose'], axis=1)

    targets_df = pd.read_csv('../input/train_targets_folds.csv')
    feature_columns = df.drop('sig_id', axis=1).columns
    target_columns = targets_df.drop(['sig_id', 'kfold'], axis=1).columns

    df = df.merge(targets_df, on='sig_id', how='left')

    train_df = df[df.kfold != fold].reset_index(drop=True)
    val_df = df[df.kfold == fold].reset_index(drop=True)

    xtrain = train_df[feature_columns].to_numpy()
    ytrain = train_df[target_columns].to_numpy()

    xval = val_df[feature_columns].to_numpy()
    yval = val_df[target_columns].to_numpy()

    train_dataset = utils.MoaDataset(features=xtrain, targets=ytrain)
    val_dataset = utils.MoaDataset(features=xval, targets=yval)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=4)

    model = utils.Model(
        nfeatures=xtrain.shape[1],
        ntargets=ytrain.shape[1],
        nlayers=params['nlayers'],
        hidden_size=params['hidden_size'],
        dropout=params['dropout']
    )

    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    eng = utils.Engine(model, optimizer, DEVICE)
    best_loss = np.inf
    early_stopping_iter = 10
    early_stopping_counter = 0

    for epoch in range(EPOCHS):
        train_loss = eng.train(train_loader)
        val_loss = eng.val(val_loader)
        print(f'Epoch: {epoch+1:03d}/{EPOCHS:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        if val_loss < best_loss:
            best_loss = val_loss
            if save_model:
                torch.save(model.state_dict(), f'model_{fold}.bin')
        else:
            early_stopping_counter += 1

        if early_stopping_counter > early_stopping_iter:
            print('Early stopping!')
            break
    return best_loss


def objective(trial):
    params = {'nlayers': trial.suggest_int('nlayers', 1, 7),
              'hidden_size': trial.suggest_int('hidden_size', 16, 2048),
              'dropout': trial.suggest_uniform('dropout', 0.0, 0.5),
              'lr': trial.suggest_loguniform('lr', 1e-5, 1e-1)}
    al_losses = []
    for f_ in range(5):
        temp_loss = run_training(f_,params, save_model=False)
        al_losses.append(temp_loss)
    return np.mean(al_losses)


if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    print('best trail:')
    trial_ = study.best_trial
    print(trial_.values)
    print(trial_.params)

    scores = 0
    for j in range(5):
        scr = run_training(j, trial_.params, save_model=True)
        scores += scr
    print(f'Average score: {scores/5}')
