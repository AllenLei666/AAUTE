import torch

def regression_loss(y_pred,y_true):
    # print(y_true.shape,y_pred.shape)
    mse_loss = torch.nn.L1Loss()
    mse_loss1 = mse_loss(y_pred[:,:3],y_true[:,:3])
    total_loss = mse_loss1 
    return total_loss



def regression_loss_w_pseudo(y_pred,y_true, y_pseudo, alpha):
    # print(y_true.shape,y_pred.shape)
    mse_loss = torch.nn.L1Loss()
    mse_loss1 = mse_loss(y_pred[:,:3],y_true[:,:3])
    mse_loss2 = mse_loss(y_pred[:,:3],y_pseudo[:,:3])
    total_loss = alpha * mse_loss1 + (1 - alpha) * mse_loss2
    return total_loss