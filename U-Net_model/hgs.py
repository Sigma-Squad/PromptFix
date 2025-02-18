import torch
import torch.nn.functional as F

def high_freq_loss(gt, pred):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sobel_y = sobel_x.transpose(2, 3)
    
    if torch.cuda.is_available():
        sobel_x, sobel_y = sobel_x.cuda(), sobel_y.cuda()

    fx_gt = F.conv2d(gt, sobel_x) + F.conv2d(gt, sobel_y)
    fx_pred = F.conv2d(pred, sobel_x) + F.conv2d(pred, sobel_y)

    return torch.norm(fx_gt - fx_pred, p=2)
