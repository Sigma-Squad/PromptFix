import torch
import torch.nn.functional as F

def high_freq_loss(gt, pred):
    B, C, H, W = gt.shape  # Ensure correct shapes
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sobel_y = sobel_x.transpose(2, 3)
    
    sobel_x = sobel_x.repeat(C, 1, 1, 1)  # Match channel dimensions
    sobel_y = sobel_y.repeat(C, 1, 1, 1)

    if torch.cuda.is_available():
        sobel_x, sobel_y = sobel_x.cuda(), sobel_y.cuda()

    fx_gt = F.conv2d(gt, sobel_x, groups=C) + F.conv2d(gt, sobel_y, groups=C)
    fx_pred = F.conv2d(pred, sobel_x, groups=C) + F.conv2d(pred, sobel_y, groups=C)

    return torch.norm(fx_gt - fx_pred, p=2)

