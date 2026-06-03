import os
import sys
import random
import clip

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import cv2

from scripts.losses import fusion_prompt_loss

vi_prompt_path = "./datasets/train/text_vi.txt"
assert os.path.exists(vi_prompt_path), "text prompt root: {} does not exist.".format(vi_prompt_path)
with open(vi_prompt_path, 'r', encoding='utf-8') as file:
    vi_lines = file.readlines()

ir_prompt_path = "./datasets/train/text_ir.txt"
assert os.path.exists(ir_prompt_path), "text prompt root: {} does not exist.".format(ir_prompt_path)
with open(ir_prompt_path, 'r', encoding='utf-8') as file:
    ir_lines = file.readlines()

fuse_prompt_path = "./datasets/train/text_fuse.txt"
assert os.path.exists(ir_prompt_path), "text prompt root: {} does not exist.".format(fuse_prompt_path)
with open(fuse_prompt_path, 'r', encoding='utf-8') as file:
    fuse_lines = file.readlines()


def read_data(root: str):
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    train_root = os.path.join(root, "train")
    val_root = os.path.join(root, "eval")
    assert os.path.exists(train_root), "train root: {} does not exist.".format(train_root)
    assert os.path.exists(val_root), "val root: {} does not exist.".format(val_root)

    train_images_visible_path = []
    train_images_infrared_path = []
    train_images_visible_gt_path = []
    train_images_infrared_gt_path = []
    val_images_visible_path = []
    val_images_infrared_path = []

    supported = [".jpg", ".JPG", ".png", ".PNG", ".bmp", 'tif', 'TIF']  # 支持的文件后缀类型

    train_visible_root = os.path.join(train_root, "Visible")
    train_infrared_root = os.path.join(train_root, "Infrared")

    train_visible_gt_root = os.path.join(train_root, "Visible_gt")
    train_infrared_gt_root = os.path.join(train_root, "Infrared_gt")

    val_visible_root = os.path.join(val_root, "Visible")
    val_infrared_root = os.path.join(val_root, "Infrared")

    train_visible_path = [os.path.join(train_visible_root, i) for i in os.listdir(train_visible_root)
                          if os.path.splitext(i)[-1] in supported]
    train_infrared_path = [os.path.join(train_infrared_root, i) for i in os.listdir(train_infrared_root)
                           if os.path.splitext(i)[-1] in supported]

    train_visible_gt_path = [os.path.join(train_visible_gt_root, i) for i in os.listdir(train_visible_gt_root)
                             if os.path.splitext(i)[-1] in supported]
    train_infrared_gt_path = [os.path.join(train_infrared_gt_root, i) for i in os.listdir(train_infrared_gt_root)
                              if os.path.splitext(i)[-1] in supported]

    val_visible_path = [os.path.join(val_visible_root, i) for i in os.listdir(val_visible_root)
                        if os.path.splitext(i)[-1] in supported]
    val_infrared_path = [os.path.join(val_infrared_root, i) for i in os.listdir(val_infrared_root)
                         if os.path.splitext(i)[-1] in supported]

    train_visible_path.sort()
    train_infrared_path.sort()
    train_visible_gt_path.sort()
    train_infrared_gt_path.sort()
    val_visible_path.sort()
    val_infrared_path.sort()

    assert len(train_visible_path) == len(
        train_infrared_path), ' The length of train dataset does not match. low:{}, high:{}'. \
        format(len(train_visible_path), len(train_infrared_path))
    assert len(val_visible_path) == len(
        val_infrared_path), ' The length of val dataset does not match. low:{}, high:{}'. \
        format(len(val_visible_path), len(val_infrared_path))
    print("Visible and Infrared images check finish")

    for index in range(len(train_visible_path)):
        # num1 = random.randint(1, 75)
        # num2 = random.randint(1, 75)
        img_visible_path = train_visible_path[index]
        img_infrared_path = train_infrared_path[index]
        # img_visible_path=train_visible_path[num1]
        # img_infrared_path=train_infrared_path[num2]
        train_images_visible_path.append(img_visible_path)
        train_images_infrared_path.append(img_infrared_path)

        img_visible_gt_path = train_visible_gt_path[index]
        img_infrared_gt_path = train_infrared_gt_path[index]
        # img_visible_gt_path=train_visible_gt_path[num1]
        # img_infrared_gt_path=train_infrared_gt_path[num2]
        train_images_visible_gt_path.append(img_visible_gt_path)
        train_images_infrared_gt_path.append(img_infrared_gt_path)

    for index in range(len(val_visible_path)):
        img_visible_path = val_visible_path[index]
        img_infrared_path = val_infrared_path[index]
        val_images_visible_path.append(img_visible_path)
        val_images_infrared_path.append(img_infrared_path)

    total_dataset_nums = len(train_visible_path) + len(train_infrared_path) + len(train_visible_gt_path) + len(
        train_infrared_gt_path) \
                         + len(val_visible_path) + len(val_infrared_path)
    print("{} images were found in the dataset.".format(total_dataset_nums))
    print("{} visible images for training.".format(len(train_visible_path)))
    print("{} infrared images for training.".format(len(train_infrared_path)))
    print("{} visible gt images for training.".format(len(train_visible_gt_path)))
    print("{} infrared gt images for training.".format(len(train_infrared_gt_path)))
    print("{} visible images for validation.".format(len(val_visible_path)))
    print("{} infrared images for validation.\n".format(len(val_infrared_path)))

    train_low_light_path_list = [train_visible_path, train_infrared_path, train_visible_gt_path, train_infrared_gt_path]
    val_low_light_path_list = [val_visible_path, val_infrared_path]
    return train_low_light_path_list, val_low_light_path_list


def get_vi_prompt():
    random_line = random.choice(vi_lines)
    random_line = random_line.strip()
    return random_line


def get_ir_prompt():
    random_line = random.choice(ir_lines)
    random_line = random_line.strip()
    return random_line


def get_fuse_prompt():
    random_line = random.choice(fuse_lines)
    random_line = random_line.strip()
    return random_line


def train_one_epoch(model, model_clip, optimizer, lr_scheduler, data_loader, device, epoch):
    model.train()
    model_clip.eval()
    loss_function_prompt = fusion_prompt_loss()

    if torch.cuda.is_available():
        loss_function_prompt = loss_function_prompt.to(device)

    accu_total_loss = torch.zeros(1).to(device)
    accu_ssim_loss = torch.zeros(1).to(device)
    accu_int_loss = torch.zeros(1).to(device)
    accu_text_loss = torch.zeros(1).to(device)

    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        I_A, I_B, I_A_gt, I_B_gt, _, task, _ = data
        text_vi_line = []
        text_ir_line = []
        text_fuse_line = []

        for index in range(len(task)):
            # default type degradation in vis image
            text_vi_line.append(get_vi_prompt())
            text_ir_line.append(get_ir_prompt())
            text_fuse_line.append(get_fuse_prompt())
        # text_vi_line="This is a visible image, and the focus is on texture information and color information."
        # text_ir_line="This is an infrared image, and the focus is on thermal radiation information."
        # text_fuse_line="A fused image combining visible light texture and color with infrared heat information."
        # batch_size = I_A.size(0)
        # text_vi = clip.tokenize([text_vi_line] * batch_size).to(device)
        # text_ir = clip.tokenize([text_ir_line] * batch_size).to(device)
        # text_fuse = clip.tokenize([text_fuse_line] * batch_size).to(device)

        text_vi = clip.tokenize(text_vi_line).to(device)
        text_ir = clip.tokenize(text_ir_line).to(device)
        text_fuse = clip.tokenize(text_fuse_line).to(device)

        if torch.cuda.is_available():
            I_A = I_A.to(device)
            I_B = I_B.to(device)
            I_A_gt = I_A_gt.to(device)
            I_B_gt = I_B_gt.to(device)

        I_fused = model(I_A, I_B, text_vi, text_ir, text_fuse)

        loss, loss_ssim, loss_int, loss_text = loss_function_prompt(I_A_gt, I_B_gt, I_fused, task)

        loss.backward()

        accu_total_loss += loss.detach()
        accu_ssim_loss += loss_ssim.detach()
        accu_int_loss += loss_int.detach()
        accu_text_loss += loss_text.detach()

        lr = optimizer.param_groups[0]["lr"]

        data_loader.desc = "[train epoch {}] loss: {:.3f}  ssim loss: {:.3f}  int loss: {:.3f}   text loss: {:.3f}  lr: {:.6f}".format(
            epoch, accu_total_loss.item() / (step + 1),
            accu_ssim_loss.item() / (step + 1), accu_int_loss.item() / (step + 1), accu_text_loss.item() / (step + 1),
            lr)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    return accu_total_loss.item() / (step + 1), accu_ssim_loss.item() / (step + 1), accu_int_loss.item() / (
                step + 1), accu_text_loss.item() / (step + 1), lr


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, lr, filefold_path):
    loss_function_prompt = fusion_prompt_loss()

    model.eval()
    accu_total_loss = torch.zeros(1).to(device)
    accu_ssim_loss = torch.zeros(1).to(device)
    accu_int_loss = torch.zeros(1).to(device)
    accu_text_loss = torch.zeros(1).to(device)
    save_epoch = 1
    save_length = 60
    cnt = 0
    save_RGB_fuse = True

    if torch.cuda.is_available():
        loss_function_prompt = loss_function_prompt.to(device)

    if epoch % save_epoch == 0:
        evalfold_path = os.path.join(filefold_path, str(epoch))
        if os.path.exists(evalfold_path) is False:
            os.makedirs(evalfold_path)

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        I_A, I_B, I_A_gt, I_B_gt, I_full, task, name = data
        text_vi_line = []
        text_ir_line = []
        text_fuse_line = []
        for index in range(len(task)):
            if task[index] == "vi_ir":
                text_vi_line.append(
                    "This is a visible image, and the focus is on texture information and color information.")
                text_ir_line.append("This is an infrared image, and the focus is on thermal radiation information.")
                text_fuse_line.append(
                    "A fused image combining visible light texture and color with infrared heat information.")

        text_vi = clip.tokenize(text_vi_line).to(device)
        text_ir = clip.tokenize(text_ir_line).to(device)
        text_fuse = clip.tokenize(text_fuse_line).to(device)

        if torch.cuda.is_available():
            I_A = I_A.to(device)
            I_B = I_B.to(device)
            I_A_gt = I_A_gt.to(device)
            I_B_gt = I_B_gt.to(device)
            I_full = I_full.to(device)

        # I_fused = model(I_A, I_B, text_vi, text_ir)

        I_fused = model(I_A, I_B, text_vi, text_ir, text_fuse)

        if epoch % save_epoch == 0:
            if cnt <= save_length:
                fused_img_Y = tensor2numpy(I_fused)
                img_full = tensor2numpy(I_full)
                img_ir = tensor2numpy(I_B_gt)
                save_pic(fused_img_Y, evalfold_path, str(name[0]))
                if save_RGB_fuse == True:
                    save_pic(img_full, evalfold_path, str(name[0]) + "vis")
                    save_pic(img_ir, evalfold_path, str(name[0]) + "ir")
                cnt += 1

        loss, loss_ssim, loss_int, loss_text = loss_function_prompt(I_A_gt, I_B_gt, I_fused, task)

        accu_total_loss += loss
        accu_ssim_loss += loss_ssim.detach()
        accu_int_loss += loss_int.detach()
        accu_text_loss += loss_text

        data_loader.desc = "[val epoch {}] loss: {:.3f}  ssim loss: {:.3f}  int loss: {:.3f} text loss: {:.3f}  lr: {:.6f}".format(
            epoch, accu_total_loss.item() / (step + 1),
            accu_ssim_loss.item() / (step + 1), accu_int_loss.item() / (step + 1), accu_text_loss.item() / (step + 1),
            lr)

    return accu_total_loss.item() / (step + 1), accu_ssim_loss.item() / (step + 1), accu_int_loss.item() / (
                step + 1), accu_text_loss.item() / (step + 1)


def mergy_Y_RGB_to_YCbCr(img1, img2):
    Y_channel = img1.squeeze(0).cpu().numpy()
    Y_channel = np.transpose(Y_channel, [1, 2, 0])

    img2 = img2.squeeze(0).cpu().numpy()
    img2 = np.transpose(img2, [1, 2, 0])

    img2_YCbCr = cv2.cvtColor(img2, cv2.COLOR_RGB2YCrCb)
    CbCr_channels = img2_YCbCr[:, :, 1:]
    merged_img_YCbCr = np.concatenate((Y_channel, CbCr_channels), axis=2)
    merged_img = cv2.cvtColor(merged_img_YCbCr, cv2.COLOR_YCrCb2RGB)
    return merged_img


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


# def save_pic(outputpic, path, index : str):
#     outputpic[outputpic > 1.] = 1
#     outputpic[outputpic < 0.] = 0
#     outputpic = cv2.UMat(outputpic).get()
#     outputpic = cv2.normalize(outputpic, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
#     outputpic=outputpic[:, :, ::-1]
#     save_path = os.path.join(path, index + ".png")
#     cv2.imwrite(save_path, outputpic)
def save_pic(outputpic, path, index):
    # 确保是 numpy
    if isinstance(outputpic, torch.Tensor):
        outputpic = outputpic.detach().cpu().numpy()

    # squeeze 到 [H,W]
    outputpic = outputpic.squeeze()

    # clip 到 [0,1]
    outputpic = np.clip(outputpic, 0.0, 1.0)

    # 映射到 [0,255]
    outputpic = (outputpic * 255.0).astype(np.uint8)

    save_path = os.path.join(path, index + ".png")
    cv2.imwrite(save_path, outputpic)


def show_img(images, imagesl, B):
    for index in range(B):
        img = images[index, :]
        img_np = np.array(img.permute(1, 2, 0).detach().cpu())
        plt.figure(1)
        plt.imshow(img_np)
        img = imagesl[index, :]

        img_np = np.array(img.permute(1, 2, 0).detach().cpu())
        plt.figure(2)
        plt.imshow(img_np)
        plt.show(block=True)


def tensor2numpy(x):
    x = x.detach().cpu()

    if x.dim() == 4:  # [1,C,H,W]
        x = x.squeeze(0)

    if x.dim() == 3:  # [C,H,W]
        x = x.permute(1, 2, 0)

    return x.numpy()


def tensor2numpy_single(L_tensor):
    L = L_tensor.squeeze(0)
    L_3 = torch.cat([L, L, L], dim=0)
    L_3 = L_3.cpu().detach().numpy()
    L_3 = np.transpose(L_3, [1, 2, 0])
    return L_3