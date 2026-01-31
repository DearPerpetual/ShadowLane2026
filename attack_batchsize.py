import json
import os
import numpy as np
import cv2
import time
import copy
import torch
from torchvision.transforms import GaussianBlur
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import ast
from clrnet.models.registry import build_net


dataset_root = 'XXXX/data/tusimple'
def show_tensor_image(tensor):
    save_path = "XXXXX/temshow/out.png"
    x = tensor.detach().cpu().numpy()
    plt.imshow(x)
    plt.savefig(save_path)

def cosin_metric(x1, x2):
    dot_product = np.sum(x1 * x2, axis=1)
    return dot_product / (np.linalg.norm(x1, axis=1) * np.linalg.norm(x2, axis=1))

def lfw_test_attack(model, identity_list, batch_size=4):
    global_loss_log = []
    for i in range(0, len(identity_list), batch_size):
        batch = identity_list[i : i + batch_size]
        input_images, target_images, labels, masks = [], [], [], []

        for img_path in batch:
            input_image = load_image(img_path[0])
            target_image = load_image(img_path[1])
            label = img_path[2]

            if input_image is not None and target_image is not None:
                input_image = cv2.resize(input_image, (1280, 720), interpolation=cv2.INTER_LINEAR)
                target_image = cv2.resize(target_image, (1280, 720), interpolation=cv2.INTER_LINEAR)
                input_images.append(input_image)
                target_images.append(target_image)
                labels.append(label)


                random_prefix = f"{random.randint(1, 1)}-"
                parts = img_path[0].split('/')
                if len(parts) >= 3:
                    mask_name = f"{random_prefix}{parts[-3]}-{parts[-2]}-{parts[-1].replace('/', '-')}"
                mask_dir = './shape_silouette'
                all_masks = os.listdir(mask_dir)
                matching_mask = next((m for m in all_masks if mask_name in m), None)
                mask_path = os.path.join(mask_dir, matching_mask) if matching_mask else None

                if mask_path and os.path.exists(mask_path):
                    mask = Image.open(mask_path).resize((1280, 720))
                    mask = torch.tensor(np.array(mask)).float().cuda().unsqueeze(0).unsqueeze(0)
                    masks.append(mask)

        if len(input_images) == 0 or len(masks) != len(input_images):
            continue

        batch_size = len(input_images)
        if batch_size == 0:
            continue

        data0 = torch.tensor(np.array(input_images)).permute(0, 3, 1, 2).cuda()
        data1 = torch.tensor(np.array(target_images)).permute(0, 3, 1, 2).cuda()
        masks = torch.stack(masks).cuda().clamp(0, 1).requires_grad_(False)


        inten = torch.full((batch_size,), 0.5, dtype=torch.float, device="cuda", requires_grad=True)
        step_size = 0.05
        num_iter = 40
        print(f"Processing batch {i//batch_size + 1}/{len(identity_list)//batch_size}")
        prev_feature_target = None


        for itr in range(num_iter):
            inten.requires_grad_()
            masks.requires_grad_()
            occl = inten.view(-1, 1, 1, 1) * data0
            if masks.dim() == 5:
                masks = masks.squeeze(2)
            mask_stn = masks.expand(-1, 3, -1, -1)
            adv = occl * mask_stn + (1 - mask_stn) * data0
            adv = adv.clamp(-1, 1)
            input_gray = adv[:, 2, :, :] * 0.2989 + adv[:, 1, :, :] * 0.5870 + adv[:, 0, :, :] * 0.1140
            target_gray = data1[:, 2, :, :] * 0.2989 + data1[:, 1, :, :] * 0.5870 + data1[:, 0, :, :] * 0.1140
            input_gray = input_gray.unsqueeze(1).expand(-1, 3, -1, -1)
            target_gray = target_gray.unsqueeze(1).expand(-1, 3, -1, -1)
            _, feature_adv = model(input_gray)
            _, feature_target = model(target_gray)

            if prev_feature_target is not None:
                feature_target = 0.9 * prev_feature_target + 0.1 * feature_target
            prev_feature_target = feature_target.clone().detach()

            loss = LaneDetection_Loss,other_Loss
            loss.backward(retain_graph=True)
            global_loss_log.append(loss.item())
            if masks.grad is not None:
                #masks = (0.9 * masks + 0.1 * (masks + step_size * 2 * masks.grad.sign())).clamp(0, 1)
                #masks = (masks + step_size * 0.1 * masks.grad.sign()).clamp(0, 1.0)

            if inten.grad is not None:
                #inten = (0.9 * inten + 0.1 * (inten + step_size * 1 * inten.grad.sign())).clamp(0.1, 0.8)
                #inten = (inten + step_size * 1 * inten.grad.sign()).clamp(0.1, 0.8)
            inten = inten.detach_()
            masks = masks.clone().detach().requires_grad_(True)
            if itr % 10 == 0:
                step_size *= 1.1
                step_size = min(step_size, 0.1)

            sim = cosin_metric(
                feature_adv.detach().cpu().numpy(),
                feature_target.detach().cpu().numpy()
            )
            y_score = np.mean(sim)
            print(f"Iteration {itr}: y_score = {y_score:.4f}: Loss = {loss:.4f}")
            strict = labels[0] == "1"
            if (y_score >= 0.20) != strict:
                for idx in range(batch_size):
                    img_save_path = f"/XXXX/attack-img/{batch[idx][0]}"
                    os.makedirs(os.path.dirname(img_save_path), exist_ok=True)
                    img = (adv[idx].detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2.0 * 255
                    img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_LINEAR)
                    cv2.imwrite(img_save_path, img)
                break

        for idx in range(batch_size):
            img_save_path = f"/XXXXX/attack-img/{batch[idx][0]}"
            os.makedirs(os.path.dirname(img_save_path), exist_ok=True)
            img = (adv[idx].detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2.0 * 255
            img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(img_save_path, img)

def format_and_replace_txt(input_file):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()
    formatted_lines = []
    for line in lines:

        line = line.strip()

        try:
            data = ast.literal_eval(line)
            formatted_line = " ".join(data)
            formatted_lines.append(formatted_line)
        except Exception as e:
            print("!!!!")

    with open(input_file, 'w') as outfile:
        for formatted_line in formatted_lines:
            outfile.write(formatted_line + "\n")
    print("Finish")

def load_image(img_path):
    image = cv2.imread(os.path.join(dataset_root, img_path))
    if image is None:
        return None
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image
def get_lfw_list(pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    data_list = []
    for pair in pairs:
        splits = pair.split()
        data_list.append(np.array([splits[0], splits[1], splits[2]]))

    return data_list

def load_json(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def generate_identity_list(a_json_path, b_json_path, output_path):

    identity_list = []

    a_data = load_json(a_json_path)
    b_data = load_json(b_json_path)

    a_images = [item["raw_file"].replace("clips", "clips-clean") for item in a_data]
    b_images = [item["raw_file"].replace("clips", "clips-clean") for item in b_data]

    for a_path in a_images:
        a_last_two = "/".join(a_path.split("/")[-3:])
        for b_path in b_images:
            b_last_two = "/".join(b_path.split("/")[-3:])
            if a_last_two == b_last_two:
                identity_list.append([a_path, b_path, "1"])

    with open(output_path, 'w') as f:
        for entry in identity_list:
            f.write(json.dumps(entry) + '\n')

    print(f"Identity list saved to {output_path}")
