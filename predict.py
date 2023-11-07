import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import json
import openpyxl
import pandas as pd
import sys
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import convnext_base as create_model

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    num_classes = 43
    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img_path = "data/img.png"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices1.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = create_model(num_classes=num_classes).to(device)
    # load model weights
    model_weight_path = "./weights/best_model0601.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        feature_scores = predict[:, :num_classes]
        print(f"feature_scores:{feature_scores}")
        predict_cla = torch.argmax(predict).numpy()



    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    claa_list=[]
    prob_list=[]
    data = {
        'class': claa_list,
        'prob': prob_list
    }
    claa_list.append(class_indict[str(predict_cla)])
    prob_list.append(predict[predict_cla].numpy())
    # data['class'] = class_indict[str(predict_cla)]
    # data['prob'] = predict[predict_cla].numpy()
    # 将字典转化为 pandas DataFrame
    df = pd.DataFrame(data)
    # 将 DataFrame 存入 Excel 文件中
    df.to_excel('output.xlsx', index=False, engine='openpyxl')
    print(print_res)
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()



if __name__ == '__main__':
    main()
