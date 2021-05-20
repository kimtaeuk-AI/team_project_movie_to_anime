import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from torchvision import transforms, models
import cv2
from torchvision.utils import save_image

model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

labels = ['background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

cmap = plt.cm.get_cmap('tab20c')
colors = (cmap(np.arange(cmap.N)) * 255).astype(np.int)[:, :3].tolist()
np.random.seed(2020)
np.random.shuffle(colors)
colors.insert(0, [0, 0, 0]) # background color must be black
colors = np.array(colors, dtype=np.uint8)

palette_map = np.empty((10, 0, 3), dtype=np.uint8)
legend = []

for i in range(21):
    legend.append(mpatches.Patch(color=np.array(colors[i]) / 255., label='%d: %s' % (i, labels[i])))
    c = np.full((10, 10, 3), colors[i], dtype=np.uint8)
    palette_map = np.concatenate([palette_map, c], axis=1)

plt.figure(figsize=(20, 2))
plt.legend(handles=legend)
plt.imshow(palette_map)

def segment(net, img):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    output = model(input_batch)['out'][0] # (21, height, width)
    #스칼라로 0 차원이다 

    output_predictions = output.argmax(0).byte().cpu().numpy() # (height, width) 
    #numpy 로 가져온다
    r = Image.fromarray(output_predictions).resize((img.shape[1], img.shape[0]))
    #numpy로 가져온걸 array로 pillow image로 가져온다 
    r.putpalette(colors)

    return r, output_predictions
#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#

img = cv2.imread('C:/Study/semantic-segmentation-pytorch-master/frame/frame0.jpg') # rgb 이미지 보기
img = cv2.resize(img, (1000, 500))
cv2.imwrite('C:/Study/semantic-segmentation-pytorch-master/frame2/frame0.jpg',img)
img = np.array(Image.open('C:/Study/semantic-segmentation-pytorch-master/frame2/frame0.jpg').convert('RGB'))
fg_h, fg_w, _ = img.shape
segment_map, pred = segment(model, img)
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
axes[0].imshow(img)
axes[1].imshow(segment_map)
mask = (pred == 15).astype(float) * 255 # 15: person
alpha = cv2.GaussianBlur(mask, (7, 7), 0).astype(float)
alpha = alpha / 255. # (height, width)
alpha = np.repeat(np.expand_dims(alpha, axis=2), 3, axis=2) # (height, width, 3)
foreground = cv2.multiply(alpha, img.astype(float))
#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#
#사람만
result = foreground.astype(np.uint8)
# result = cv2.resize(result,(500,400))
Image.fromarray(result).save('C:/Study/semantic-segmentation-pytorch-master/frame_human/frame0.jpg')
#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#
