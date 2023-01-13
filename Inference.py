import torch
from torchvision import transforms
import torchvision.models as models
from PIL import Image

# 전처리
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # ImageNet으로 학습된 weight
    weight = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = models.mobilenet_v3_small(weight).to(DEVICE)
    model.eval()
    
    for _ in range(10):
        input_image_name = input('이미지 파일 : ')
        input_image = Image.open(input_image_name)
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0).to(DEVICE) # 모델에서 요구하는 형태의 미니배치 생성
        
        with torch.no_grad():
            output = model(input_batch)
            
            # 정답 확인 : https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            print(torch.argmax(probabilities))
        
if __name__ == '__main__':
    main()