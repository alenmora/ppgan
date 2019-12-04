import argparse
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image

TARGET_IMG_SIZE = 224
img_to_tensor = transforms.ToTensor()


def make_model():
    model = models.vgg16(pretrained=True)
    model = model.eval()
    return model

def extract_feature(model, imgpath):
    model.eval()

    img = Image.open(imgpath)
    img = img.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))
    tensor = img_to_tensor(img).unsqueeze(0)

    result = model(Variable(tensor))
    result_npy = result.data.cpu().numpy()
    return result_npy[0]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str)
    args = parser.parse_args()

    model = make_model()
    feature_vector = extract_feature(model, args.img_path)
    # print(feature_vector.shape)
    print(feature_vector)

