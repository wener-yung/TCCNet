from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataloader import *
from model.TCCNet import TCCNet
from all_config.config_main import config
"""
python MyTest.py --load your_model
"""

def safe_save(img, save_path):
    os.makedirs(save_path.replace(save_path.split('/')[-1], ""), exist_ok=True)
    img.save(save_path)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def __call__(self, img):
        for i in range(3):
            img[:, :, i] -= float(self.mean[i])
        for i in range(3):
            img[:, :, i] /= float(self.std[i])
        return img


class AutoTest:
    def __init__(self, test_dataset, data_root, model_path):
        assert isinstance(test_dataset, list), "error"
        self.data_root = data_root
        self.test_dataset = test_dataset
        self.dataloader = {}
        for dst in self.test_dataset:
            self.dataloader[dst] = DataLoader(Test_Dataset(data_root, [dst], config),
                                              batch_size=config.video_batchsize,
                                              shuffle=False, num_workers=8)
        self.model = TCCNet(config).cuda()
        self.model.load_checkpoint(config.load)
        self.tag_dir = config.result_path
        self.model.eval()

    def test(self):
        print('Saving to ', self.tag_dir)
        with torch.no_grad():
            for dst in self.test_dataset:
                for data in tqdm(self.dataloader[dst], desc="test:%s" % dst):
                    img = data['img']
                    path_li = data['path']
                    img = img.to(device=device, dtype=torch.float32)

                    result = self.model.my_eval(img)
                    result = torch.sigmoid(result)
                    b, t, c, h, w = result.shape

                    for i in range(t):
                        result_i = result[:, i, ::]
                        path_i = path_li[i]

                        for idx in range(b):
                            res_idx = result_i[idx]
                            path = path_i[idx]
                            npres = res_idx.squeeze().cpu().numpy()
                            if '612' in dst:
                                save_path = path.replace(self.data_root, self.tag_dir). \
                                    replace(".bmp", ".tif").replace('Frame', 'Pred')
                            elif '300' in dst:
                                save_path = path.replace(self.data_root, self.tag_dir). \
                                    replace('Frame', 'Pred')
                            elif 'E' in dst:
                                save_path = path.replace(self.data_root, self.tag_dir). \
                                    replace('Frame', 'Pred').replace('.bmp', 'png')
                            safe_save(Image.fromarray((npres * 255).astype(np.uint8)), save_path)


if __name__ == "__main__":
    test_list = ["CVC-ClinicDB-612-Test", "CVC-ClinicDB-612-Valid", "CVC-ColonDB-300", "ETIS"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if config.load:
        at = AutoTest(test_list,
                      config.video_testset_root,
                      config.load)
        at.test()
    else:
        print("no load!")
