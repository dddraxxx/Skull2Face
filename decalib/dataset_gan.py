from PIL import Image
import os
import torch
from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms as tf
import numpy as np
from pathlib import Path as pa

class DatasetGAN(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        if transform is not None:
            self.transform = transform
        else:
            self.transform = tf.Compose([
                tf.Resize((224, 224)),
                tf.ToTensor(),
            ])
        # glob images with png in root
        # if root is iterable
        if isinstance(root, (list, tuple)):
            self.imgs = []
            for r in root:
                self.imgs += list(sorted(pa(r).glob('*.png')))
        else:
            self.imgs = list(sorted(pa(root).glob('*.png')))

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {
            'images': img,
            'id': img_path.stem[-5:],
        }
    def __len__(self):
        return len(self.imgs)
    
def build_dataset(root, transform=None):
    return DatasetGAN(root, transform)

def build_train(cfg):
    root = cfg.gan_tr
    return build_dataset(root)
def build_val(cfg):
    root = cfg.gan_val
    return build_dataset(root)


# crop face
# save
if __name__=='__main__':
    import datasets.detectors as detectors
    face_detector = detectors.FAN()
    from pathlib import Path as pa
    from tqdm import tqdm
    dir = pa('/mnt/sdh/sgraph/data/00001')
    # set n process to do below
    def crop_face(img_path):
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        bbox, _ = face_detector.run(img)
        # bbox to int
        bbox = [int(x) for x in bbox]
        # print(bbox)
        img_crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        img_crop = Image.fromarray(img_crop)
        return img_crop
    # for ddir in dir.parent.glob('0000[0-9]'):atase
    for ddir in sorted(dir.parent.glob('000[1-9][0-9]')):
        tar_dir = dir.parent / '{}_crop'.format(ddir.name)
        tar_dir.mkdir(exist_ok=True)

        # for training
        tq = tqdm(list(ddir.glob('*[13579].png')))
        for img_path in tq:
            # tqdm set description
            tq.set_description(desc='Processing {}'.format(img_path))
            if (tar_dir / img_path.name).exists():
                # the following is use to correct afore error; you should use tar_dir/img_path.name if just want to resize
                # img = Image.open(tar_dir/img_path).convert("RGB")
                try:
                    img = crop_face(img_path)
                except:
                    # remove tar_dir/img_path.name
                    (tar_dir / img_path.name).unlink()
                    tq.write('Error for this img: {}'.format(img_path))
                    continue
                # resize it to 224x224
                img_resize = img.resize((224, 224))
                img_resize.save(tar_dir / img_path.name)
                continue
            # because I already executed it and I just try to resize the process images, no need to crop new faces
            continue
            try:
                crop_face(img_path).save(tar_dir / img_path.name)
            except:
                tq.write('Error for this img: {}'.format(img_path))
                continue
        # for validation
        # for img_path in tqdm(list(ddir.glob('*[02468].png'))):
        #     crop_face(img_path)
            # img = Image.open(img_path).convert("RGB")
            # arr = np.array(img)
            # ts = tf.ToTensor()(img)
            # print(arr.max(), ts.max())
            # break