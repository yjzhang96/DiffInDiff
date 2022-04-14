from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import torchvision.transforms.functional as TF
import torch
import numpy as np

class RestoreDataset(Dataset):
    def __init__(self, dataroot, datatype, degrade_type=None,  patch_size=16, split='train', data_len=-1, need_LQ=False):
        self.datatype = datatype
        self.patch_size = patch_size
        # self.r_res = r_resolution
        self.data_len = data_len
        self.need_LQ = need_LQ
        self.split = split
        if degrade_type:
            if degrade_type == 'blur':
                self.degrade_index = 0
            elif degrade_type == 'rain':
                self.degrade_index = 1
            elif degrade_type == 'noise':
                self.degrade_index = 2
            elif degrade_type == 'lowlight':
                self.degrade_index = 3
            else:
                raise TypeError('degrade type {:s} not found'.format(degrade_type))

        if datatype == 'lmdb':
            self.env = lmdb.open(dataroot, readonly=True, lock=False,
                                 readahead=False, meminit=False)
            # init the datalen
            with self.env.begin(write=False) as txn:
                self.dataset_len = int(txn.get("length".encode("utf-8")))
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        elif datatype == 'img':
            self.LQ_path = Util.get_paths_from_images(
                '{}/input'.format(dataroot))
            self.HQ_path = Util.get_paths_from_images(
                '{}/target'.format(dataroot))
            # if self.need_LQ:
            #     self.LQ_path = Util.get_paths_from_images(
            #         '{}/LQ_{}'.format(dataroot, l_resolution))
            self.dataset_len = len(self.HQ_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HQ = None
        img_LQ = None

        if self.datatype == 'lmdb':
            with self.env.begin(write=False) as txn:
                HQ_img_bytes = txn.get(
                    'HQ_{}_{}'.format(
                        self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                LQ_img_bytes = txn.get(
                    'LQ_{}_{}_{}'.format(
                        self.l_res, self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                if self.need_LQ:
                    LQ_img_bytes = txn.get(
                        'LQ_{}_{}'.format(
                            self.l_res, str(index).zfill(5)).encode('utf-8')
                    )
                # skip the invalid index
                while (HQ_img_bytes is None) or (LQ_img_bytes is None):
                    new_index = random.randint(0, self.data_len-1)
                    HQ_img_bytes = txn.get(
                        'HQ_{}_{}'.format(
                            self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    LQ_img_bytes = txn.get(
                        'LQ_{}_{}_{}'.format(
                            self.l_res, self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    if self.need_LQ:
                        LQ_img_bytes = txn.get(
                            'LQ_{}_{}'.format(
                                self.l_res, str(new_index).zfill(5)).encode('utf-8')
                        )
                img_HQ = Image.open(BytesIO(HQ_img_bytes)).convert("RGB")
                img_LQ = Image.open(BytesIO(LQ_img_bytes)).convert("RGB")
                if self.need_LQ:
                    img_LQ = Image.open(BytesIO(LQ_img_bytes)).convert("RGB")
        else:
            img_HQ = Image.open(self.HQ_path[index]).convert("RGB")
            img_LQ = Image.open(self.LQ_path[index]).convert("RGB")
            # if self.need_LQ:
            #     img_LQ = Image.open(self.LQ_path[index]).convert("RGB")
        # if self.need_LQ:
        #     [img_LQ, img_LQ, img_HQ] = Util.transform_augment(
        #         [img_LQ, img_LQ, img_HQ], split=self.split, min_max=(-1, 1))
        #     return {'LQ': img_LQ, 'HQ': img_HQ, 'LQ': img_LQ, 'Index': index}
        # else:
            if (self.split=='train'):
                ps = self.patch_size
                ### Data Augmentation ###
                
                w,h = img_HQ.size
                padw = ps-w if w<ps else 0
                padh = ps-h if h<ps else 0

                # Reflect Pad in case image is smaller than patch_size
                if padw!=0 or padh!=0:
                    img_LQ = TF.pad(img_LQ, (0,0,padw,padh), padding_mode='reflect')
                    img_HQ = TF.pad(img_HQ, (0,0,padw,padh), padding_mode='reflect')

                aug    = random.randint(0, 2)
                if aug == 1:
                    img_LQ = TF.adjust_gamma(img_LQ, 1)
                    img_HQ = TF.adjust_gamma(img_HQ, 1)

                aug    = random.randint(0, 2)
                if aug == 1:
                    sat_factor = 1 + (0.2 - 0.4*np.random.rand())
                    img_LQ = TF.adjust_saturation(img_LQ, sat_factor)
                    img_HQ = TF.adjust_saturation(img_HQ, sat_factor)

                img_LQ = TF.to_tensor(img_LQ)
                img_HQ = TF.to_tensor(img_HQ)

                hh, ww = img_HQ.shape[1], img_HQ.shape[2]

                rr     = random.randint(0, hh-ps)
                cc     = random.randint(0, ww-ps)
                aug    = random.randint(0, 8)

                # Crop patch
                img_LQ = img_LQ[:, rr:rr+ps, cc:cc+ps]
                img_HQ = img_HQ[:, rr:rr+ps, cc:cc+ps]

                # Data Augmentations
                if aug==1:
                    img_LQ = img_LQ.flip(1)
                    img_HQ = img_HQ.flip(1)
                elif aug==2:
                    img_LQ = img_LQ.flip(2)
                    img_HQ = img_HQ.flip(2)
                elif aug==3:
                    img_LQ = torch.rot90(img_LQ,dims=(1,2))
                    img_HQ = torch.rot90(img_HQ,dims=(1,2))
                elif aug==4:
                    img_LQ = torch.rot90(img_LQ,dims=(1,2), k=2)
                    img_HQ = torch.rot90(img_HQ,dims=(1,2), k=2)
                elif aug==5:
                    img_LQ = torch.rot90(img_LQ,dims=(1,2), k=3)
                    img_HQ = torch.rot90(img_HQ,dims=(1,2), k=3)
                elif aug==6:
                    img_LQ = torch.rot90(img_LQ.flip(1),dims=(1,2))
                    img_HQ = torch.rot90(img_HQ.flip(1),dims=(1,2))
                elif aug==7:
                    img_LQ = torch.rot90(img_LQ.flip(2),dims=(1,2))
                    img_HQ = torch.rot90(img_HQ.flip(2),dims=(1,2))
            elif self.split == 'val':
                # Validate on center crop
                if self.patch_size >0:
                    ps = self.patch_size
                    img_LQ = TF.center_crop(img_LQ, (ps,ps))
                    img_HQ = TF.center_crop(img_HQ, (ps,ps))

                img_LQ = TF.to_tensor(img_LQ)
                img_HQ = TF.to_tensor(img_HQ)

            elif self.split == 'test':
                # test
                img_LQ = TF.to_tensor(img_LQ)
                img_HQ = TF.to_tensor(img_HQ)
            # [img_LQ, img_HQ] = Util.transform_augment(
            #     [img_LQ, img_HQ], split=self.split, min_max=(-1, 1))
            return {'HQ': img_HQ, 'LQ': img_LQ, 'index': self.degrade_index}
