import os
import glob
import imageio
import utils.data_utils as utils
import torch.utils.data as data


class IMAGEDATA(data.Dataset):
    def __init__(self, args, name='', train=True):
        self.args = args
        self.name = name
        self.train = train

        if train:
            self._set_filesystem(args.dir_data)
        else:
            self._set_filesystem(args.dir_data_test)

        self.images_gt, self.images_input = self._scan()

        self.num_image = len(self.images_gt)
        print("Number of images to load:", self.num_image)

        if train:
            self.repeat = max(args.test_every // max((self.num_image // args.batch_size), 1), 1)
            print("Dataset repeat:", self.repeat)

        if args.process:
            self.data_gt, self.data_input = self._load(self.images_gt, self.images_input)

    def _set_filesystem(self, dir_data):
        print("Loading {} => {} DataSet".format("train" if self.train else "test", self.name))
        self.apath = dir_data
        self.dir_gt = os.path.join(self.apath, 'GT')
        self.dir_input = os.path.join(self.apath, 'INPUT')
        print("DataSet GT path:", self.dir_gt)
        print("DataSet INPUT path:", self.dir_input)

    def _scan(self):
        names_gt = sorted(glob.glob(os.path.join(self.dir_gt, '*')))
        names_input = sorted(glob.glob(os.path.join(self.dir_input, '*')))
        assert len(names_gt) == len(names_input), "len(names_gt) must equal len(names_input)"
        return names_gt, names_input

    def _load(self, names_gt, names_input):
        print('Loading image dataset...')
        data_input = [imageio.imread(filename)[:, :, :3] for filename in names_input]
        data_gt = [imageio.imread(filename)[:, :, :3] for filename in names_gt]
        return data_gt, data_input

    def __getitem__(self, idx):
        if self.args.process:
            input, gt, filename = self._load_file_from_loaded_data(idx)
        else:
            input, gt, filename = self._load_file(idx)

        input, gt = self.get_patch(input, gt, self.args.size_must_mode)
        input_tensor, gt_tensor = utils.np2Tensor(input, gt, rgb_range=self.args.rgb_range)

        return input_tensor, gt_tensor, filename

    def __len__(self):
        if self.train:
            return self.num_image * self.repeat
        else:
            return self.num_image

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_gt)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_gt = self.images_gt[idx]
        f_input = self.images_input[idx]
        gt = imageio.imread(f_gt)[:, :, :3]
        input = imageio.imread(f_input)[:, :, :3]
        filename, _ = os.path.splitext(os.path.basename(f_gt))
        return input, gt, filename

    def _load_file_from_loaded_data(self, idx):
        idx = self._get_index(idx)
        gt = self.data_gt[idx]
        input = self.data_input[idx]
        filename = os.path.splitext(os.path.split(self.images_gt[idx])[-1])[0]
        return input, gt, filename

    def get_patch(self, input, gt, size_must_mode=1):
        if self.train:
            input, gt = utils.get_patch(input, gt, patch_size=self.args.patch_size)
            h, w, c = input.shape
            if h != self.args.patch_size or w != self.args.patch_size:
                input = utils.bicubic_resize(input, size=(self.args.patch_size, self.args.patch_size))
                gt = utils.bicubic_resize(gt, size=(self.args.patch_size, self.args.patch_size))
                h, w, c = input.shape
            new_h, new_w = h - h % size_must_mode, w - w % size_must_mode
            input, gt = input[:new_h, :new_w, :], gt[:new_h, :new_w, :]
            if not self.args.no_augment:
                input, gt = utils.data_augment(input, gt)
        else:
            h, w, c = input.shape
            new_h, new_w = h - h % size_must_mode, w - w % size_must_mode
            input, gt = input[:new_h, :new_w, :], gt[:new_h, :new_w, :]
        return input, gt
