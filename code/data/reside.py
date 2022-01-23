import os
from data import imagedata


class RESIDE(imagedata.IMAGEDATA):
    def __init__(self, args, name='RESIDE', train=True):
        super(RESIDE, self).__init__(args, name=name, train=train)

    def _set_filesystem(self, dir_data):
        print("Loading {} => {} DataSet".format("train" if self.train else "test", self.name))
        self.apath = dir_data
        self.dir_gt = os.path.join(self.apath, 'clear')
        self.dir_input = os.path.join(self.apath, 'hazy')
        print("DataSet clear path:", self.dir_gt)
        print("DataSet hazy path:", self.dir_input)
