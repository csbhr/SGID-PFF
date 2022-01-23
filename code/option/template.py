def set_template(args):
    if args.template == 'Pre_Dehaze':
        args.task = "PreDehaze"
        args.model = "PRE_DEHAZE_T"
        args.save = "Pre_Dehaze"
        args.data_train = 'RESIDE'
        args.dir_data = '../dataset/RESIDE/ITS_train'
        args.data_test = 'RESIDE'
        args.dir_data_test = '../dataset/SOTS/indoor'
        args.t_channels = 1
        args.n_feat = 32
        args.n_resblock = 3
        args.size_must_mode = 4
        args.loss = '1*L1'
        args.other_loss = 'grad+others'
        args.lr = 1e-4
        args.lr_decay = 200
        args.epochs = 500
        args.batch_size = 8
        args.mid_loss_weight = 0.05
        args.save_middle_models = True
        args.save_images = False
        # args.resume = True
        # args.load = args.save
        # args.test_only = True
    elif args.template == 'ImageDehaze_SGID_PFF':
        args.task = "ImageDehaze"
        args.model = "DEHAZE_SGID_PFF"
        args.save = "ImageDehaze_SGID_PFF"
        args.data_train = 'RESIDE'
        args.dir_data = '../dataset/RESIDE/ITS_train'
        args.data_test = 'RESIDE'
        args.dir_data_test = '../dataset/SOTS/indoor'
        args.t_channels = 1
        args.n_feat = 32
        args.n_resblock = 3
        args.size_must_mode = 4
        args.loss = '1*L1'
        args.other_loss = 'grad+refer+others'
        args.lr = 1e-4
        args.lr_decay = 200
        args.epochs = 500
        args.batch_size = 8
        args.mid_loss_weight = 0.05
        args.save_middle_models = True
        args.save_images = False
        # args.resume = True
        # args.load = args.save
        # args.test_only = True
    else:
        raise NotImplementedError('Template [{:s}] is not found'.format(args.template))
