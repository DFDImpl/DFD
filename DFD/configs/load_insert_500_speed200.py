from easydict import EasyDict
from time import localtime, strftime
# set experiment configs
opt = EasyDict()

opt.task_name = "load_insert_500_speed200"
opt.files = [4,5,6,7]
opt.src_domain_idx = [0,2,3]
opt.tgt_domain_idx = [1]

opt.num_source = len(opt.src_domain_idx)
opt.num_target = len(opt.tgt_domain_idx)
opt.num_domain = opt.num_source+opt.num_target

opt.all_domain_idx = opt.src_domain_idx+opt.tgt_domain_idx


opt.shuffle = True

opt.use_pretrain_R = False
opt.fix_u_r = False

opt.d_loss_type = "ADDA_loss"#"ADDA_loss"  # "DANN_loss_mean" # "CIDA_loss" # "GRDA_loss" # "DANN_loss"

opt.lambda_gan = 0.4
opt.lambda_reconstruct = 50
opt.lambda_u_concentrate = 0.8
opt.lambda_beta = 0.8
opt.lambda_beta_alpha = 0.8

# for warm up
opt.init_lr = 1e-4
opt.peak_lr_e = 1e-4
opt.peak_lr_d = 1e-4
opt.final_lr = 1e-8
opt.warmup_steps = 20

opt.seed = 2333
opt.num_epoch = 120
opt.batch_size = 16

opt.use_visdom = False  # True
opt.visdom_port = 2000
tmp_time = localtime()
opt.outf = f'result_save/{opt.task_name}/{strftime("%Y-%m-%d-%H-%M-%S", tmp_time)}'
opt.save_interval = 5
opt.test_interval = 1

opt.device = "cuda"
opt.gpu_device = "0"
opt.gamma = 100
opt.beta1 = 0.9
opt.weight_decay = 5e-4
opt.normalize_domain = False
opt.no_bn = True  # do not use batch normalization

# network parameter
opt.num_hidden = 512
opt.num_class = 3
opt.input_dim = 128  # the dimension of input data x

opt.u_dim = 32  # the dimension of local domain index u
opt.beta_dim = 8

# for grda discriminator
opt.sample_v = 20

# how many nodes to save
opt.save_sample = 100
