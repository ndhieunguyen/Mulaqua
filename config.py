class BaseConfig:
    def __init__(self):
        # Paths
        self.cache_dir = "cache"
        self.data_folder = "data/augmented"
        self.output_dir = "output"

        # Data
        self.prepare_dataset = False
        self.k_folds = 5
        self.fold = 5

        # Model
        self.max_sequence_length = 1024
        self.pretrained_name = "QizhiPei/biot5-plus-base"
        self.accelerator = "cuda"
        self.trainable_layers = "none"
        self.cls_num_heads = 4

        # Hyperparameters
        self.seed = 0
        self.chosen_feature = "selfies"
        self.learning_rate = 1e-4
        self.per_device_train_batch_size = 128
        self.per_device_eval_batch_size = 128
        self.num_train_epochs = 200
        self.weight_decay = 0.01
        self.push_to_hub = False
        self.dropout = 0.2
        self.warmup_ratio = 0.05
        self.report_to = "none"
        self.save_total_limit = 1
        self.loss_function = "cross_entropy"  # "cross_entropy" or "focal_loss"
        self.focal_gamma = 2.0

        self.swin_used = "molnextr"  # "ocsr" or "molnextr" or "none"
        self.swin_ocsr_path = "checkpoints/ocsr.pth"
        self.swin_molnextr_path = "checkpoints/molnextr.pth"
        self.log_file = "log.out"


config = BaseConfig()
