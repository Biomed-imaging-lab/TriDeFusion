from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        self.add_argument('--ckpt-freq', type=int, default=50, help='how many epochs to wait before saving model')
        self.add_argument('--print-freq', type=int, default=100, help='how many minibatches to wait before printing training status')
        self.add_argument('--log-freq', type=int, default=1, help='how many epochs to wait before logging training status')
        self.add_argument('--plot-epochs', type=int, default=5, help='how many epochs to wait before plotting test output')
        self.parser.add_argument('--cmap', type=str, default='inferno', help='Colormap for output images')
        self.add_argument('--debug', action='store_true', help='Enable debug mode')
        
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=50, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        
        self.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
        parser.add_argument('--lr', type=float, 
                            default=1e-4, 
                            help='initial learning rate for Adam')
        self.add_argument('--wd', type=float, default=0., help='weight decay for AdamW')

        self.add_argument('--noise-levels-train', type=list, default=[1, 2, 4, 8, 16])
        self.add_argument('--noise-levels-test', type=list, default=[1])

        self.add_argument('--training-type', type=str, default='standard', choices=['standard', 'distillation'],
                          help='Type of training: "standard" or "distillation"')
        self.isTrain = True
        return parser