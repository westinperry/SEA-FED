import argparse
from .testing_options import str2bool


class TrainOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self):
        parser = argparse.ArgumentParser(description="Training Options")

        # Basic config
        parser.add_argument('--UseCUDA', type=str2bool, nargs='?', const=True, default=True, help='Use CUDA if available')
        parser.add_argument('--NumWorker', type=int, default=1, help='Number of dataloader workers')
        parser.add_argument('--Seed', type=int, default=1, help='Random seed')
        parser.add_argument('--IsDeter', type=str2bool, default=False, help='Deterministic training (reproducibility)')
        
        # Dataset & model
        parser.add_argument('--Dataset', type=str, default='UCSD_P2_256', help='Dataset name')
        parser.add_argument('--ImgChnNum', type=int, default=1, help='Number of image channels')
        parser.add_argument('--FrameNum', type=int, default=16, help='Frames per video clip')
        parser.add_argument('--DataRoot', type=str, required=True, help='Path to dataset root')
        parser.add_argument('--ProximalMu', type=float, default=0.0, help='μ strength for FedProx proximal term during local training')

        # Training config
        parser.add_argument('--BatchSize', type=int, default=14, help='Batch size')
        parser.add_argument('--LR', type=float, default=1e-4, help='Learning rate')
        parser.add_argument('--EpochNum', type=int, default=100, help='Number of training epochs')
        parser.add_argument('--TextLogInterval', type=int, default=5, help='Console log interval (in batches)')
        parser.add_argument('--SaveCheckInterval', type=int, default=10, help='Epoch interval to save checkpoints')
        parser.add_argument('--PlotGraph', type=str2bool, default=False, help='Plot loss graph at end of training')
        
        # Select Model
        parser.add_argument('--ModelName', type=str, default='AE', help='Model name: AE (baseline) or Gated_AE (with gates)')

        # Output config
        parser.add_argument('--ModelRoot', type=str, required=True, help='Directory to save model and logs')
        parser.add_argument('--OutputFile', type=str, required=True, help='Filename to save the final model')

        # TensorBoard
        parser.add_argument('--IsTbLog', type=str2bool, default=False, help='Enable TensorBoard logging')

        # Resume training
        parser.add_argument('--IsResume', action='store_true', help='Resume training from checkpoint')
        parser.add_argument('--ResumePath', type=str, default='', help='Path to model checkpoint')
        parser.add_argument('--Round', type=int, help='RoundNumber')

        parser.add_argument('--IsSaveSEAdapter', type=str2bool, default=False, help='Save SEBlock and Adapter features during training')
        parser.add_argument('--ClientID', type=int, default=0, help='Client ID for saving personalized features')

        parser.add_argument('--Mode', type=str, choices=['train', 'eval'], required=True, help='train or eval')


        self.initialized = True
        self.parser = parser
        return parser

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = f'\t[default: {default}]'
            message += f'{k:>25}: {str(v):<30}{comment}\n'
        message += '----------------- End -------------------'
        print(message)
        self.message = message

    def parse(self, is_print=True):
        parser = self.initialize()
        opt = parser.parse_args()
        if is_print:
            self.print_options(opt)
        self.opt = opt
        return opt
