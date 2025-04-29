import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class TestOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self):
        parser = argparse.ArgumentParser()

        # Runtime settings
        parser.add_argument('--UseCUDA', help='Use CUDA?', type=str2bool, nargs='?', default=True)
        parser.add_argument('--NumWorker', help='num of worker for dataloader', type=int, default=1)
        parser.add_argument('--Mode', help='script mode', choices=['train', 'eval'], default='eval')
        parser.add_argument('--Seed', type=int, default=1)
        parser.add_argument('--IsDeter', type=str2bool, help='set False for efficiency', default=True)

        # Model
        parser.add_argument('--ModelSetting', help='Conv3D/Conv3DSpar', type=str, default='Conv3DSpar')
        parser.add_argument('--ModelFilePath', help='Path to pretrained model', type=str, default=None)
        parser.add_argument('--ModelRoot', help='Path to model folder', type=str, default='./models/')

        # Data
        parser.add_argument('--Dataset', help='Dataset name', type=str, default='UCSD_P2_256')
        parser.add_argument('--DataRoot', type=str, default='', help='Root path to dataset folder')
        parser.add_argument('--DatasetRoot', type=str, default='', help='Relative path to dataset subfolder (used in testing)')
        
        # Select Model
        parser.add_argument('--ModelName', help='Model name: AE (simple) or Gated_AE (with gates)', type=str, default='AE')

        # Input shape
        parser.add_argument('--ImgChnNum', help='image channel (1=gray, 3=RGB)', type=int, default=1)
        parser.add_argument('--FrameNum', help='frame num for video clip', type=int, default=16)
        parser.add_argument('--BatchSize', help='testing batchsize', type=int, default=1)

        # Output/logging
        parser.add_argument('--OutRoot', help='Path to save results', type=str, default='./results/')
        parser.add_argument('--Suffix', help='Suffix to identify experiment version', type=str, default='Non')
        parser.add_argument('--IsTbLog', type=str2bool, default=False, help='Log to TensorBoard')
        parser.add_argument('--PlotScores', type=str2bool, help='Plot anomaly scores', default=True)
        parser.add_argument('--PlotROC', type=str2bool, help='Plot ROC curve', default=True)
        parser.add_argument('--Round', type=int, help='RoundNumber')

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
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
        self.message = message

    def parse(self, is_print=True):
        parser = self.initialize()
        opt = parser.parse_args()
        if is_print:
            self.print_options(opt)
        self.opt = opt
        return self.opt
