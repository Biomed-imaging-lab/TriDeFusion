import configparser
from .utils.logger import Logger
from .options.train_options import TrainOptions


SHOW_LOG = True

class Trainer:
    def __init__(self, config_path="config.ini"):
        self.logger = Logger(SHOW_LOG).get_logger(__name__)
        self.config_path = config_path
        self.config = configparser.ConfigParser()
        self.config.read("config.ini")
        self.opt = TrainOptions().parse()






#         self.name = name
#         folder_name = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#         self.parser = argparse.ArgumentParser(description="Trainer")
#         self.parser.add_argument('--network', 
#                                  type=nn.Module, 
#                                  default='./dataset', 
#                                  help='Root directory of the dataset')
#         self.add_argument('--exp-dir', 
#                           type=str, 
#                           default='./experiments', 
#                           help='Directory to save experiments')
#         self.parser.add_argument('--debug', 
#                           action='store_true', 
#                           help='Enable debug mode')
#         self.parser.add_argument('--data-root', 
#                                  type=str, 
#                                  default='./dataset', 
#                                  help='Root directory of the dataset')
#         self.parser.add_argument('--imsize', 
#                                  type=int, 
#                                  default=256, 
#                                  help='Input image size')
#         self.parser.add_argument("--lr", 
#                                  type=float, 
#                                  default=1e-4, 
#                                  help='Learning rate')
#         self.parser.add_argument('--epochs', 
#                                  type=int, 
#                                  default=50, 
#                                  help='Number of epochs to train')
#         self.parser.add_argument('--batch-size', 
#                                  type=int, 
#                                  default=8, 
#                                  help='Batch size for training')
#         self.parser.add_argument('--cmap', 
#                                  type=str, 
#                                  default='inferno', 
#                                  help='Colormap for output images')
#         self.parser.add_argument('--training-type', 
#                                  type=str, 
#                                  default='standard', 
#                                  choices=['standard', 'distillation'],
#                                  help='Type of training: "standard" or "distillation"')
#         self.project_path = os.path.join(os.getcwd(), "experiments", folder_name)
#         self.model_path = os.path.join(os.getcwd(), "experiments", folder_name)
#         self.metrics_path = os.path.join(self.project_path, "metrics.yml")
#         self.config_path = os.path.join(self.project_path, "config.yml")
#         self.logs_path = os.path.join(self.project_path, "logs.txt")
#         self.log.info("Trainer is ready")


#     def _configure_optimizer_and_scheduler(model, lr, wd):
#         """Configure optimizer and scheduler"""
#         optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=[0.9, 0.99])
#         scheduler = OneCycleScheduler(lr_max=lr, div_factor=10, pct_start=0.3)
#         return optimizer, scheduler
    
#     def _calc_loss(args, task_loss_fn, noisy_target, student_output, kd_loss_fn = None, teacher_output = None):
#         reconstruction_loss = task_loss_fn(student_output, noisy_target)
#         if args.training_type =='standard':
#             return reconstruction_loss
#         else:
#             distillation_loss = kd_loss_fn(
#                         F.log_softmax(student_output, dim=1),
#                         F.softmax(teacher_output, dim=1)
#                     )
#             total_loss = reconstruction_loss + 0.5 * distillation_loss
#             return total_loss

#     def train(self):
#         print('Start training........................................................')
#         try:
#             pass
#         except KeyboardInterrupt:
#             self.log.info('Keyboard Interrupt captured...Saving models & training logs')
#         self.save_model()

#     def save_model(self, 
#                    denoiser_path: str, 
#                    params: dict, 
#                    metrics: dict):
#         os.makedirs(self.project_path, exist_ok=True)
#         self.config[self.name] = params
#         os.remove('config.ini')
#         with open('config.ini', 'w') as configfile:
#             self.config.write(configfile)

#         torch.save(model.state_dict(), args.ckpt_dir + "/model_epoch{}.pth".format(epoch))

#         with open(self.metrics_path, 'w') as metricsfile:
#             yaml.dump(metrics, metricsfile)
#         self.log.info(f"Metrics saved at {self.metrics_path}")


# def train(args):
#     device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
#     optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
#                                  weight_decay=args.wd, betas=[0.9, 0.99])
#     print('Start training........................................................')
#     # try:
#     #     for epoch in range(1, args.epochs + 1):
#     #         model.train()


if __name__ == "__main__":
    trainer = Trainer()