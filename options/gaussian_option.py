import argparse

class Gaussian_Options:
    def __init__(self, parser) -> None:
        self.parser = parser

    def get_parser(self):
        return self.parser
    
    def eval(self):
        self.parser.add_argument("--ckpt_iteration", default=30000, type=int)
        self.parser.add_argument("--render_train", action='store_true', default=False)
        self.parser.add_argument("--render_video", action='store_true', default=False)
        self.parser.add_argument("--train_view", nargs="+", type=int, default=[0])
        self.parser.add_argument("--interpolation", type=int, default=5)
    
    def dynamic_training(self):
        # Adaptive strategy
        pass
    
    def gcn_training(self):
        self.parser.add_argument("--epoch", default=101, type=int)
        self.parser.add_argument("--batch_size", default=32, type=int)
        self.parser.add_argument("--no_mapping", action='store_true', default=False)
        self.parser.add_argument("--num_stage", default=4, type=int)
        self.parser.add_argument("--linear_size", default=128, type=int)
        self.parser.add_argument("--dropout", default=0., type=float)
        self.parser.add_argument("--Rscale_ratio", default=0.5, type=float)
        self.parser.add_argument("--input_size", default=10, type=int)
        self.parser.add_argument("--output_size", default=1, type=int)
        self.parser.add_argument("--predict_more", action='store_true', default=False)
        self.parser.add_argument("--metrics", action='store_true', default=False)
        self.parser.add_argument("--cam_id", default=0, type=int)
        self.parser.add_argument("--frames", default=150, type=int)
        self.parser.add_argument("--noise_init", default=0.1, type=float)
        self.parser.add_argument("--noise_step", default=100, type=int)
        self.parser.add_argument("--exp_name", default="", type=str)
        
    
    def initial(self):
        self.parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
        self.parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
        self.parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
        self.parser.add_argument("--start_checkpoint", type=str, default = None)
        self.parser.add_argument("--batch", type=int, default=1)
        self.parser.add_argument("--max_gaussian_size", type=int, default=2e5)

        # Train strategy
        self.parser.add_argument("--jointly_iteration", type=int, default=1000, help="warm up iter")
        self.parser.add_argument("--max_points", type=int, default=100, help="The numbers of Initialize key points")
        self.parser.add_argument("--adaptive_points_num", type=int, default=0, help="Upperbound of the increasing key points")
        self.parser.add_argument("--set_iter", type=int, default=30) # need to change name
        self.parser.add_argument("--d", type=int, default=4)
        self.parser.add_argument("--w", type=int, default=256)
        self.parser.add_argument("--feature_dim", type=int, default=32, help="motion feature dimensions")

        
        self.parser.add_argument("--use_time_decay", action='store_true', default=False, help="Smooth the temporary input")
        self.parser.add_argument("--time_noise_ratio", type=float, default=0.5, help="decay time noise ratio")
        self.parser.add_argument("--time_noise_iteration", type=int, default=10000, help="After this iteration, no noise will add to the temporary input")

        
        self.parser.add_argument("--xyz_spatial_noise", action='store_true', default=False)
        self.parser.add_argument("--xyz_noise_iteration", type=int, default=10000, help="After this iteration, no noise will add to the xyz position")

        
        self.parser.add_argument("--nearest_num", type=int, default=6)
        self.parser.add_argument("--step_opacity", action='store_true', default=False)
        self.parser.add_argument("--step_opacity_iteration", type=int, default=5000)
        self.parser.add_argument("--opacity_type",  type=str, default="implicit")
        self.parser.add_argument("--beta", type=float, default=0.1)
        self.parser.add_argument("--feature_amplify", type=float, default=5)
        self.parser.add_argument("--resize", type=float, default=1)
        self.parser.add_argument("--knn_type", type=str, default="hybird")
        self.parser.add_argument("--norm_rotation", action='store_true', default=False)
        self.parser.add_argument("--ratio", type=float, default=0.5, help="HyperNeRF dataset resize")

        self.parser.add_argument("--densify_from_teaching", action='store_true', default=False)
        self.parser.add_argument('--densify_from_grad', type=str, choices=['True', 'False'], default="True")
        self.parser.add_argument("--teaching_threshold", type=float, default=0.2)
        

        self.parser.add_argument("--second_stage_iteration", type=int, default=30000)
        self.parser.add_argument("--third_stage_iteration", type=int, default=40000)

        self.parser.add_argument("--adaptive_from_iter", type=int, default=3000)
        self.parser.add_argument("--adaptive_end_iter", type=int, default=10000)
        self.parser.add_argument("--adaptive_interval", type=int, default=200)
        self.parser.add_argument("--seed", type=int, default=1)
