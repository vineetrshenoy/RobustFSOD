import os
from detectron2.utils import comm
from detectron2.engine import launch
from detectron2.data import MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer
from defrcn.config import get_cfg, set_global_cfg
from defrcn.evaluation import DatasetEvaluators, verify_results
from defrcn.engine import Trainer, TwoSteamTrainer, default_argument_parser, default_setup
import wandb

def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_global_cfg(cfg)
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    if comm.is_main_process():
        pathsplit = cfg.OUTPUT_DIR.split('/')
        group = pathsplit[-1]
        dataset = pathsplit[1]
        split = pathsplit[-3][-1]
        group = '{}-{}-{}'.format(dataset,split,group)
        wandb.tensorboard.patch(root_logdir=cfg.OUTPUT_DIR)
        config_dictionary = dict(yaml=cfg)
        run = wandb.init(project="MFDC",
                   entity="vshenoy",
                   sync_tensorboard=True,
                   group=group,
                   config=config_dictionary,
                   tags=[args.tags[0]],
                   settings=wandb.Settings(start_method="fork"))

    if cfg.DATASETS.TWO_STREAM:
        trainer = TwoSteamTrainer(cfg)
    else:
        trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    #main(args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
