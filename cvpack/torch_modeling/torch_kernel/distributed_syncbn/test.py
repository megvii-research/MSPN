import torch
import apex
import os
from torch import nn
import argparse
import torch.distributed as dist
from syncbn import DistributedSyncBN
try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--sync_bn", action="store_true")
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )

    device = torch.device('cuda')
    model = DistributedSyncBN(3).to(device)
    nn.init.constant_(model.weight, 1)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    torch.manual_seed(args.local_rank)
    x = torch.rand(2, 3, 2, 2, device=device, requires_grad=True)
    # if dist.get_rank() == 1:
    #     print(x)
    y = model(x)
    if dist.get_rank() == 0:
        print(y)

    torch.manual_seed(0)
    x1 = torch.rand(2, 3, 2, 2, device=device, requires_grad=True)
    torch.manual_seed(1)
    x2 = torch.rand(2, 3, 2, 2, device=device, requires_grad=True)
    x = torch.cat([x1, x2])
    # if dist.get_rank() == 0:
    #     print(x.pow(2).mean(3).mean(2).mean(0))

    model = nn.BatchNorm2d(3).to(device)
    nn.init.constant_(model.weight, 1)
    y = model(x)

    if dist.get_rank() == 0:
        print(y)



if __name__ == "__main__":
    main()