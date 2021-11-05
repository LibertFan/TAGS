import os
from tensorboard.backend.event_processing import event_accumulator


def main():
    data_dir = "/Users/fanzhihao/Documents/Research/NIPS2021"
    tensorboard_path = os.path.join(data_dir, "events.out.tfevents.uniter.base.pm.dm")
    ea = event_accumulator.EventAccumulator(tensorboard_path)
    ea.Reload()
    print(ea.scalars.Keys())

    # val_psnr = ea.scalars.Items('val_psnr')
    # print(len(val_psnr))
    # print([(i.step, i.value) for i in val_psnr])


if __name__ == "__main__":
    main()
