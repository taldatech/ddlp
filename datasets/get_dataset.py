# datasets
from datasets.traffic_ds import TrafficDataset, TrafficDatasetImage
from datasets.clevrer_ds import CLEVREREpDataset, CLEVREREpDatasetImage
from datasets.shapes_ds import generate_shape_dataset_torch
from datasets.balls_ds import Balls, BallsImage
from datasets.obj3d_ds import Obj3D, Obj3DImage
from datasets.phyre_ds import PhyreDataset, PhyreDatasetImage
from datasets.langtable_ds import LanguageTableDataset, LanguageTableDatasetImage


def get_video_dataset(ds, root, seq_len=1, mode='train', image_size=128):
    # load data
    if ds == "traffic":
        dataset = TrafficDataset(path_to_npy=root, image_size=image_size, mode=mode, sample_length=seq_len)
    elif ds == 'clevrer':
        dataset = CLEVREREpDataset(root=root, mode=mode, sample_length=seq_len)
    elif ds == 'balls':
        dataset = Balls(root=root, mode=mode, sample_length=seq_len)
    elif ds == 'obj3d':
        dataset = Obj3D(root=root, mode=mode, sample_length=seq_len)
    elif ds == 'obj3d128':
        image_size = 128
        dataset = Obj3D(root=root, mode=mode, sample_length=seq_len, res=image_size)
    elif ds == 'phyre':
        dataset = PhyreDataset(root=root, mode=mode, sample_length=seq_len, image_size=image_size)
    elif ds == 'langtable':
        dataset = LanguageTableDataset(root=root, mode=mode, sample_length=seq_len, image_size=image_size)
    else:
        raise NotImplementedError
    return dataset


def get_image_dataset(ds, root, mode='train', image_size=128, seq_len=1):
    # set seq_len > 1 when training with use_tracking
    # load data
    if ds == "traffic":
        dataset = TrafficDatasetImage(path_to_npy=root, image_size=image_size, mode=mode, sample_length=seq_len)
    elif ds == 'clevrer':
        dataset = CLEVREREpDatasetImage(root=root, mode=mode, sample_length=seq_len)
    elif ds == 'balls':
        dataset = BallsImage(root=root, mode=mode, sample_length=seq_len)
    elif ds == 'obj3d':
        dataset = Obj3DImage(root=root, mode=mode, sample_length=seq_len)
    elif ds == 'obj3d128':
        image_size = 128
        dataset = Obj3DImage(root=root, mode=mode, sample_length=seq_len, res=image_size)
    elif ds == 'phyre':
        dataset = PhyreDatasetImage(root=root, mode=mode, sample_length=seq_len, image_size=image_size)
    elif ds == 'shapes':
        if mode == 'train':
            dataset = generate_shape_dataset_torch(img_size=image_size, num_images=40_000)
        else:
            dataset = generate_shape_dataset_torch(img_size=image_size, num_images=2_000)
    elif ds == 'langtable':
        dataset = LanguageTableDatasetImage(root=root, mode=mode, sample_length=seq_len, image_size=image_size)
    else:
        raise NotImplementedError
    return dataset
