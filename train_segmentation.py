import argparse
import datetime
import os
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb

wandb.init(project="Hodgenet", entity="s183983")

from hodgenet import HodgeNetModel
from meshdata import HodgenetMeshDataset
root = 'C:/Users/lowes/OneDrive/Skrivebord/DTU/8_Semester/Advaced_Geometric_DL/BU_3DFE_3DHeatmaps_crop/'


def main(args):
    torch.set_default_dtype(torch.float64)  # needed for eigenvalue problems
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    mesh_files_train = glob.glob(os.path.join(root,"train","*.vtk"))
    mesh_files_val = glob.glob(os.path.join(root,"val","*.vtk"))
    seg_files_train = [os.path.join(root,"labels",'_'.join(os.path.basename(file).split('_')[0:2])+".npy")
                       for file in mesh_files_train]
    seg_files_val = [os.path.join(root,"labels",'_'.join(os.path.basename(file).split('_')[0:2])+".npy")
                     for file in mesh_files_val]
    mesh_files_test = glob.glob(os.path.join(root,"test","*.vtk"))
    seg_files_test = [os.path.join(root,"labels",'_'.join(os.path.basename(file).split('_')[0:2])+".npy")
                       for file in mesh_files_train]
    """
    t_files = os.listdir(root+'train')
    v_files = os.listdir(root+'val')
    
    for t_file in t_files:
        mesh_files_train.append(os.path.join(args.mesh_path, t_file))
        seg_files_train.append(os.path.join(args.seg_path, t_file))
    
    for v_file in v_files:
        mesh_files_train.append(os.path.join(args.mesh_path, v_file))
        seg_files_train.append(os.path.join(args.seg_path, v_file))
    """
    #files = sorted([f.split('.')[0] for f in os.listdir(args.mesh_path)])
    #cutoff = round(0.85 * len(files) + 0.49)

    #for i in files[:cutoff]:
        #mesh_files_train.append(os.path.join(args.mesh_path, f'{i}.off'))
        #seg_files_train.append(os.path.join(args.seg_path, f'{i}.seg'))
    #for i in files[cutoff:]:
        #mesh_files_val.append(os.path.join(args.mesh_path, f'{i}.off'))
        #seg_files_val.append(os.path.join(args.seg_path, f'{i}.seg'))

    features = ['vertices'] if args.no_normals else ['vertices', 'normals']

    dataset = HodgenetMeshDataset(
        mesh_files_train,
        decimate_range=None if args.fine_tune is not None else (1000, 99999),
        edge_features_from_vertex_features=features,
        triangle_features_from_vertex_features=features,
        max_stretch=0 if args.fine_tune is not None else 0.05,
        random_rotation=False, segmentation_files=seg_files_train,
        normalize_coords=True,
        lm_ids = args.lm_ids)

    validation = HodgenetMeshDataset(
        mesh_files_val, decimate_range=None,
        edge_features_from_vertex_features=features,
        triangle_features_from_vertex_features=features, max_stretch=0,
        random_rotation=False, segmentation_files=seg_files_val,
        normalize_coords=True,
        lm_ids = args.lm_ids)
    
    test_set = HodgenetMeshDataset(
        mesh_files_test, decimate_range=None,
        edge_features_from_vertex_features=features,
        triangle_features_from_vertex_features=features, max_stretch=0,
        random_rotation=False, segmentation_files=seg_files_test,
        normalize_coords=True,
        lm_ids = args.lm_ids)

    def mycollate(b): return b
    dataloader = DataLoader(dataset, batch_size=args.bs,
                            num_workers=args.num_workers, shuffle=True,
                            collate_fn=mycollate)
    validationloader = DataLoader(validation, batch_size=args.bs,
                                  num_workers=args.num_workers,
                                  collate_fn=mycollate)

    example = dataset[0]
    hodgenet = HodgeNetModel(
        example['int_edge_features'].shape[1],
        example['triangle_features'].shape[1],
        num_output_features=args.n_out_features, mesh_feature=False,
        num_eigenvectors=args.n_eig, num_extra_eigenvectors=args.n_extra_eig,
        resample_to_triangles=True,
        num_vector_dimensions=args.num_vector_dimensions)

    model = nn.Sequential(
        hodgenet,
        nn.Linear(args.n_out_features*args.num_vector_dimensions *
                  args.num_vector_dimensions, 32),
        nn.BatchNorm1d(32),
        nn.LeakyReLU(),
        nn.Linear(32, 32),
        nn.BatchNorm1d(32),
        nn.LeakyReLU(),
        nn.Linear(32, dataset.n_seg_categories))

    # categorical variables
    loss = nn.CrossEntropyLoss()

    # optimization routine
    print(sum(x.numel() for x in model.parameters()), 'parameters')
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    if args.fine_tune is not None:
        checkpoint = torch.load(os.path.join(args.fine_tune))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['opt_state_dict'])
        starting_epoch = checkpoint['epoch'] + 1
        print(f'Fine tuning! Starting at epoch {starting_epoch}')
    else:
        starting_epoch = 0

    train_writer = SummaryWriter(os.path.join(
        args.out, datetime.datetime.now().strftime('train-%m%d%y-%H%M%S')),
                                 flush_secs=1)
    val_writer = SummaryWriter(os.path.join(
        args.out, datetime.datetime.now().strftime('val-%m%d%y-%H%M%S')),
                               flush_secs=1)

    def epoch_loop(dataloader, epochname, epochnum, writer, optimize=True):
        epoch_loss, epoch_acc, epoch_acc_weighted, epoch_size = 0, 0, 0, 0
        pbar = tqdm(total=len(dataloader), desc=f'{epochname} {epochnum}')
        for batch in dataloader:
            if optimize:
                optimizer.zero_grad()

            batch_loss, batch_acc, batch_acc_weighted = 0, 0, 0

            seg_estimates = torch.split(model(batch), [m['triangles'].shape[0]
                                                       for m in batch], dim=0)
            for mesh, seg_estimate in zip(batch, seg_estimates):
                gt_segs = mesh['segmentation'].squeeze(-1)
                areas = mesh['areas']
                ll = loss(seg_estimate, gt_segs)
                acc = (seg_estimate.argmax(1) == gt_segs).float().mean()
                acc_w =  ((seg_estimate.argmax(1) == gt_segs) * areas).sum() / areas.sum()
                batch_loss +=ll
                batch_acc += acc
                batch_acc_weighted += acc_w
                
                wandb.log({"loss": ll,
                           "accuracy": acc,
                           "accuracy_w": acc_w})

                # Optional
                wandb.watch(model)

            epoch_loss += batch_loss.item()
            epoch_acc += batch_acc.item()
            epoch_acc_weighted += batch_acc_weighted.item()
            epoch_size += len(batch)

            batch_loss /= len(batch)
            batch_acc /= len(batch)
            batch_acc_weighted /= len(batch)

            pbar.set_postfix({
                'loss': batch_loss.item(),
                'accuracy': batch_acc.item(),
                'accuracy_weighted': batch_acc_weighted.item(),
            })
            pbar.update(1)

            if optimize:
                batch_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()

        writer.add_scalar('loss', epoch_loss / epoch_size, epochnum)
        writer.add_scalar('accuracy', epoch_acc / epoch_size, epochnum)
        writer.add_scalar('accuracy_weighted',
                          epoch_acc_weighted / epoch_size, epochnum)

        pbar.close()

    for epoch in range(starting_epoch, starting_epoch+args.n_epochs+1):
        model.train()
        epoch_loop(dataloader, 'Epoch', epoch, train_writer)

        # compute validation score
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                epoch_loop(validationloader, 'Validation',
                           epoch, val_writer, optimize=False)

            torch.save({
                'model_state_dict': model.state_dict(),
                'opt_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }, os.path.join(args.out,
                            f'{epoch}_finetune.pth'
                            if args.fine_tune is not None else f'{epoch}.pth'))
                            
    model.eval()
    with torch.no_grad():
        epoch_loop(validationloader, 'Test',
                   epoch, val_writer, optimize=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='out/vase')
    parser.add_argument('--mesh_path', type=str, default='BU_3DFE_3DHeatmaps_crop')
    parser.add_argument('--seg_path', type=str, default='data/BU_3DFE_3DHeatmaps_crop')
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--n_eig', type=int, default=32)
    parser.add_argument('--n_extra_eig', type=int, default=32)
    parser.add_argument('--n_out_features', type=int, default=32)
    parser.add_argument('--fine_tune', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_vector_dimensions', type=int, default=4)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--no_normals', action='store_true', default=False)
    parser.add_argument('--lm_ids', action='store_true', default=0)
    

    args = parser.parse_args()
    wandb.config = {
      "learning_rate": args.lr,
      "epochs": args.n_epochs,
      "batch_size": args.bs,
      "lm_ids": args.lm_ids
    }
    main(args)  
