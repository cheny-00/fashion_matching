import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train import TripletEmbedTrainer
from model import SiameseNetwork, EmbedModel
from triplet_loss import euclidean_dist, cosine_similarity
from utils import save_checkpoints, get_transform, load_model, build_optim_params
from data import get_filenames, get_label, read_single_pic, DFSiameseDataset, TripletEmbedDataset, batch_convert_fn

def pickup_module(model_name): # => move to init
    modules = {
        "embed_model":{
            "trainer": TripletEmbedTrainer,
            "model": EmbedModel,
            "dataset": TripletEmbedDataset
        }
    }
    m = modules[model_name]
    trainer_class, model_class, dataset_class = m['trainer'], m['model'], m['dataset']
    
    return trainer_class, model_class, dataset_class

def run_train(params, dataset_path):
    
    device = params['device']
    train_data_dir = dataset_path['train_data_dir']
    
    trainer_class, model_class, dataset_class = pickup_module(params['model_name'])
      
    # perpare data
    train_files_dict = get_filenames(train_data_dir, build_labels_fuc=get_label)
    # train_dataset = DFSiameseDataset(train_files_dict, transform=get_transform())
    train_dataset = dataset_class(train_files_dict, n_instances=4, do_resample=params['do_resample'], transform=get_transform())
    assert params['train_batch_size'] % params['n_instances'] == 0, "train_batch size % n_instances != 0 "
    params['train_batch_size'] = params['train_batch_size'] // params['n_instances']
    n_classes = train_dataset.n_classes
    params['num_classes'] = n_classes
    print(f'number of classes: {n_classes}')
    
    
    # perpare dataloader
    train_iter = DataLoader(train_dataset,
                            batch_size=params['train_batch_size'],
                            shuffle=True,
                            drop_last=True,
                            collate_fn=batch_convert_fn)

    # model = SiameseNetwork()
    model = model_class(params, n_classes)
    if params['reuse_checkpoint']:
        model, last_epoch, last_loss = load_model(model_class ,params['load_ckpt_path'], params, params['num_classes'])
        print(last_epoch, last_loss)
    
    optim_list = list()
    params_groups, center_params_group =  build_optim_params(model, params)
    optimizer = torch.optim.Adam(params_groups, lr=params['lr'])
    center_optimizer = torch.optim.SGD(center_params_group, lr=params['center_lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 70], gamma=0.1)
    optim_list = [optimizer, center_optimizer]
    
    trainer = trainer_class(model, device, optim_list, params)
    trainer.scheduler = scheduler
    if params['use_tensorboard']:
        trainer.init_tensorboard('./.tb_logs')
    
    
    for i in range(params['epochs']):
        trainer.n_epoch += 1
        trainer.train(train_iter)
        if i > 15 and (i + 1) % 10 == 0:
            save_checkpoints(i, model, trainer.losses, params['ckpt_prefix'] + f"{i + 1}.pt")
        train_dataset.data = train_files_dict
        train_dataset._form_dataset()
        
    save_checkpoints(i, model, trainer.losses, params['ckpt_save_path'])
    trainer.tb_writer.flush()
    
    
    
def compare_images(model, x, y, dist_fn, threshold=0.5):
    model.eval()
    x, y = x.unsqueeze(0), y.unsqueeze(0)
    with torch.no_grad():
        emb_x, feat_x = model(x)
        emb_y, feat_y = model(y)
        dist_xy = dist_fn(emb_x, emb_y)
        print(dist_xy, torch.argmax(feat_x), torch.argmax(feat_y))
        pred = (dist_xy.item() > threshold)
        print(pred, dist_xy.item())

    return pred

def inference_single(model, x, y, dist_fn, threshold):
    print(compare_images(model, x, y, dist_fn, threshold))
    

def run_test(params, dataset_path, dist_fn, eval_params):
    
    test_date_dir = dataset_path['test_data_dir']
    test_files_dict = get_filenames(test_date_dir, build_labels_fuc=get_label)
    
    
    test_dataset = DFSiameseDataset(test_files_dict,
                                    use_record=eval_params['use_record'],
                                    load_record_path=eval_params['load_record_path'],
                                    transform=get_transform(False))
    
    if eval_params['save_dataset_record']:
        test_dataset.save_dataset_record(test_dataset.record_pairs_id, eval_params['save_record_path'])
    
    test_iter = DataLoader(test_dataset,
                           batch_size=params['test_batch_size'],
                           shuffle=False,
                           drop_last=False)

    ckpt_path = eval_params['load_ckpt_path']
    model, _, _ = load_model(EmbedModel, ckpt_path, eval_params, eval_params['num_classes'])
    device = params['device']
    threshold = eval_params['threshold']
    model.to(device)
    n_correct = 0
    n_total = 0
    model.eval()

    tqdm_test_iter = tqdm(test_iter)
    num_pos = num_neg = 0
    f1_score = 0
    with torch.no_grad():
        for data in tqdm_test_iter:
            x, y, target = data['x'].to(device), data['y'].to(device), data['target'].to(device)
            # out_x, out_y, out = model(x, y)
            emb_x, _ = model(x)
            emb_y, _ = model(y)
            # pred = (out > threshold).float()
            dist_xy = dist_fn(emb_x, emb_y)
            # print(dist_xy, target)
            pred = (dist_xy > threshold).float()
            n_correct += (pred == target.float()).sum().item()
            n_total += target.size(0)
            num_pos += torch.sum(target)
            # pp = num_
            num_neg += target.size(0) - torch.sum(target)
        acc = n_correct / n_total
        print(f"{ckpt_path} Accuracy: {acc:.5f}, number of positive: {num_pos}, number of negative: {num_neg}")
    return acc



if __name__ == "__main__":
    import sys
    from config import params, dataset_path, eval_params
    
    
    inp_args = sys.argv
    process = 'train'
    if len(inp_args) > 1:
        process = sys.argv[1]
        params['process'] = process
    if process == 'train':
        run_train(params, dataset_path)
    if process == 'test':
        run_test(params, dataset_path, F.cosine_similarity, eval_params)
    if process == 'inference_once':
        model, _, _ = load_model(EmbedModel, eval_params['load_ckpt_path'], params, params['num_classes'])
        model.to(params['device'])
        img1_path = "deepfashion_train_test_256/train_test_256/test/fashionMENTees_Tanksid0000481201_4full.jpg"
        # img1_path = "deepfashion_train_test_256/train_test_256/test/fashionMENShirts_Polosid0000113801_1front.jpg"
        img2_path = "deepfashion_train_test_256/train_test_256/test/fashionMENShirts_Polosid0000113801_3back.jpg"
        inference_single(model, read_single_pic(img1_path, get_transform(False)).to(params['device']), read_single_pic(img2_path, get_transform(False)).to(params['device']), cosine_similarity, threshold=eval_params['threshold'])
    