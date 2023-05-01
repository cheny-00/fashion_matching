import torch
from torchvision import transforms

from transforms.build import build_transforms

def save_checkpoints(i, model, loss, save_path):
    model.eval()
    torch.save({
        'epochs': i,
        'model_state_dict': model.state_dict(),
        'loss': loss
    }, save_path)

# refer to https://github.com/adambielski/siamese-triplet
# def get_transform():
#     transform = transforms.Compose([
#         # transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         # transforms.Normalize((mean,), (std,))
#     ])
#     return transform

def get_transform(is_train=True):
    return build_transforms(is_train)
    

def load_model(MODEL_CLASS, checkpoint_path, params, emb_size):
    model = MODEL_CLASS(params, emb_size)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    epochs = checkpoint['epochs']
    loss = checkpoint['loss']
    return model, epochs, loss


def build_optim_params(model, params):
    regular_parameters = []
    regular_parameter_names = []

    center_parameters = []
    center_parameters_names = []

    for name, parameter in model.named_parameters():
        if parameter.requires_grad is False:
            print(f'Parameter {name} does not need a Grad. Excluding from the optimizer...')
            continue
        elif 'center' in name:
            center_parameters.append(parameter)
            center_parameters_names.append(name)
        else:
            regular_parameters.append(parameter)
            regular_parameter_names.append(name)

    param_groups = [
        {"params": regular_parameters, "names": regular_parameter_names},
    ]

    center_params_group = [
        {'params': center_parameters, "names": center_parameters_names}
    ]
    return param_groups, center_params_group


