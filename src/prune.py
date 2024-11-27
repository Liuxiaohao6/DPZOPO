import torch
import numpy as np

def custom_mask(args, score):
    mask_dict = {}
    params_data = torch.cat([param.data.abs().view(-1) for param in score.values()])
    mask = params_data <= torch.kthvalue(params_data, int(args.pruning_ratio * params_data.numel()))[0]
    if args.pruning_mode == "rank":
        pruned_params = params_data[mask]
        pruned_params = torch.sort(pruned_params)[0]
    start = 0
    for name, param in score.items():
        end = start + param.data.numel()
        layer_mask = mask[start:end].view(param.data.size())

        if args.pruning_mode == "rank":
            rank_score = torch.zeros_like(layer_mask).to(dtype=torch.float)
            rank_pruned_param = param[layer_mask].data.abs()
            rank = torch.searchsorted(pruned_params, rank_pruned_param)
            rank_score[layer_mask] = args.upper_bound - (args.upper_bound - args.lower_bound) * rank / pruned_params.data.numel()
            mask_dict[name] = rank_score.to_sparse()
        else:
            mask_dict[name] = layer_mask.to_sparse()
        start = end
    return mask_dict

def zo_perturb_parameters(named_parameters_to_optim, random_seed, zo_eps, idx, scaling_factor=1):
    if torch.all(random_seed[idx] == 0).item():
        seed = np.random.randint(1000000000, size=len(named_parameters_to_optim))
        random_seed[idx] = torch.tensor(seed, dtype=torch.int)
    for layer, (name, param) in enumerate(named_parameters_to_optim):
        torch.manual_seed(random_seed[idx][layer])
        z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
        param.data.add_(z.mul_(scaling_factor * zo_eps))

def vector_wise_estimate_free(model, named_parameters_to_optim, zero_order_eps, q):
    free_seed = torch.zeros(q, len(named_parameters_to_optim)).to(dtype=torch.int)
    free_grad = torch.zeros(q)
    input = torch.ones((1, 1), dtype=torch.int).to(model.device)
    for idx in range(q):
        zo_perturb_parameters(named_parameters_to_optim, free_seed, zero_order_eps, idx)
        output1 = model(input)[0]
        zo_perturb_parameters(named_parameters_to_optim, free_seed, zero_order_eps, idx, -2)
        output2 = model(input)[0]
        projected_grad = torch.sum(output1 - output2) / 2
        free_grad[idx] = projected_grad
        zo_perturb_parameters(named_parameters_to_optim, free_seed, zero_order_eps, idx)
    return free_seed, free_grad

@torch.no_grad()
def linearize(model):
    signs = {}
    for name, param in model.named_parameters():
        signs[name] = torch.sign(param)
        param.abs_()
    return signs

@torch.no_grad()
def nonlinearize(model, signs):
    for name, param in model.named_parameters():
        param.mul_(signs[name])

def model_pruning_with_data_free(args, model):
    model.eval()

    named_parameters_to_optim = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            named_parameters_to_optim.append((name, param))
            
    signs = linearize(model)
    score_dict = {}
    
    #compute gradient
    seed1, grad1 = vector_wise_estimate_free(model, named_parameters_to_optim, args.zero_order_eps, args.query_nums)
    for idx in range(args.query_nums):
        for layer, (name, param) in enumerate(named_parameters_to_optim):
            torch.manual_seed(seed1[idx][layer])
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            # param.data = param.data + z * grad1[idx] * args.zo_eps / args.query_nums
            param.data.add_(z.mul_(grad1[idx]).mul_(args.zero_order_eps / args.query_nums))

    seed2, grad2 = vector_wise_estimate_free(model, named_parameters_to_optim, args.zero_order_eps, args.query_nums)
    for idx in range(args.query_nums):
        for layer, (name, param) in enumerate(named_parameters_to_optim):
            torch.manual_seed(seed1[idx][layer])
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data.sub_(z.mul_(grad1[idx]).mul_(args.zero_order_eps / args.query_nums))
            
    for idx in range(args.query_nums):    
        for layer, (name, param) in enumerate(named_parameters_to_optim):
            torch.manual_seed(seed1[idx][layer])
            z1 = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            torch.manual_seed(seed2[idx][layer])
            z2 = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            Hg = (z2.mul_(grad2[idx]).sub_(z1.mul_(grad1[idx]))).div_(args.zero_order_eps)
            if name in score_dict:
                score_dict[name].add_(Hg.mul_(param.data).div_(args.query_nums))
            else:
                score_dict[name] = Hg.mul_(param.data).div_(args.query_nums)
                
    nonlinearize(model, signs)
    del signs
    mask = custom_mask(args, score_dict)
    return mask      
    