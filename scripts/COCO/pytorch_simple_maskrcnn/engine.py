import sys
import time

import torch

from scripts.coco_pretrained_weights.pytorch_simple_maskrcnn.utils import Meter, TextArea
try:
    from scripts.coco_pretrained_weights.pytorch_simple_maskrcnn.coco_eval import CocoEvaluator, prepare_for_coco
except:
    pass

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    # for p in optimizer.param_groups:
    #     p["lr"] = args.lr_epoch

    iters = len(data_loader)

    t_m = Meter("total")
    m_m = Meter("model")
    b_m = Meter("backward")
    model.train()
    A = time.time()
    for i, (images, targets) in enumerate(data_loader):
        T = time.time()
        num_iters = epoch * len(data_loader) + i

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        S = time.time()
        
        losses = model(images, targets)
        total_loss = sum(losses.values())
        m_m.update(time.time() - S)
            
        S = time.time()
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        b_m.update(time.time() - S)

        if num_iters % print_freq == 0:
            print("epoch:{}\t".format(epoch), "iter:{}/{}\t".format(i,iters), "\t".join("{}:{:.3f}".format(name, loss.item()) for name, loss in losses.items()))

        t_m.update(time.time() - T)
        if i >= iters - 1:
            break
           
    A = time.time() - A
    print("iter: {:.1f}, total: {:.1f} [s], model: {:.1f} [s], backward: {:.1f} [s]".format(1000*A/iters,1000*t_m.avg,1000*m_m.avg,1000*b_m.avg))
    return A / iters
            

def evaluate(model, data_loader, device, saved_model_path, generate=True):
    if generate:
        iter_eval = generate_results(model, data_loader, device, saved_model_path)

    dataset = data_loader.dataset.dataset
    iou_types = ["bbox", "segm"]
    coco_evaluator = CocoEvaluator(dataset.coco, iou_types)

    results = torch.load(saved_model_path, map_location="cpu")

    S = time.time()
    coco_evaluator.accumulate(results)
    print("accumulate: {:.1f}s".format(time.time() - S))

    # collect outputs of buildin function print
    temp = sys.stdout
    sys.stdout = TextArea()

    coco_evaluator.summarize()

    output = sys.stdout
    sys.stdout = temp
        
    return output, iter_eval
    
    
# generate results file   
@torch.no_grad()   
def generate_results(model, data_loader, device, saved_model_path):
    iters = len(data_loader)
    ann_labels = data_loader.dataset.dataset.ann_labels
        
    t_m = Meter("total")
    m_m = Meter("model")
    coco_results = []
    model.eval()
    A = time.time()
    for i, (images, targets) in enumerate(data_loader):
        T = time.time()
        
        # image = image.to(device)
        # target = {k: v.to(device) for k, v in target.items()}

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        S = time.time()
        torch.cuda.synchronize()
        outputs = model(images)
        outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
        output = outputs.pop()
        m_m.update(time.time() - S)

        targets = targets[0]
        prediction = {targets["image_id"].item(): {k: v.cpu() for k, v in output.items()}}
        coco_results.extend(prepare_for_coco(prediction, ann_labels))

        t_m.update(time.time() - T)
        if i >= iters - 1:
            break
     
    A = time.time() - A 
    print("iter: {:.1f}, total: {:.1f}, model: {:.1f}".format(1000*A/iters,1000*t_m.avg,1000*m_m.avg))
    
    S = time.time()
    print("all gather: {:.1f}s".format(time.time() - S))
    torch.save(coco_results, saved_model_path)
        
    return A / iters
    

