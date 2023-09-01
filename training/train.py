

# Train the model for one epoch
def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    for images, targets in train_loader:
        optimizer.zero_grad()
        # The model expects a list of images as input
        # Also, these images must be placed in the same device (gpu/cpu) as the model
        images = [image.to(device) for image in images]
        targets = [{k:v.to(device) for k,v in t.items()} for t in targets]
        
        # compute loss
        loss_dict = model(images, targets)
        loss_class = loss_dict["classification"]
        loss_boxes_regr = loss_dict["bbox_regression"]
        total_loss = loss_class + loss_boxes_regr
        total_loss.backward()
        optimizer.step()