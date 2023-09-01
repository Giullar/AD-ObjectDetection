

def collate_fn(batch):
    #We want to obtain the following format for each batch:
    # ((tens_img_1,...,tens_img_n), ({boxes_1:tens_boxes_1, labels_1:tens_labels1},..., {boxes_n:tens_boxes_n, labels_n:tens_labelsn}))
    
    #return tuple(zip(*[(x[0], {"boxes": x[1]["boxes"],"labels": x[1]["labels"]}) for x in batch]))
    return tuple(zip(*batch))