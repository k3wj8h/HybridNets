import torch
from torch import nn

class ModelWithLoss(nn.Module):
    def __init__(self, model, debug=False):
        super().__init__()
        self.criterion = FocalLoss()
        self.seg_criterion1 = TverskyLoss(alpha=0.7, beta=0.3, gamma=4.0 / 3)
        self.seg_criterion2 = FocalLossSeg(alpha=0.25)
        self.model = model
        self.debug = debug

    def forward(self, imgs, annotations, seg_annot, obj_list=None):
        _, regression, classification, anchors, segmentation = self.model(imgs)

        if self.debug:
          cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations, imgs=imgs, obj_list=obj_list)
        else:
          cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)
      
        tversky_loss = self.seg_criterion1(segmentation, seg_annot)
        focal_loss = self.seg_criterion2(segmentation, seg_annot)
        seg_loss = tversky_loss + focal_loss

        return cls_loss, reg_loss, seg_loss, regression, classification, anchors, segmentation
		
		
class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, classifications, regressions, anchors, annotations, **kwargs):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]  # assuming all image sizes are the same, which it is
        dtype = anchors.dtype

        anchor_widths = anchor[:, 3] - anchor[:, 1]
        anchor_heights = anchor[:, 2] - anchor[:, 0]
        anchor_ctr_x = anchor[:, 1] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 0] + 0.5 * anchor_heights

        for j in range(batch_size):

            classification = classifications[j, :, :] # [46035,4]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j] # true annotations: size:[a,5]  format:[x1, y1, x2, y2, class]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            #print(f'bbox_annotation.shape[0]: {bbox_annotation}')
            #print(f'classification: {classification}')
            #print(f'regression: {regression}')

            # if there is no annotation
            if bbox_annotation.shape[0] == 0:
                alpha_factor = torch.ones_like(classification) * alpha
                if torch.cuda.is_available():
                  alpha_factor = alpha_factor.cuda()
                alpha_factor = 1. - alpha_factor
                focal_weight = classification # p_t
                focal_weight = alpha_factor * torch.pow(focal_weight, gamma) # = (1-a)*p_t^(gamma)
                bce = -(torch.log(1.0 - classification)) # = -log(1-p_t)
                cls_loss = focal_weight * bce # = -(1-a) * p_t^(gamma) * log(1-p_t)
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).to(dtype).cuda())
                else:
                    regression_losses.append(torch.tensor(0).to(dtype))
                classification_losses.append(cls_loss.sum())

                continue

            # iou calculated for all anchor - bbox pair
            IoU = self.calc_iou(anchor[:, :], bbox_annotation[:, :4]) #shape: [num_of_anchors, num_of_bbox]
            
            # find which anchor has the best match with thee given bbox
            IoU_max, IoU_argmax = torch.max(IoU, dim=1) #shape: [46035]
           

            #######################
            # Classification loss #
            #######################
            #targets = torch.ones_like(classification) * -1
            targets = torch.zeros_like(classification) # shape: [num_anchors, num_categories]=[46035,4] we will sign with 1 if anchor belongs to a category
            
            if torch.cuda.is_available():
                targets = targets.cuda()
            
            # annotations assigned to the given anchor [num_annot, (x1, y1, x2, y2, class)]=[46035,5]
            assigned_annotations = bbox_annotation[IoU_argmax, :]
            
            positive_indices = torch.full_like(IoU_max,False,dtype=torch.bool) #torch.ge(IoU_max, 0.2) shape: [num_anchors]=[46035]
                        
            # filter boxes with area at least (10x10), shape: [num_anchor]=[46035]
            tensorA = (assigned_annotations[:, 2] - assigned_annotations[:, 0]) * (assigned_annotations[:, 3] - assigned_annotations[:, 1]) > 10 * 10
            
            # keep only annotations with IoU > 0.5 or small boxes with IoU > 0.15, shape: [num_anchor]=[46035]
            positive_indices[torch.logical_or(torch.logical_and(tensorA,IoU_max >= 0.5),torch.logical_and(~tensorA,IoU_max >= 0.15))] = True

            num_positive_anchors = positive_indices.sum()
            
            # sign 1 if the anchor assigned to the annotation, others remains 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1
            
            # apply focal loss
            alpha_factor = torch.ones_like(targets) * alpha
            if torch.cuda.is_available():
                alpha_factor = alpha_factor.cuda()

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            cls_loss = focal_weight * bce

            # avoid negative loss
            zeros = torch.zeros_like(cls_loss)
            if torch.cuda.is_available():
                zeros = zeros.cuda()
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, zeros)

            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.to(dtype), min=1.0))


            ###################
            # Regression loss #
            ###################
            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dy, targets_dx, targets_dh, targets_dw))
                targets = targets.t()

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(torch.le(regression_diff, 1.0 / 9.0),
                                              0.5 * 9.0 * torch.pow(regression_diff, 2),
                                              regression_diff - 0.5 / 9.0)
                regression_losses.append(regression_loss.mean())
            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).to(dtype).cuda())
                else:
                    regression_losses.append(torch.tensor(0).to(dtype))

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), \
               torch.stack(regression_losses).mean(dim=0, keepdim=True) * 50  # https://github.com/google/automl/blob/6fdd1de778408625c1faf368a327fe36ecd41bf7/efficientdet/hparams_config.py#L233


    def calc_iou(self, a, b):
        # a(anchor) [boxes, (y1, x1, y2, x2)]
        # b(gt, coco-style) [boxes, (x1, y1, x2, y2)]

        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        iw = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 0])
        ih = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 1])
        iw = torch.clamp(iw, min=0)
        ih = torch.clamp(ih, min=0)
        ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
        ua = torch.clamp(ua, min=1e-8)
        intersection = iw * ih
        IoU = intersection / ua
      
        return IoU
		
		
class TverskyLoss(nn.modules._Loss):

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, gamma: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert y_true.size(0) == y_pred.size(0)
        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        y_true = y_true.view(bs, num_classes, -1)
        y_pred = y_pred.view(bs, num_classes, -1)

        loss = 1.0 - self.compute_score(y_pred, y_true.type_as(y_pred), dims=dims)

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        return self.aggregate_loss(loss)

    def aggregate_loss(self, loss):
        return loss.mean() ** self.gamma

    def compute_score(self, output, target, smooth=0.0, eps=1e-7, dims=(0, 2)) -> torch.Tensor:
        assert output.size() == target.size()
        intersection = torch.sum(output * target, dim=dims)  # TP
        fp = torch.sum(output * (1.0 - target), dim=dims)
        fn = torch.sum((1 - output) * target, dim=dims)
        return intersection / (intersection + self.alpha * fp + self.beta * fn).clamp_min(eps)
		
		
class FocalLossSeg(nn.modules_Loss):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        return self.focal_loss_with_logits(y_pred, y_true, self.alpha, self.gamma)

    def focal_loss_with_logits(self, output: torch.Tensor, target: torch.Tensor, alpha, gamma):
        target = target.type(output.type())
        logpt = nn.functional.binary_cross_entropy_with_logits(output, target, reduction='none')
        pt = torch.exp(-logpt)
        loss = (1.0-pt).pow(gamma) * logpt
        loss *= alpha * target + (1-alpha) * (1-target)
        return loss.mean()