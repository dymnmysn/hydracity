import torch
import numpy as np

# mIoU calculation utility
def calculate_miou(preds, labels, num_classes=23, ignore_index=0):
    """Calculates mean IoU, ignoring the specified class index (like background)."""
    iou_per_class = []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        true_positive = ((preds == cls) & (labels == cls)).sum().item()
        false_positive = ((preds == cls) & (labels != cls)).sum().item()
        false_negative = ((preds != cls) & (labels == cls)).sum().item()
        
        denominator = true_positive + false_positive + false_negative
        if denominator == 0:
            iou = float('nan')  # Skip this class if there's no presence
        else:
            iou = true_positive / denominator
        
        iou_per_class.append(iou)
    
    # Filter out any nan values and compute mean IoU
    iou_per_class = [iou for iou in iou_per_class if not torch.isnan(torch.tensor(iou))]
    if len(iou_per_class) == 0:
        return 0.0  # No classes to calculate IoU
    else:
        return sum(iou_per_class) / len(iou_per_class)

        
def calculate_classwise_intersection_union(preds, labels, num_classes=23):
    classwise_intersection = torch.zeros(num_classes, dtype=torch.float32)
    classwise_union = torch.zeros(num_classes, dtype=torch.float32)
    
    for cls in range(0, num_classes):  # Ignore class 0 (usually background)
        intersection = torch.logical_and(preds == cls, labels == cls).sum().item()
        union = torch.logical_or(preds == cls, labels == cls).sum().item()
        
        classwise_intersection[cls] = intersection
        classwise_union[cls] = union

    return classwise_intersection, classwise_union

def calculate_final_miou_from_batches(batch_results, num_classes=23):

    # Initialize accumulators for the total intersection and union
    total_classwise_intersection = torch.zeros(num_classes, dtype=torch.float32)
    total_classwise_union = torch.zeros(num_classes, dtype=torch.float32)

    # Accumulate intersections and unions across all batches
    for classwise_intersection, classwise_union in batch_results:
        total_classwise_intersection += classwise_intersection
        total_classwise_union += classwise_union

    # Now compute the IoUs
    classwise_iou = []
    valid_classes = 0
    iou_sum = 0.0

    for cls in range(1, num_classes):  # Ignore class 0 (usually background)
        intersection = total_classwise_intersection[cls]
        union = total_classwise_union[cls]
        
        if union > 0:
            iou = intersection / union
            classwise_iou.append(iou.item())
            iou_sum += iou.item()
            valid_classes += 1
        else:
            classwise_iou.append(float('nan'))  # No instance of this class

    # Calculate mean IoU (ignoring NaN classes)
    mean_iou = iou_sum / valid_classes if valid_classes > 0 else float('nan')

    # Calculate total IoU (total intersection over total union)
    total_intersection = total_classwise_intersection.sum().item()
    total_union = total_classwise_union.sum().item()
    total_iou = total_intersection / total_union if total_union > 0 else float('nan')

    return classwise_iou, mean_iou, total_iou