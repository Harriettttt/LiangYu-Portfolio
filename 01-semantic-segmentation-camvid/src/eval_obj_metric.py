import numpy as np
from scipy.ndimage import label

class_labels_dict = {
    "Sky": 0,
    "Building": 1,
    "Pole": 2,
    "Road": 3,
    "LaneMarking": 4,
    "SideWalk": 5,
    "Pavement": 6,
    "Tree": 7,
    "SignSymbol": 8,
    "Fence": 9,
    "Car_Bus": 10,
    "Pedestrian": 11,
    "Bicyclist": 12,
    "Unlabelled": 13,
}

def compute_object_level_stats_percentile(
    gt_masks: np.ndarray, 
    pred_masks: np.ndarray,
    class_label: str,   # "Pedestrian"
    iou_threshold: float = 0.25,
    size_percentile: float = 50.0
):
    """
    Computes object-level metrics (precision, recall, F1) *separately* for 
    'small' vs. 'large' objects of a single class (class_label), where the 
    boundary between small & large is determined by the user-specified 
    percentile (size_percentile) of GT object areas.
    
    1) First pass: gather ALL ground-truth object areas across the dataset 
       for 'class_label' and find the area threshold = np.percentile(...).
    2) Second pass: do IoU-based matching (GT vs. predicted objects). 
       - If an object's area < threshold => "small"
       - If an object's area >= threshold => "large"

    Returns:
      p_small, r_small, f1_small, p_large, r_large, f1_large
    """
    class_idx = class_labels_dict[class_label]  # Mapping from class name to index

    # --- 1) First pass: gather GT objects for area distribution ---
    all_gt_areas = []   # Collect GT object areas for class_label

    frame_data = []

    N = len(gt_masks)
    for i in range(N):
        gt_frame = gt_masks[i]
        pred_frame = pred_masks[i]

        # Binary masks for the class of interest
        gt_binary = (gt_frame == class_idx).astype(np.uint8)
        pred_binary = (pred_frame == class_idx).astype(np.uint8)

        # Label connected components
        gt_labeled, gt_num_objs = label(gt_binary)
        pred_labeled, pred_num_objs = label(pred_binary)

        # Collect GT objects
        gt_objects = []
        for obj_id in range(1, gt_num_objs + 1):
            coords = np.where(gt_labeled == obj_id)
            area = len(coords[0])
            all_gt_areas.append(area)
            gt_objects.append(coords)

        # Collect predicted objects
        pred_objects = []
        for obj_id in range(1, pred_num_objs + 1):
            coords = np.where(pred_labeled == obj_id)
            pred_objects.append(coords)

        # Save for the second pass
        frame_data.append({
            "gt_objects": gt_objects,
            "pred_objects": pred_objects
        })

    # If no GT objects exist, return zeros
    if len(all_gt_areas) == 0:
        return 0, 0, 0, 0, 0, 0

    # --- Determine the dynamic size threshold ---
    dynamic_threshold = np.percentile(all_gt_areas, size_percentile)

    # --- 2) Second pass: IoU-based matching + "small"/"large" counting ---
    tp_small = fp_small = fn_small = 0
    tp_large = fp_large = fn_large = 0

    total_small_gt = total_large_gt = 0
    total_small_pred = total_large_pred = 0

    for i in range(N):
        gt_objects = frame_data[i]["gt_objects"]
        pred_objects = frame_data[i]["pred_objects"]

        labeled_gt_objs = []
        for coords in gt_objects:
            area = len(coords[0])
            size_cat = "small" if area < dynamic_threshold else "large"
            pixel_set = set(zip(coords[0], coords[1]))
            labeled_gt_objs.append((pixel_set, size_cat))

        labeled_pred_objs = []
        for coords in pred_objects:
            area = len(coords[0])
            size_cat = "small" if area < dynamic_threshold else "large"
            pixel_set = set(zip(coords[0], coords[1]))
            labeled_pred_objs.append((pixel_set, size_cat))

        matched_gt = set()
        matched_pred = set()

        for gt_idx, (gt_pixels, gt_size_cat) in enumerate(labeled_gt_objs):
            best_iou = 0.0
            best_pred_idx = None
            
            for pred_idx, (pred_pixels, pred_size_cat) in enumerate(labeled_pred_objs):
                intersection = len(gt_pixels.intersection(pred_pixels))
                union = len(gt_pixels.union(pred_pixels))
                iou = intersection / union if union > 0 else 0.0

                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = pred_idx
            
            if best_iou >= iou_threshold and best_pred_idx is not None:
                if gt_size_cat == "small":
                    tp_small += 1
                else:
                    tp_large += 1
                matched_gt.add(gt_idx)
                matched_pred.add(best_pred_idx)

        for gt_idx, (_, gt_size_cat) in enumerate(labeled_gt_objs):
            if gt_idx not in matched_gt:
                if gt_size_cat == "small":
                    fn_small += 1
                else:
                    fn_large += 1

        for pred_idx, (_, pred_size_cat) in enumerate(labeled_pred_objs):
            if pred_idx not in matched_pred:
                if pred_size_cat == "small":
                    fp_small += 1
                else:
                    fp_large += 1

        for coords, size_cat in labeled_gt_objs:
            if size_cat == "small":
                total_small_gt += 1
            else:
                total_large_gt += 1
        
        for coords, size_cat in labeled_pred_objs:
            if size_cat == "small":
                total_small_pred += 1
            else:
                total_large_pred += 1

    # --- 3) Compute metrics ---
    precision_small = tp_small / (tp_small + fp_small + 1e-10)
    recall_small = tp_small / (tp_small + fn_small + 1e-10)
    f1_small = 2 * precision_small * recall_small / (precision_small + recall_small + 1e-10)

    precision_large = tp_large / (tp_large + fp_large + 1e-10)
    recall_large = tp_large / (tp_large + fn_large + 1e-10)
    f1_large = 2 * precision_large * recall_large / (precision_large + recall_large + 1e-10)

    # --- 4) Print summary table ---
    print(f"\n{'Metric':<30}{'Small Objects':<20}{'Large Objects':<20}")
    print("="*70)
    print(f"{'Class Name':<30}{class_label:<20}")
    print(f"{'Size Threshold (pixels)':<30}{dynamic_threshold:<20.2f}")
    print("="*70)
    print(f"{'Total Ground Truths':<30}{total_small_gt:<20}{total_large_gt:<20}")
    print(f"{'Total Predicted':<30}{total_small_pred:<20}{total_large_pred:<20}")
    print(f"{'Precision':<30}{precision_small:<20.4f}{precision_large:<20.4f}")
    print(f"{'Recall':<30}{recall_small:<20.4f}{recall_large:<20.4f}")
    print(f"{'F1 Score':<30}{f1_small:<20.4f}{f1_large:<20.4f}")
    print(f"{'False Positives (FP)':<30}{fp_small:<20}{fp_large:<20}")
    print(f"{'False Negatives (FN)':<30}{fn_small:<20}{fn_large:<20}")
    print(f"{'True Positives (TP)':<30}{tp_small:<20}{tp_large:<20}")
    print("="*70)

    return precision_small, recall_small, f1_small, precision_large, recall_large, f1_large

