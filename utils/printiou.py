def print_miou_waymo(classwise_iou, mean_iou, total_iou):
    waymo = {
        0: 'TYPE_UNDEFINED',
        1: 'TYPE_CAR',
        2: 'TYPE_TRUCK',
        3: 'TYPE_BUS',
        4: 'TYPE_OTHER_VEHICLE',
        5: 'TYPE_MOTORCYCLIST',
        6: 'TYPE_BICYCLIST',
        7: 'TYPE_PEDESTRIAN',
        8: 'TYPE_SIGN',
        9: 'TYPE_TRAFFIC_LIGHT',
        10: 'TYPE_POLE',
        11: 'TYPE_CONSTRUCTION_CONE',
        12: 'TYPE_BICYCLE',
        13: 'TYPE_MOTORCYCLE',
        14: 'TYPE_BUILDING',
        15: 'TYPE_VEGETATION',
        16: 'TYPE_TREE_TRUNK',
        17: 'TYPE_CURB',
        18: 'TYPE_ROAD',
        19: 'TYPE_LANE_MARKER',
        20: 'TYPE_OTHER_GROUND',
        21: 'TYPE_WALKABLE',
        22: 'TYPE_SIDEWALK'
    }

    print("Classwise IoU:")
    for i, iou in enumerate(classwise_iou, start=1):  # Start at 1 to skip class 0
        class_name = waymo.get(i, f"Class {i}")
        print(f"  {class_name}: {iou:.4f}" if not isinstance(iou, float) or not iou != iou else f"  {class_name}: N/A")

    print(f"\nMean IoU: {mean_iou:.4f}")
    print(f"Total IoU: {total_iou:.4f}")

def print_miou_kitti(classwise_iou, mean_iou, total_iou):
    kitti = {
        0: "unlabeled",
        1: "car",
        2: "bicycle",
        3: "motorcycle",
        4: "truck",
        5: "other-vehicle",
        6: "person",
        7: "bicyclist",
        8: "motorcyclist",
        9: "road",
        10: "parking",
        11: "sidewalk",
        12: "other-ground",
        13: "building",
        14: "fence",
        15: "vegetation",
        16: "trunk",
        17: "terrain",
        18: "pole",
        19: "traffic-sign"
    }

    print("Classwise IoU:")
    for i, iou in enumerate(classwise_iou, start=1):  # Start at 1 to skip class 0
        class_name = kitti.get(i, f"Class {i}")
        print(f"  {class_name}: {iou:.4f}" if not isinstance(iou, float) or not iou != iou else f"  {class_name}: N/A")

    print(f"\nMean IoU: {mean_iou:.4f}")
    print(f"Total IoU: {total_iou:.4f}")
