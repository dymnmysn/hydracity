import os
from pandaset import DataSet, geometry
import numpy as np

if __name__=='__main__':
    # Path to the PandaSet dataset
    datapath = "/media/yasin/extra/projeler/pandaset-devkit/data"

    # Output path for saving the npz files
    output_path = "/media/yasin/extra/projeler/pandaset-devkit/output"  # Replace with your desired output directory

    # Initialize dataset
    dataset = DataSet(datapath)

    # Iterate over all sequences
    for seq_name in dataset.sequences():
        print(f"Processing sequence: {seq_name}")
        sequence = dataset[seq_name]
        sequence.load_lidar().load_semseg()
        sequence.lidar.set_sensor(0)  # Use the primary lidar sensor
        
        # Create sequence directory structure
        seq_dir = os.path.join(output_path, seq_name)
        images_dir = os.path.join(seq_dir, "images")
        labels_dir = os.path.join(seq_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        # Iterate over all indices in the sequence
        for idx in range(len(sequence.lidar._data_structure)):
            # Process lidar points
            pandar64_points = sequence.lidar[idx].to_numpy()
            ego_points = geometry.lidar_points_to_ego(pandar64_points[:, :3], sequence.lidar.poses[idx])
            ego_points = ego_points[:, [1, 0, 2]]  # Swap axes to desired order
            ego_points[..., 2] -= 1.75  # Adjust Z for ground clearance
            intensity = pandar64_points[:, 3:4]
            image = np.concatenate((ego_points, intensity), axis=1)
            
            # Process semantic segmentation labels
            label = sequence.semseg[idx].to_numpy()[:image.shape[0]].squeeze()
            
            # Save in 16-bit and 8-bit formats
            filename = f"{idx:06d}.npz"
            np.savez_compressed(os.path.join(images_dir, filename), image=image.astype(np.float16))
            np.savez_compressed(os.path.join(labels_dir, filename), label=label.astype(np.uint8))
            
            #print(f"Saved index {idx} for sequence {seq_name}")
            #intensity = np.clip(intensity,0,65)/65
            #intensity.mean(),intensity.std()

    print("Processing complete.")
