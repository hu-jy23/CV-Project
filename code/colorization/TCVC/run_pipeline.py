import os
import subprocess
import sys

def run_commands():
    # Define the base paths
    test_dir = "./test"
    
    cmd0 = [
        "python", "video_to_frames.py",
        "--video_path", os.path.join(test_dir, "input.mp4"),
        "--color", os.path.join(test_dir, "color"),
        "--bw", os.path.join(test_dir, "input_frames")
    ]

    # Command 1: Extract reference frames (stage 1)
    cmd1 = [
        "python", "extract_ref_frame.py",
        "--input", os.path.join(test_dir, "input_frames"),
        "--output", os.path.join(test_dir, "stage1_input/stage1_input"),
        "--all_frame_output", os.path.join(test_dir, "stage2_input")
    ]
    
    
    # Command 2: Extract reference frames with ground truth
    cmd2 = [
        "python", "stage1/test.py",
        "--data_root_val", "./test/stage1_input",
        "--test_video_output_path", "./test/stage1_output"
    ]

    # cmd2 = [
    #     "python", "extract_ref_frame.py",
    #     "--input", os.path.join(test_dir, "input_frames"),
    #     "--output", os.path.join(test_dir, "stage1_input/stage1_input"),
    #     "--all_frame_output", os.path.join(test_dir, "stage2_input"),
    #     "--use_ground_truth_ref",
    #     "--gt_source", os.path.join(test_dir, "color"),
    #     "--gt_target", os.path.join(test_dir, "stage1_output")
    # ]
    
    # Command 3: Run stage2 colorization
    cmd3 = [
        "python", "stage2/inference_colorvid.py",
        "--test_path", os.path.join(test_dir, "stage2_input"),
        "--ref_path", os.path.join(test_dir, "stage1_output"),
        "--test_output_path", os.path.join(test_dir, "stage2_output")
    ]
    
    # Command 4: Combine frames into final video
    cmd4 = [
        "python", "image_sequence_to_video.py",
        "--input", os.path.join(test_dir, "stage2_output"),
        "--output", os.path.join(test_dir, "final_output"),
        "--fps", "30"
    ]
    
    # Execute commands in sequence
    # commands = [cmd0, cmd1, cmd2, cmd3, cmd4]
    commands = [cmd0, cmd1, cmd2, cmd3, cmd4]
    
    for i, cmd in enumerate(commands, 1):
        print(f"\nExecuting Command {i}: {' '.join(cmd)}")
        try:
            # Run the command and wait for completion
            result = subprocess.run(cmd, check=True)
            print(f"Command {i} completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error executing Command {i}: {e}")
            sys.exit(1)
    
    print("\nAll commands executed successfully!")

if __name__ == "__main__":
    # Create necessary directories if they don't exist
    os.makedirs("./test/stage1_input", exist_ok=True)
    os.makedirs("./test/stage2_input", exist_ok=True)
    os.makedirs("./test/stage1_output", exist_ok=True)
    os.makedirs("./test/stage2_output", exist_ok=True)
    os.makedirs("./test/final_output", exist_ok=True)
    
    run_commands()