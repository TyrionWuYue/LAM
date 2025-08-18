#!/usr/bin/env python3
"""
Simple inference script for LAM using preprocessed data.
Load data -> model forward -> save result images and videos.
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from safetensors.torch import load_file
from collections import defaultdict
import json
import time
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip

from lam.runners.infer.head_utils import load_flame_params, prepare_motion_seqs, preprocess_image
from lam.models import ModelLAM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/inference/lam-20k-8gpu.yaml", help="Config file path")
    parser.add_argument("--model_path", type=str, default="/inspire/hdd/project/urbanlowaltitude/fengkairui-25026/wuyue/FastAvatar/model_zoo/lam_models/releases/lam/lam-20k/step_045500/model.safetensors", help="Model safetensors file path")
    parser.add_argument("--data_dir", type=str, default="/inspire/hdd/project/urbanlowaltitude/fengkairui-25026/wuyue/nersemble_FLAME/sequence_EXP-5-mouth_part-4/247/cam_222200037", help="Input data directory")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--input_frame", type=int, default=0, help="Input frame number (0-based, default: 0 for first frame)")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, cpu, cuda, cuda:0, cuda:1, etc.)")
    parser.add_argument("--fps", type=int, default=30, help="FPS for output videos (default: 30)")
    parser.add_argument("--save_video", action="store_true", help="Save videos in addition to images")
    return parser.parse_args()

def load_preprocessed_data(data_dir, quiet=False):
    """
    Load preprocessed data from a directory containing processed_data subdirectories.
    
    Args:
        data_dir: Path to the data directory (e.g., /Data/wuyue/nersemble_FLAME/sequence_EXP-1-head_part-7/318/cam_222200037)
    
    Returns:
        dict: Dictionary containing loaded data
    """
    if not quiet:
        print(f"Loading data from: {data_dir}")
    
    # Check if processed_data directory exists
    processed_data_dir = os.path.join(data_dir, "processed_data")
    if not os.path.exists(processed_data_dir):
        raise ValueError(f"processed_data directory not found at {processed_data_dir}")
    
    # Find all frame subdirectories (00000, 00001, etc.)
    frame_dirs = []
    for item in os.listdir(processed_data_dir):
        item_path = os.path.join(processed_data_dir, item)
        if os.path.isdir(item_path) and item.isdigit():
            frame_dirs.append((int(item), item_path))
    
    # Sort by frame number
    frame_dirs.sort(key=lambda x: x[0])
    
    if not frame_dirs:
        raise ValueError(f"No frame directories found in {processed_data_dir}")
    
    if not quiet:
        print(f"Found {len(frame_dirs)} frame directories")
    
    # Load transforms.json for camera poses
    transforms_json = os.path.join(data_dir, "transforms.json")
    if os.path.exists(transforms_json):
        with open(transforms_json, 'r') as f:
            transforms_data = json.load(f)
        if not quiet:
            print("Loaded transforms.json for camera poses")
    else:
        if not quiet:
            print("Warning: transforms.json not found, using identity poses")
        transforms_data = None
    
    # Load data from each frame directory
    rgbs, masks, intrs, bg_colors, c2ws = [], [], [], [], []
    flame_params = defaultdict(list)
    
    for frame_num, frame_dir in frame_dirs:
        if not quiet:
            print(f"Loading frame {frame_num:05d} from {frame_dir}")
        
        # Load processed data files
        rgb_path = os.path.join(frame_dir, 'rgb.npy')
        mask_path = os.path.join(frame_dir, 'mask.npy')
        intr_path = os.path.join(frame_dir, 'intrs.npy')
        bg_color_path = os.path.join(frame_dir, 'bg_color.npy')
        
        # Check if all required files exist
        required_files = [rgb_path, mask_path, intr_path, bg_color_path]
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        # Load data
        rgb = torch.from_numpy(np.load(rgb_path)).float()
        mask = torch.from_numpy(np.load(mask_path)).float()
        intr = torch.from_numpy(np.load(intr_path)).float()
        bg_color = np.load(bg_color_path)
        
        # Add batch dimension if needed
        if len(rgb.shape) == 3:
            rgb = rgb.unsqueeze(0)  # [1, 3, H, W]
        if len(mask.shape) == 3:
            mask = mask.unsqueeze(0)  # [1, 1, H, W]
        
        # Load camera pose from transforms.json
        if transforms_data is not None:
            # Find the frame info in transforms.json
            frame_info = None
            for f in transforms_data['frames']:
                if str(frame_num).zfill(5) in f.get('file_path', ''):
                    frame_info = f
                    break
            if frame_info is None:
                try:
                    frame_info = transforms_data['frames'][frame_num]
                except:
                    frame_info = None
            
            if frame_info is not None:
                # Load camera pose using the same method as training
                c2w = torch.eye(4)
                c2w = np.array(frame_info["transform_matrix"])
                c2w[:3, 1:3] *= -1
                c2w = torch.FloatTensor(c2w)
            else:
                c2w = torch.eye(4)
        else:
            c2w = torch.eye(4)
        
        rgbs.append(rgb)
        masks.append(mask)
        intrs.append(intr)
        bg_colors.append(bg_color.item())
        c2ws.append(c2w)
    
    # Load FLAME parameters
    flame_param_dir = os.path.join(data_dir, "flame_param")
    if os.path.exists(flame_param_dir):
        flame_files = []
        for frame_num, _ in frame_dirs:
            flame_file = os.path.join(flame_param_dir, f"{frame_num:05d}.npz")
            if os.path.exists(flame_file):
                flame_files.append((frame_num, flame_file))
        
        if flame_files:
            if not quiet:
                print(f"Found {len(flame_files)} FLAME parameter files")
            for frame_num, flame_file in flame_files:
                flame_param = load_flame_params(flame_file)
                for k, v in flame_param.items():
                    flame_params[k].append(v)
        else:
            if not quiet:
                print("Warning: No FLAME parameter files found")
    else:
        if not quiet:
            print("Warning: flame_param directory not found")
    
    # Load canonical FLAME parameters for shape
    canonical_flame_path = os.path.join(data_dir, "canonical_flame_param.npz")
    if os.path.exists(canonical_flame_path):
        canonical_params = np.load(canonical_flame_path, allow_pickle=True)
        canonical_shape = torch.from_numpy(canonical_params['shape']).float()
        if not quiet:
            print("Loaded canonical FLAME shape parameters")
    else:
        if not quiet:
            print("Warning: canonical_flame_param.npz not found, using zero shape")
        canonical_shape = torch.zeros(300)
    
    # Stack all data
    rgbs = torch.cat(rgbs, dim=0)  # [N, 3, H, W]
    masks = torch.cat(masks, dim=0)  # [N, 1, H, W]
    intrs = torch.stack(intrs, dim=0)  # [N, 4, 4]
    bg_colors = torch.tensor(bg_colors, dtype=torch.float32).unsqueeze(-1).repeat(1, 3)  # [N, 3]
    c2ws = torch.stack(c2ws, dim=0)  # [N, 4, 4]
    
    # Stack FLAME parameters
    for k in flame_params:
        flame_params[k] = torch.stack(flame_params[k])  # [N, ...]
    
    # Use canonical shape for all frames
    num_frames = len(frame_dirs)
    flame_params['betas'] = canonical_shape  # [N, 300]
    
    # Add batch dimension to match training format
    rgbs = rgbs.unsqueeze(0)  # [1, N, 3, H, W]
    masks = masks.unsqueeze(0)  # [1, N, 1, H, W]
    intrs = intrs.unsqueeze(0)  # [1, N, 4, 4]
    bg_colors = bg_colors.unsqueeze(0)  # [1, N, 3]
    c2ws = c2ws.unsqueeze(0)  # [1, N, 4, 4]
    
    for k in flame_params:
        flame_params[k] = flame_params[k].unsqueeze(0)  # [1, N, ...]
    
    if not quiet:
        print(f"Successfully loaded {num_frames} frames")
        print(f"RGB shape: {rgbs.shape}")
        print(f"Mask shape: {masks.shape}")
        print(f"Intrinsics shape: {intrs.shape}")
        print(f"Camera poses shape: {c2ws.shape}")
        print(f"Background colors shape: {bg_colors.shape}")
        print(f"FLAME parameters: {list(flame_params.keys())}")
    
    return {
        'rgbs': rgbs,
        'masks': masks,
        'intrs': intrs,
        'c2ws': c2ws,
        'bg_colors': bg_colors,
        'flame_params': flame_params
    }


def infer_and_save(config, model_path, data_dir, output_dir, input_frame=0, device="auto", quiet=False, save_images=True, fps=30, save_video=False):
    import torch._dynamo
    torch._dynamo.config.disable = True
    cfg = OmegaConf.load(config)
    device = get_device(device, quiet)
    if not quiet:
        print(f"Using device: {device}")
    model = build_model(cfg, model_path, quiet).to(device)
    model.eval()
    
    if not quiet:
        print("Loading data...")
    data = load_preprocessed_data(data_dir, quiet)

    # Check if input frame is valid
    num_frames = data['rgbs'].shape[1]
    if input_frame >= num_frames:
        raise ValueError(f"Input frame {input_frame} is out of range. Available frames: 0-{num_frames-1}")
    
    if not quiet:
        print(f"Using frame {input_frame} as input (0-based index)")

    # Use specified frame as input
    input_image = data['rgbs'][:, input_frame:input_frame+1].to(device)  # [1, 1, 3, H, W]
    input_c2ws = data['c2ws'][:, input_frame:input_frame+1].to(device)   # [1, 1, 4, 4]
    input_intrs = data['intrs'][:, input_frame:input_frame+1].to(device) # [1, 1, 4, 4]
    # Set input background colors to white [1, 1, 1]
    input_bg_colors = torch.ones_like(data['bg_colors'][:, input_frame:input_frame+1].to(device))  # White background
    input_flame_params = {k: v[:, input_frame:input_frame+1].to(device) for k, v in data['flame_params'].items()}

    # Use all frames as target for rendering with white background
    target_c2ws = data['c2ws'].to(device)
    target_intrs = data['intrs'].to(device)
    # Set background colors to white [1, 1, 1]
    target_bg_colors = torch.ones_like(data['bg_colors'].to(device))  # White background
    target_flame_params = {k: v.to(device) for k, v in data['flame_params'].items()}

    import time
    start_time = time.time()
    with torch.no_grad():
        # LAM model infer_single_view pass
        res = model.infer_single_view(
            image=input_image,
            source_c2ws=input_c2ws,
            source_intrs=input_intrs,
            render_c2ws=target_c2ws,
            render_intrs=target_intrs,
            render_bg_colors=target_bg_colors,
            flame_params=target_flame_params
        )
    end_time = time.time()
    total_time = end_time - start_time
    if not quiet:
        print(f"\n=== Inference Summary ===")
        print(f"Input frame: {input_frame}")
        print(f"Output frames: {target_c2ws.shape[1]}")
        print(f"Total inference time: {total_time:.2f} seconds")
        print(f"Average time per frame: {total_time/target_c2ws.shape[1]:.2f} seconds")
        print(f"FPS: {target_c2ws.shape[1]/total_time:.2f}")
        print(f"========================\n")
    
    # Get the rendered RGB output (exactly like one_shot_infer.py)
    if "comp_rgb" in res:
        pred_rgb = res["comp_rgb"]
    elif "sliced_comp_rgb" in res:
        pred_rgb = res["sliced_comp_rgb"]
    else:
        raise ValueError("No suitable RGB output found in results")
    
    # Save results if requested
    if save_images:
        results = pred_rgb.detach().cpu().numpy()
        if not quiet:
            print(f"Pred_rgb shape: {pred_rgb.shape}")
            print(f"Results numpy shape: {results.shape}")
        
        # Get masks for background replacement
        if 'masks' in data:
            masks = data['masks'].detach().cpu().numpy()
        else:
            # Create dummy masks if not available
            masks = np.ones((1, results.shape[1], 1, results.shape[-2], results.shape[-1]))
        
        # Load original RGB images and create ground truth with white background
        # Initialize strictly from masks to avoid NHWC/NCHW ambiguity
        num_frames = masks.shape[1]
        H, W = masks.shape[-2], masks.shape[-1]
        gt_results_white = np.zeros((1, num_frames, 3, H, W), dtype=np.float32)
        
        # Load transforms.json to get original image paths
        transforms_json = os.path.join(data_dir, "transforms.json")
        if os.path.exists(transforms_json):
            with open(transforms_json, 'r') as f:
                transforms_data = json.load(f)
            
            # Determine the number of frames from masks for consistency
            num_frames = masks.shape[1]
            
            for i in range(num_frames):  # For each frame
                # Get original image path from transforms.json
                if i < len(transforms_data['frames']):
                    frame_info = transforms_data['frames'][i]
                    original_image_path = os.path.join(data_dir, frame_info['file_path'])
                    
                    if os.path.exists(original_image_path):
                        # Load original RGB image
                        from PIL import Image
                        import cv2
                        original_rgb = np.array(Image.open(original_image_path)) / 255.0  # Normalize to [0,1]
                        if len(original_rgb.shape) == 3:
                            original_rgb = original_rgb.transpose(2, 0, 1)  # (H, W, 3) -> (3, H, W)
                        
                        # Get mask for this frame and resize to match render resolution
                        mask_2d = masks[0, i, 0]  # Shape: (H, W) - this is already 512x512
                        
                        # Resize original image to match render resolution (512x512)
                        target_h, target_w = mask_2d.shape
                        original_rgb_resized = np.zeros((3, target_h, target_w))
                        for c in range(3):
                            original_rgb_resized[c] = cv2.resize(original_rgb[c], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                        
                        # Apply white background using mask
                        # Transpose to (H, W, 3) for easier broadcasting
                        rgb_frame = original_rgb_resized.transpose(1, 2, 0)  # Shape: (H, W, 3)
                        # Keep foreground, replace background with white
                        rgb_frame_white = rgb_frame * mask_2d[:, :, None] + (1 - mask_2d[:, :, None]) * 1.0
                        # Transpose back to (3, H, W)
                        gt_results_white[0, i] = rgb_frame_white.transpose(2, 0, 1)
                    else:
                        # Fallback: use the processed data but apply white background
                        gt_results = data['rgbs'].detach().cpu().numpy()
                        mask_2d = masks[0, i, 0]  # Shape: (H, W)
                        gt_frame = gt_results[0, i].transpose(1, 2, 0)  # Shape: (H, W, 3)
                        gt_frame_white = gt_frame * mask_2d[:, :, None] + (1 - mask_2d[:, :, None]) * 1.0
                        gt_results_white[0, i] = gt_frame_white.transpose(2, 0, 1)
                else:
                    # Fallback: use the processed data but apply white background
                    gt_results = data['rgbs'].detach().cpu().numpy()
                    mask_2d = masks[0, i, 0]  # Shape: (H, W)
                    gt_frame = gt_results[0, i].transpose(1, 2, 0)  # Shape: (H, W, 3)
                    gt_frame_white = gt_frame * mask_2d[:, :, None] + (1 - mask_2d[:, :, None]) * 1.0
                    gt_results_white[0, i] = gt_frame_white.transpose(2, 0, 1)
        else:
            # Fallback: use the processed data but apply white background
            gt_results = data['rgbs'].detach().cpu().numpy()
            num_frames = masks.shape[1]
            for i in range(num_frames):
                mask_2d = masks[0, i, 0]  # Shape: (H, W)
                gt_frame = gt_results[0, i].transpose(1, 2, 0)  # Shape: (H, W, 3)
                gt_frame_white = gt_frame * mask_2d[:, :, None] + (1 - mask_2d[:, :, None]) * 1.0
                gt_results_white[0, i] = gt_frame_white.transpose(2, 0, 1)
        
        save_results(results, gt_results_white, masks, output_dir, quiet, fps, save_video)
    
    return {
        'inference_time': total_time,
        'fps': target_c2ws.shape[1]/total_time,
        'num_frames': target_c2ws.shape[1],
        'input_frame': input_frame,
        'data_path': data_dir,
    }


def build_model(cfg, model_path, quiet=False):
    """Build and load LAM model."""
    model = ModelLAM(**cfg.model)
    
    # Load weights directly from safetensors file
    if not quiet:
        print(f"Loading model from {model_path}")
    
    if model_path.endswith('.safetensors'):
        ckpt = load_file(model_path, device='cpu')
    else:
        ckpt = torch.load(model_path, map_location='cpu')
    
    # Load state dict with shape checking
    state_dict = model.state_dict()
    for k, v in ckpt.items():
        if k in state_dict:
            if state_dict[k].shape == v.shape:
                state_dict[k].copy_(v)
            else:
                if not quiet:
                    print(f"WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.")
        else:
            if not quiet:
                print(f"WARN] unexpected param {k}: {v.shape}")
    
    if not quiet:
        print("Model loaded successfully")
    
    return model


def get_device(device_str, quiet=False):
    """Get device based on device string."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    elif device_str == "cpu":
        return torch.device('cpu')
    elif device_str.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(device_str)
        else:
            if not quiet:
                print(f"Warning: CUDA not available, falling back to CPU")
            return torch.device('cpu')
    else:
        if not quiet:
            print(f"Warning: Unknown device '{device_str}', falling back to CPU")
        return torch.device('cpu')


def save_results(results, gt_results, masks, output_dir, quiet=False, fps=30, save_video=False):
    """Save inference results and ground truth as images and videos with white backgrounds using masks."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for ground truth and inference results
    gt_dir = os.path.join(output_dir, "gt")
    infer_dir = os.path.join(output_dir, "infer")
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(infer_dir, exist_ok=True)
    
    # Convert to uint8 images
    pred_images = (np.clip(results, 0, 1.0) * 255).astype(np.uint8)
    gt_images = (np.clip(gt_results, 0, 1.0) * 255).astype(np.uint8)
    
    if not quiet:
        print(f"Results shape: {pred_images.shape}")
        print(f"Ground truth shape: {gt_images.shape}")
        print(f"Masks shape: {masks.shape}")
    
    # Simplify shape handling - just use squeeze/transpose as needed
    if len(pred_images.shape) == 4 and pred_images.shape[-1] == 3:  # (N, H, W, C)
        pred_images = np.transpose(pred_images, (0, 3, 1, 2))  # (N, C, H, W)
    
    if len(gt_images.shape) == 5:  # (B, N, C, H, W)
        gt_images = gt_images.squeeze(0)  # (N, C, H, W)
    
    if len(masks.shape) == 5:  # (B, N, 1, H, W)
        masks = masks.squeeze(0)  # (N, 1, H, W)
    
    # Save each frame
    for i in range(pred_images.shape[0]):
        pred_img = pred_images[i]  # Shape: (C, H, W)
        gt_img = gt_images[i]      # Shape: (C, H, W)
        mask = masks[i]            # Shape: (1, H, W)
        
        # Convert to (H, W, C) for PIL
        if pred_img.shape[0] == 3:  # RGB
            pred_img = np.transpose(pred_img, (1, 2, 0))
        elif pred_img.shape[0] == 1:  # Grayscale
            pred_img = pred_img[0]  # Remove channel dimension
        
        if gt_img.shape[0] == 3:  # RGB
            gt_img = np.transpose(gt_img, (1, 2, 0))
        elif gt_img.shape[0] == 1:  # Grayscale
            gt_img = gt_img[0]  # Remove channel dimension
        
        # Save prediction to infer/ subdirectory
        pred_save_path = os.path.join(infer_dir, f"{i:04d}.png")
        Image.fromarray(pred_img.astype(np.uint8)).save(pred_save_path)
        
        # Save ground truth to gt/ subdirectory (background already replaced with white)
        gt_save_path = os.path.join(gt_dir, f"{i:04d}.png")
        Image.fromarray(gt_img.astype(np.uint8)).save(gt_save_path)
    
    if not quiet:
        print(f"Saved {pred_images.shape[0]} prediction images to {infer_dir}")
        print(f"Saved {gt_images.shape[0]} ground truth images to {gt_dir}")
    
    # Generate videos if requested
    if save_video:
        if not quiet:
            print(f"Generating videos with FPS: {fps}")
        
        # Prepare image sequences for video
        pred_image_list = []
        gt_image_list = []
        
        for i in range(pred_images.shape[0]):
            pred_img = pred_images[i]  # Shape: (C, H, W)
            gt_img = gt_images[i]      # Shape: (C, H, W)
            mask = masks[i]            # Shape: (1, H, W)
            
            # Convert to (H, W, C) for PIL
            if pred_img.shape[0] == 3:  # RGB
                pred_img = np.transpose(pred_img, (1, 2, 0))
            elif pred_img.shape[0] == 1:  # Grayscale
                pred_img = pred_img[0]  # Remove channel dimension
            
            if gt_img.shape[0] == 3:  # RGB
                gt_img = np.transpose(gt_img, (1, 2, 0))
            elif gt_img.shape[0] == 1:  # Grayscale
                gt_img = gt_img[0]  # Remove channel dimension
            
            pred_image_list.append(pred_img.astype(np.uint8))
            gt_image_list.append(gt_img.astype(np.uint8))
        
        # Create video clips
        try:
            pred_clip = ImageSequenceClip(pred_image_list, fps=fps)
            gt_clip = ImageSequenceClip(gt_image_list, fps=fps)
            
            # Save videos
            pred_video_path = os.path.join(output_dir, "prediction.mp4")
            gt_video_path = os.path.join(output_dir, "ground_truth.mp4")
            
            pred_clip.write_videofile(pred_video_path, fps=fps, verbose=False, logger=None)
            gt_clip.write_videofile(gt_video_path, fps=fps, verbose=False, logger=None)
            
            if not quiet:
                print(f"Saved prediction video to: {pred_video_path}")
                print(f"Saved ground truth video to: {gt_video_path}")
                
        except Exception as e:
            if not quiet:
                print(f"Warning: Failed to generate videos: {e}")
                print("Make sure moviepy is installed: pip install moviepy")


def main():
    args = parse_args()
    results = infer_and_save(
        config=args.config,
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        input_frame=args.input_frame,
        device=args.device,
        quiet=False,
        fps=args.fps,
        save_video=args.save_video
    )
    print("\n=== Inference Results ===")
    print(f"Data directory: {args.data_dir}")
    print(f"Input frame: {args.input_frame}")
    print(f"Output frames: {results['num_frames']}")
    print(f"Inference time: {results['inference_time']:.2f} seconds")
    print(f"FPS: {results['fps']:.2f}")
    print(f"Output saved to: {args.output_dir}")
    print("=========================")


if __name__ == "__main__":
    main() 