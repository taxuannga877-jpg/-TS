"""
过渡态预测的预测脚本

使用方法：
    python predict.py --checkpoint outputs/run_xxx/best_model.pt --input_dir test_data --output_dir predictions
"""
import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import yaml

sys.path.insert(0, os.path.dirname(__file__))

from models.ts_predictor_complete import create_reaction_centric_model
from data.dataset_complete import TSFFHybridDataset, collate_fn
from utils.metrics import calculate_rmsd
from torch.utils.data import DataLoader


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='预测TS结构')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='包含R和P xyz文件的输入目录')
    parser.add_argument('--output_dir', type=str, default='./predictions',
                       help='预测结果输出目录')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='预测批次大小')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备：cuda 或 cpu')
    parser.add_argument('--save_confidence', action='store_true',
                       help='保存置信度分数')
    
    return parser.parse_args()


def save_xyz(atoms, coords, filename):
    """保存坐标到XYZ文件"""
    with open(filename, 'w') as f:
        f.write(f"{len(atoms)}\n")
        f.write("Predicted TS structure\n")
        for atom, coord in zip(atoms, coords):
            f.write(f"{atom}  {coord[0]:.8f}  {coord[1]:.8f}  {coord[2]:.8f}\n")


def main():
    args = parse_args()
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint['config']
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print("Creating model...")
    model = create_reaction_centric_model(config['model']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded (trained for {checkpoint['epoch']} epochs)")
    print(f"✓ Best validation RMSD: {checkpoint['val_rmsd']:.4f} Å\n")
    
    # Load dataset (for prediction, we don't need ground truth TS)
    print(f"Loading data from {args.input_dir}...")
    
    # Note: This assumes test data has the same structure as train data
    # For actual test data without TS, you may need to modify the dataset class
    try:
        dataset = TSFFHybridDataset(
            data_dir=args.input_dir,
            cutoff=config['model']['cutoff'],
            augment=False  # No augmentation for prediction
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Note: If test data doesn't have TS files, you need to modify the dataset class")
        return
    
    print(f"✓ Found {len(dataset)} reactions\n")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prediction
    print("Predicting TS structures...")
    all_predictions = []
    all_confidences = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            data_r = batch['data_r'].to(device)
            data_p = batch['data_p'].to(device)
            
            # Predict
            ts_pred, alpha, delta, confidence, reaction_scores = model(data_r, data_p)
            
            # Process batch
            batch_size = data_r.batch.max().item() + 1
            for i in range(batch_size):
                mask = (data_r.batch == i)
                
                # Get predicted coordinates
                pred_coords = ts_pred[mask].cpu().numpy()
                atoms = [dataset.atomic_numbers_to_symbols[z.item()] for z in data_r.z[mask]]
                
                # Get confidence (average over all atoms)
                avg_confidence = confidence[mask].mean().item()
                
                all_predictions.append({
                    'coords': pred_coords,
                    'atoms': atoms,
                    'confidence': avg_confidence,
                    'reaction_id': batch_idx * args.batch_size + i
                })
                all_confidences.append(avg_confidence)
    
    # Save predictions
    print(f"\nSaving predictions to {args.output_dir}...")
    
    for idx, pred in enumerate(all_predictions):
        # Create reaction directory
        rxn_dir = os.path.join(args.output_dir, f'rxn_{idx:04d}')
        os.makedirs(rxn_dir, exist_ok=True)
        
        # Save TS coordinates
        ts_file = os.path.join(rxn_dir, 'ts_pred.xyz')
        save_xyz(pred['atoms'], pred['coords'], ts_file)
        
        # Save confidence if requested
        if args.save_confidence:
            conf_file = os.path.join(rxn_dir, 'confidence.txt')
            with open(conf_file, 'w') as f:
                f.write(f"Average Confidence: {pred['confidence']:.4f}\n")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Prediction completed!")
    print(f"{'='*60}")
    print(f"Total reactions: {len(all_predictions)}")
    print(f"Average confidence: {np.mean(all_confidences):.4f}")
    print(f"Min confidence: {np.min(all_confidences):.4f}")
    print(f"Max confidence: {np.max(all_confidences):.4f}")
    print(f"Results saved in: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

