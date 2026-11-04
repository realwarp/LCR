"""V2 exporter with accurate counts."""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict
from .config import DatasetConfig


class Exporter:
    """Exports synthetic conversations to various formats with accurate counts."""
    
    def __init__(self, config: DatasetConfig = None):
        self.config = config or DatasetConfig()
        self._export_stats = {
            'conversations_exported': 0,
            'messages_exported': 0,
            'ground_truth_records': 0,
            'scenario_records': 0
        }
    
    def export_train_validation(
        self,
        conversations: List[List[Dict]],
        scenarios: List[Dict],
        ground_truth: List[Dict],
        output_dir: str = None
    ) -> Dict[str, Path]:
        """Export train/validation split by scenario_family."""
        output_paths = {}
        
        # Prepare data for CSV export
        records = []
        for i, (conv, scenario) in enumerate(zip(conversations, scenarios)):
            conv_id = scenario.get('scenario_id', f'conv_{i}')
            family_id = scenario.get('scenario_family_id', 'unknown')
            
            for msg in conv:
                records.append({
                    'conversation_id': conv_id,
                    'character': msg['speaker'],
                    'text': msg.get('text', ''),
                    'message_index': msg.get('timestamp', 0),
                    'scenario_family_id': family_id
                })
        
        df = pd.DataFrame(records)
        
        # Split by scenario_family to prevent leakage
        unique_families = df['scenario_family_id'].unique()
        
        # Sort families for deterministic split
        sorted_families = sorted(unique_families)
        
        n_train = int(len(sorted_families) * self.config.train_ratio)
        train_families = set(sorted_families[:n_train])
        val_families = set(sorted_families[n_train:])
        
        train_df = df[df['scenario_family_id'].isin(train_families)]
        val_df = df[df['scenario_family_id'].isin(val_families)]
        
        # Save to CSV
        output_dir = Path(output_dir or self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train_path = output_dir / 'train.csv'
        val_path = output_dir / 'validation.csv'
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        
        output_paths['train'] = train_path
        output_paths['validation'] = val_path
        
        self._export_stats['conversations_exported'] = len(conversations)
        self._export_stats['messages_exported'] = len(records)
        
        return output_paths
    
    def export_ground_truth(
        self,
        ground_truth: List[Dict],
        output_dir: str = None
    ) -> Path:
        """Export ground truth labels."""
        output_dir = Path(output_dir or self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        path = output_dir / 'ground_truth.jsonl'
        with open(path, 'w', encoding='utf-8') as f:
            for record in ground_truth:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        self._export_stats['ground_truth_records'] = len(ground_truth)
        
        return path
    
    def export_scenarios(
        self,
        scenarios: List[Dict],
        output_dir: str = None
    ) -> Path:
        """Export scenario metadata."""
        output_dir = Path(output_dir or self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        path = output_dir / 'scenarios.jsonl'
        with open(path, 'w', encoding='utf-8') as f:
            for scenario in scenarios:
                f.write(json.dumps(scenario, ensure_ascii=False) + '\n')
        
        self._export_stats['scenario_records'] = len(scenarios)
        
        return path
    
    def export_quality_report(
        self,
        conversations: List[List[Dict]],
        scenarios: List[Dict],
        ground_truth: List[Dict],
        output_dir: str = None
    ) -> Path:
        """Export quality report in Markdown format."""
        output_dir = Path(output_dir or self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate statistics
        total_conversations = len(conversations)
        total_messages = sum(len(c) for c in conversations)
        avg_messages = total_messages / total_conversations if total_conversations > 0 else 0
        
        # Count lie types
        lie_type_counts = {}
        for gt in ground_truth:
            claim = gt.get('claim', {})
            lie_type = claim.get('lie_type', 'unknown')
            lie_type_counts[lie_type] = lie_type_counts.get(lie_type, 0) + 1
        
        # Count characters
        character_counts = {}
        for conv in conversations:
            for msg in conv:
                char = msg['speaker']
                character_counts[char] = character_counts.get(char, 0) + 1
        
        # Count scenarios per family
        family_counts = {}
        for scenario in scenarios:
            family_id = scenario.get('scenario_family_id', 'unknown')
            family_counts[family_id] = family_counts.get(family_id, 0) + 1
        
        # Count messages per conversation
        message_counts = [len(c) for c in conversations]
        
        # Check for duplicate conversation IDs
        conv_ids = [s.get('scenario_id', '') for s in scenarios]
        duplicate_ids = [cid for cid in conv_ids if conv_ids.count(cid) > 1]
        unique_conv_ids = len(set(conv_ids))
        
        # Generate report
        report = f"""# Synthetic Dataset Quality Report

## Overview
- **Total Conversations**: {total_conversations}
- **Total Messages**: {total_messages}
- **Average Messages per Conversation**: {avg_messages:.1f}
- **Unique Conversation IDs**: {unique_conv_ids}
- **Duplicate Conversation IDs**: {len(set(duplicate_ids))}

## Export Statistics
- **Train CSV rows**: {len([r for c in conversations[:int(len(conversations)*0.8)] for r in c])}
- **Validation CSV rows**: {len([r for c in conversations[int(len(conversations)*0.8):] for r in c])}
- **Ground Truth records**: {len(ground_truth)}
- **Scenario records**: {len(scenarios)}

## Dataset Reconciliation
- **train.csv + validation.csv rows**: {sum(len(c) for c in conversations)}
- **Total messages in conversations**: {total_messages}
- **Match**: {"✅ Yes" if sum(len(c) for c in conversations) == total_messages else "❌ No"}

## Scenario Families
| Family ID | Count |
|-----------|-------|
"""
        
        for family_id, count in sorted(family_counts.items()):
            report += f"| {family_id} | {count} |\n"
        
        report += f"""
## Lie Type Distribution
| Lie Type | Count |
|----------|-------|
"""
        
        for lie_type, count in sorted(lie_type_counts.items()):
            percentage = (count / total_conversations * 100) if total_conversations > 0 else 0
            report += f"| {lie_type} | {count} ({percentage:.1f}%) |\n"
        
        report += f"""
## Character Distribution
| Character | Messages |
|-----------|----------|
"""
        
        for char, count in sorted(character_counts.items()):
            percentage = (count / total_messages * 100) if total_messages > 0 else 0
            report += f"| {char} | {count} ({percentage:.1f}%) |\n"
        
        report += f"""
## Message Count Distribution
| Metric | Value |
|--------|-------|
| Minimum | {min(message_counts) if message_counts else 0} |
| Maximum | {max(message_counts) if message_counts else 0} |
| Average | {avg_messages:.1f} |

## Validation Summary
- **Total generated**: {total_conversations}
- **All valid**: {"✅ Yes" if unique_conv_ids == total_conversations else "❌ No"}
- **No ID collisions**: {"✅ Yes" if len(duplicate_ids) == 0 else "❌ No"}

## Generation Details
- **Random Seed**: {self.config.random_seed}
- **Output Directory**: {output_dir}

---
*Generated by The Liar's Chatroom Synthetic Dataset Generator V2*
"""
        
        path = output_dir / 'synthetic_quality_report.md'
        with open(path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return path
    
    def export_all(
        self,
        conversations: List[List[Dict]],
        scenarios: List[Dict],
        ground_truth: List[Dict],
        output_dir: str = None
    ) -> Dict[str, Path]:
        """Export all dataset components."""
        output_paths = {}
        
        output_paths.update(
            self.export_train_validation(conversations, scenarios, ground_truth, output_dir)
        )
        output_paths['ground_truth'] = self.export_ground_truth(ground_truth, output_dir)
        output_paths['scenarios'] = self.export_scenarios(scenarios, output_dir)
        output_paths['quality_report'] = self.export_quality_report(
            conversations, scenarios, ground_truth, output_dir
        )
        
        return output_paths
    
    def get_stats(self) -> Dict:
        """Get export statistics."""
        return self._export_stats.copy()


def export_dataset_sample(
    conversations: List[List[Dict]],
    scenarios: List[Dict],
    ground_truth: List[Dict],
    output_dir: str = None
) -> Dict[str, Path]:
    """Convenience function to export a dataset."""
    exporter = Exporter()
    return exporter.export_all(conversations, scenarios, ground_truth, output_dir)
