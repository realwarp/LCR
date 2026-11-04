"""Dataset exporter for synthetic conversations."""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict
from .config import DatasetConfig


class Exporter:
    """Exports synthetic conversations to various formats."""
    
    def __init__(self, config: DatasetConfig = None):
        self.config = config or DatasetConfig()
        self._exported_count = 0
    
    def export_train_validation(
        self,
        conversations: List[List[Dict]],
        scenarios: List[Dict],
        output_dir: str = None
    ) -> Dict[str, Path]:
        """Export train/validation split."""
        output_paths = {}
        
        # Prepare data for CSV export
        records = []
        for conv, scenario in zip(conversations, scenarios):
            for msg in conv:
                records.append({
                    'id': f"{scenario['world']['name']}_{conv.index(msg)}_{msg['speaker']}",
                    'conversation_id': scenario['world']['name'],
                    'character': msg['speaker'],
                    'text': msg['message'],
                    'label': 1 if msg.get('is_liar', False) else 0,
                    'message_index': msg.get('timestamp', 0)
                })
        
        df = pd.DataFrame(records)
        
        # Split by conversation (scenario family) to prevent leakage
        unique_conversations = list(set(r['conversation_id'] for r in records))
        
        # Group by scenario family
        train_convs = unique_conversations[:int(len(unique_conversations) * self.config.train_ratio)]
        val_convs = unique_conversations[int(len(unique_conversations) * self.config.train_ratio):]
        
        train_df = df[df['conversation_id'].isin(train_convs)]
        val_df = df[df['conversation_id'].isin(val_convs)]
        
        # Save to CSV
        output_dir = Path(output_dir or self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train_path = output_dir / 'train.csv'
        val_path = output_dir / 'validation.csv'
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        
        output_paths['train'] = train_path
        output_paths['validation'] = val_path
        
        self._exported_count += len(records)
        
        return output_paths
    
    def export_ground_truth(
        self,
        scenarios: List[Dict],
        output_dir: str = None
    ) -> Path:
        """Export ground truth labels."""
        output_dir = Path(output_dir or self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        records = []
        for scenario in scenarios:
            record = {
                'conversation_id': scenario['world']['name'],
                'liar': scenario['liar'],
                'characters': scenario['characters'],
                'claim': scenario['claim'],
                'world': scenario['world']
            }
            records.append(record)
        
        path = output_dir / 'ground_truth.jsonl'
        with open(path, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        self._exported_count += len(records)
        
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
        
        self._exported_count += len(scenarios)
        
        return path
    
    def export_quality_report(
        self,
        conversations: List[List[Dict]],
        scenarios: List[Dict],
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
        for scenario in scenarios:
            claim = scenario.get('claim', {})
            lie_type = claim.get('lie_type', 'unknown')
            lie_type_counts[lie_type] = lie_type_counts.get(lie_type, 0) + 1
        
        # Count characters
        character_counts = {}
        for conv in conversations:
            for msg in conv:
                char = msg['speaker']
                character_counts[char] = character_counts.get(char, 0) + 1
        
        # Generate report
        report = f"""# Synthetic Dataset Quality Report

## Overview
- **Total Conversations**: {total_conversations}
- **Total Messages**: {total_messages}
- **Average Messages per Conversation**: {avg_messages:.1f}

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
## Generation Details
- **Random Seed**: {self.config.random_seed}
- **Output Directory**: {output_dir}
- **Exported Count**: {self._exported_count}

## Dataset Files
- Training: `train.csv`
- Validation: `validation.csv`
- Ground Truth: `ground_truth.jsonl`
- Scenarios: `scenarios.jsonl`

---
*Generated by The Liar's Chatroom Synthetic Dataset Generator*
"""
        
        path = output_dir / 'synthetic_quality_report.md'
        with open(path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return path
    
    def export_all(
        self,
        conversations: List[List[Dict]],
        scenarios: List[Dict],
        output_dir: str = None
    ) -> Dict[str, Path]:
        """Export all dataset components."""
        output_paths = {}
        
        output_paths.update(
            self.export_train_validation(conversations, scenarios, output_dir)
        )
        output_paths['ground_truth'] = self.export_ground_truth(scenarios, output_dir)
        output_paths['scenarios'] = self.export_scenarios(scenarios, output_dir)
        output_paths['quality_report'] = self.export_quality_report(
            conversations, scenarios, output_dir
        )
        
        return output_paths


def export_dataset_sample(
    conversations: List[List[Dict]],
    scenarios: List[Dict],
    output_dir: str = None
) -> Dict[str, Path]:
    """Convenience function to export a dataset."""
    exporter = Exporter()
    return exporter.export_all(conversations, scenarios, output_dir)
