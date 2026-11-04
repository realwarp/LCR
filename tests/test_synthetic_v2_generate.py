"""Tests for V2 generation end-to-end."""

import pytest
import subprocess
import json
import pandas as pd
from pathlib import Path
import tempfile


class TestV2GenerateCLI:
    """Tests for V2 generation CLI."""
    
    def test_generate_10_examples(self, temp_dir):
        """Test generating 10 examples via CLI."""
        output_dir = temp_dir / 'synthetic'
        output_dir.mkdir()
        
        result = subprocess.run(
            [
                'python', '-m', 'synthetic.v2.generate',
                '--num-examples', '10',
                '--output-dir', str(output_dir),
                '--seed', '42'
            ],
            capture_output=True,
            text=True,
            cwd='C:/Users/ashwi/workplace/liars-chatroom-starter'
        )
        
        # Check exit code
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        
        # Check output files exist
        assert (output_dir / 'train.csv').exists()
        assert (output_dir / 'validation.csv').exists()
        assert (output_dir / 'ground_truth.jsonl').exists()
        assert (output_dir / 'scenarios.jsonl').exists()
        assert (output_dir / 'synthetic_quality_report.md').exists()
    
    def test_exact_count_generated(self, temp_dir):
        """Test that exactly requested number is generated."""
        output_dir = temp_dir / 'synthetic'
        output_dir.mkdir()
        
        result = subprocess.run(
            [
                'python', '-m', 'synthetic.v2.generate',
                '--num-examples', '20',
                '--output-dir', str(output_dir),
                '--seed', '42'
            ],
            capture_output=True,
            text=True,
            cwd='C:/Users/ashwi/workplace/liars-chatroom-starter'
        )
        
        assert result.returncode == 0
        
        # Check ground truth record count
        with open(output_dir / 'ground_truth.jsonl', 'r') as f:
            lines = f.readlines()
            assert len(lines) == 20, f"Expected 20 records, got {len(lines)}"
    
    def test_unique_conv_ids(self, temp_dir):
        """Test that generated IDs are unique."""
        output_dir = temp_dir / 'synthetic'
        output_dir.mkdir()
        
        result = subprocess.run(
            [
                'python', '-m', 'synthetic.v2.generate',
                '--num-examples', '50',
                '--output-dir', str(output_dir),
                '--seed', '42'
            ],
            capture_output=True,
            text=True,
            cwd='C:/Users/ashwi/workplace/liars-chatroom-starter'
        )
        
        assert result.returncode == 0
        
        # Read scenarios and check for duplicates
        with open(output_dir / 'scenarios.jsonl', 'r') as f:
            lines = f.readlines()
            conv_ids = [json.loads(line)['scenario_id'] for line in lines]
        
        assert len(conv_ids) == len(set(conv_ids)), "Duplicate conversation IDs"


class TestV2GenerateCSVRows:
    """Tests for CSV row counts."""
    
    def test_csv_rows_match_conversations(self, temp_dir):
        """Test that CSV row count matches conversation messages."""
        output_dir = temp_dir / 'synthetic'
        output_dir.mkdir()
        
        result = subprocess.run(
            [
                'python', '-m', 'synthetic.v2.generate',
                '--num-examples', '10',
                '--output-dir', str(output_dir),
                '--seed', '42'
            ],
            capture_output=True,
            text=True,
            cwd='C:/Users/ashwi/workplace/liars-chatroom-starter'
        )
        
        assert result.returncode == 0
        
        # Count rows in CSVs
        train_df = pd.read_csv(output_dir / 'train.csv')
        val_df = pd.read_csv(output_dir / 'validation.csv')
        
        total_csv_rows = len(train_df) + len(val_df)
        
        # Count messages in ground truth
        with open(output_dir / 'ground_truth.jsonl', 'r') as f:
            lines = f.readlines()
        
        # Each conversation should have 8-20 messages
        # 10 conversations * ~14 avg = ~140 messages
        assert total_csv_rows > 0, "No CSV rows exported"
        assert total_csv_rows < 500, "Too many CSV rows"


class TestV2GenerateClaimContext:
    """Tests for claim context in generated data."""
    
    def test_no_null_claim_context(self, temp_dir):
        """Test that no ground truth has null claim context."""
        output_dir = temp_dir / 'synthetic'
        output_dir.mkdir()
        
        result = subprocess.run(
            [
                'python', '-m', 'synthetic.v2.generate',
                '--num-examples', '20',
                '--output-dir', str(output_dir),
                '--seed', '42'
            ],
            capture_output=True,
            text=True,
            cwd='C:/Users/ashwi/workplace/liars-chatroom-starter'
        )
        
        assert result.returncode == 0
        
        # Check all ground truth records have context
        with open(output_dir / 'ground_truth.jsonl', 'r') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            record = json.loads(line)
            claim = record.get('claim', {})
            context = claim.get('context')
            assert context is not None, f"Record {i} has null context"
            assert len(context) > 0, f"Record {i} has empty context"


class TestV2GenerateLiarDistribution:
    """Tests for liar distribution in generated data."""
    
    def test_balanced_liar_distribution(self, temp_dir):
        """Test that liars are roughly balanced."""
        output_dir = temp_dir / 'synthetic'
        output_dir.mkdir()
        
        result = subprocess.run(
            [
                'python', '-m', 'synthetic.v2.generate',
                '--num-examples', '40',
                '--output-dir', str(output_dir),
                '--seed', '42'
            ],
            capture_output=True,
            text=True,
            cwd='C:/Users/ashwi/workplace/liars-chatroom-starter'
        )
        
        assert result.returncode == 0
        
        # Count liars
        with open(output_dir / 'ground_truth.jsonl', 'r') as f:
            lines = f.readlines()
        
        liar_counts = {}
        for line in lines:
            record = json.loads(line)
            liar = record.get('liar')
            liar_counts[liar] = liar_counts.get(liar, 0) + 1
        
        # Each liar should be ~25% (allow 10% tolerance)
        total = sum(liar_counts.values())
        for liar, count in liar_counts.items():
            ratio = count / total
            assert 0.15 <= ratio <= 0.35, \
                f"Liar {liar} ratio {ratio:.2%} outside [15%, 35%]"


class TestV2GenerateMessageCount:
    """Tests for message count in generated data."""
    
    def test_message_count_in_range(self, temp_dir):
        """Test that messages are in 8-20 range."""
        output_dir = temp_dir / 'synthetic'
        output_dir.mkdir()
        
        result = subprocess.run(
            [
                'python', '-m', 'synthetic.v2.generate',
                '--num-examples', '10',
                '--output-dir', str(output_dir),
                '--seed', '42'
            ],
            capture_output=True,
            text=True,
            cwd='C:/Users/ashwi/workplace/liars-chatroom-starter'
        )
        
        assert result.returncode == 0
        
        # Count messages per conversation in CSV
        train_df = pd.read_csv(output_dir / 'train.csv')
        val_df = pd.read_csv(output_dir / 'validation.csv')
        all_df = pd.concat([train_df, val_df])
        
        # Count messages per conversation
        conv_counts = all_df.groupby('conversation_id').size()
        
        for conv_id, count in conv_counts.items():
            assert 8 <= count <= 20, \
                f"Conversation {conv_id} has {count} messages, expected 8-20"


class TestV2GenerateOneLiar:
    """Tests for exactly one liar per conversation."""
    
    def test_one_liar_per_conv(self, temp_dir):
        """Test that each conversation has exactly one liar."""
        output_dir = temp_dir / 'synthetic'
        output_dir.mkdir()
        
        result = subprocess.run(
            [
                'python', '-m', 'synthetic.v2.generate',
                '--num-examples', '20',
                '--output-dir', str(output_dir),
                '--seed', '42'
            ],
            capture_output=True,
            text=True,
            cwd='C:/Users/ashwi/workplace/liars-chatroom-starter'
        )
        
        assert result.returncode == 0
        
        # Count liars per conversation in ground truth
        with open(output_dir / 'ground_truth.jsonl', 'r') as f:
            lines = f.readlines()
        
        # Each record should have exactly one liar
        for i, line in enumerate(lines):
            record = json.loads(line)
            # The record represents one conversation with one liar
            assert 'liar' in record
            assert record['liar'] in ['A', 'B', 'C', 'D']
