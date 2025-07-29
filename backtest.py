#!/usr/bin/env python3
"""
Bitcoin Fee Estimator Backtest Tool

A minimal tool to reproduce fee estimator benchmark results using exported CSV data.
This tool compares fee estimation strategies from different providers by simulating 
their performance against historical block data.

Usage:
    python backtest.py [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD] [--providers PROVIDER1,PROVIDER2]

Example:
    python backtest.py --start-date 2024-04-01 --end-date 2024-04-07 --providers AUGUR,WHAT_THE_FEE
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from io import StringIO
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from colorama import Fore, Style
from tabulate import tabulate
from tqdm import tqdm


class FeeEstimatorBacktest:
    """Main backtesting class for Bitcoin fee estimators."""
    
    AVG_BLOCK_DURATION_MINS = 10
    AVG_BLOCK_COUNT_IN_DAY = 144
    MIN_SATS_PER_KB = 1000

    def __init__(self, block_fees_file: str, provider_estimates_file: str):
        """
        Initialize the backtesting tool.
        
        Args:
            block_fees_file: Path to CSV file with historical block fee data
            provider_estimates_file: Path to CSV file with fee estimates from providers
        """
        self.block_fees_file = block_fees_file
        self.provider_estimates_file = provider_estimates_file
        
        # Fee rate percentiles used for confirmation thresholds
        self.UNDERESTIMATE_PERCENTILE = "p5_fee_rate"    # 5th percentile
        self.OVERESTIMATE_PERCENTILE = "p75_fee_rate"    # 75th percentile
        
    def load_block_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load and preprocess historical block fee data.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with processed block data
        """
        print("Loading block fee data...")
        
        # Read block data in chunks to handle large files
        chunk_list = []
        chunk_size = 50000
        
        for chunk in tqdm(pd.read_csv(self.block_fees_file, chunksize=chunk_size), 
                         desc="Reading block data"):
            # Convert timestamp and filter by date range
            chunk['_OCCURRED_AT'] = pd.to_datetime(chunk['_OCCURRED_AT'])
            
            # Filter by date range
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date) + timedelta(days=1)  # Include end date
            
            chunk = chunk[
                (chunk['_OCCURRED_AT'] >= start_dt) & 
                (chunk['_OCCURRED_AT'] < end_dt)
            ]
            
            if not chunk.empty:
                chunk_list.append(chunk)
        
        if not chunk_list:
            raise ValueError(f"No block data found in date range {start_date} to {end_date}")
            
        block_fees = pd.concat(chunk_list, ignore_index=True)
        
        # Sort and remove duplicates
        block_fees = block_fees.sort_values(by='_OCCURRED_AT', ascending=True)
        block_fees = block_fees.drop_duplicates(subset=['HEIGHT'], keep='first')
        
        print("Processing block fee percentiles...")
        
        # Extract percentile fee rates from JSON data
        def extract_percentile(percentile_json: str, target_percentile: int) -> float:
            """Extract specific percentile from JSON data."""
            try:
                percentiles = json.loads(percentile_json)
                for p in percentiles:
                    if p['percentile'] == target_percentile:
                        return float(p['fee_rate'])
                return 0.0
            except (json.JSONDecodeError, KeyError, TypeError):
                return 0.0
        
        # Extract relevant percentiles
        tqdm.pandas(desc="Extracting percentiles")
        block_fees['p5_fee_rate'] = block_fees['PERCENTILE_FEE_RATES'].progress_apply(
            lambda x: extract_percentile(x, 5)
        )
        block_fees['p10_fee_rate'] = block_fees['PERCENTILE_FEE_RATES'].progress_apply(
            lambda x: extract_percentile(x, 10)
        )
        block_fees['p25_fee_rate'] = block_fees['PERCENTILE_FEE_RATES'].progress_apply(
            lambda x: extract_percentile(x, 25)
        )
        block_fees['p75_fee_rate'] = block_fees['PERCENTILE_FEE_RATES'].progress_apply(
            lambda x: extract_percentile(x, 75)
        )
        block_fees['median_fee_rate'] = block_fees['PERCENTILE_FEE_RATES'].progress_apply(
            lambda x: extract_percentile(x, 50)
        )
        
        print(f"Loaded {len(block_fees)} blocks from {start_date} to {end_date}")
        return block_fees
    
    def load_provider_data(self, start_date: str, end_date: str, providers: List[str]) -> pd.DataFrame:
        """
        Load and preprocess fee estimates from providers.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            providers: List of provider names to include
            
        Returns:
            DataFrame with processed fee estimates
        """
        print("Loading provider fee estimate data...")
        
        chunk_list = []
        chunk_size = 100000
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date) + timedelta(days=1)
        
        for chunk in tqdm(pd.read_csv(self.provider_estimates_file, chunksize=chunk_size),
                         desc="Reading provider data"):
            # Convert timestamp and filter
            chunk['ESTIMATED_AT'] = pd.to_datetime(chunk['ESTIMATED_AT'])
            
            # Filter by date range and providers
            chunk = chunk[
                (chunk['ESTIMATED_AT'] >= start_dt) & 
                (chunk['ESTIMATED_AT'] < end_dt) &
                (chunk['PROVIDER'].isin(providers))
            ]
            
            if not chunk.empty:
                chunk_list.append(chunk)
        
        if not chunk_list:
            raise ValueError(f"No provider data found for {providers} in date range {start_date} to {end_date}")
            
        fee_estimates = pd.concat(chunk_list, ignore_index=True)
        
        print("Processing provider fee estimates...")
        
        # Process fee estimates based on provider type
        processed_estimates = []
        
        for provider in tqdm(providers, desc="Processing providers"):
            provider_data = fee_estimates[fee_estimates['PROVIDER'] == provider].copy()
            
            if provider == 'BITCOINER_LIVE':
                # Process time-based targets
                processed_estimates.extend(self._process_bitcoiner_live(provider_data))
            else:
                # Process block-based targets
                processed_estimates.extend(self._process_block_targets(provider_data))
        
        result_df = pd.DataFrame(processed_estimates)
        print(f"Processed {len(result_df)} fee estimates from {len(providers)} providers")
        return result_df
    
    def _process_block_targets(self, data: pd.DataFrame) -> List[Dict]:
        """Process providers that use block-based targets."""
        results = []
        
        for _, row in data.iterrows():
            try:
                # Parse block target JSON
                block_targets = json.loads(row['BLOCK_TARGET'])
                
                for target_info in block_targets:
                    block_target = target_info['block_target']
                    
                    # Normalize block targets
                    if block_target in [2, 3]:
                        block_target = 1
                    elif block_target == 10:
                        block_target = 12
                    
                    # Only include standard targets
                    if block_target in [1, 12, 144]:
                        results.append({
                            'estimated_at': row['ESTIMATED_AT'],
                            'provider': row['PROVIDER'],
                            'block_target': block_target,
                            'fee_rate_sats_per_kb': target_info['fee_rate']
                        })
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
                
        return results
    
    def _process_bitcoiner_live(self, data: pd.DataFrame) -> List[Dict]:
        """Process Bitcoiner.Live time-based targets."""
        results = []
        
        # Map time targets to block targets (10 min average block time)
        time_to_block_mapping = {
            30: 1,    # 30 minutes → 1 block
            120: 12,  # 120 minutes → 12 blocks  
            1440: 144 # 1440 minutes → 144 blocks
        }
        
        for _, row in data.iterrows():
            try:
                # Parse time target JSON
                time_targets = json.loads(row['TIME_TARGET'])
                
                for target_info in time_targets:
                    minutes = target_info['minutes_target']
                    
                    if minutes in time_to_block_mapping:
                        results.append({
                            'estimated_at': row['ESTIMATED_AT'],
                            'provider': row['PROVIDER'],
                            'block_target': time_to_block_mapping[minutes],
                            'fee_rate_sats_per_kb': target_info['fee_rate']
                        })
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
                
        return results
    
    def run_backtest(self, start_date: str, end_date: str, providers: List[str]) -> Dict:
        """
        Run the complete backtest simulation.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format  
            providers: List of provider names to test
            
        Returns:
            Dictionary with performance metrics for each provider
        """
        # Load data
        block_fees = self.load_block_data(start_date, end_date)
        fee_estimates = self.load_provider_data(start_date, end_date, providers)
        
        # Initialize metrics tracking
        metrics = {
            provider: {
                str(target): {
                    'total_estimates': 0,
                    'missed_estimates': 0,
                    'overestimates': [],
                    'underestimates': []
                }
                for target in [1, 12, 144]
            }
            for provider in providers
        }
        
        # Convert to lists for processing
        block_fees_list = block_fees.sort_values('_OCCURRED_AT').to_dict('records')
        fee_estimates = fee_estimates.sort_values('estimated_at')
        
        print("Running backtest simulation...")
        
        # Simulate fee estimates against historical blocks
        for i, current_block in tqdm(enumerate(block_fees_list), 
                                   total=len(block_fees_list),
                                   desc="Simulating estimates"):
            
            # Find relevant fee estimates for this block
            last_block_timestamp = block_fees_list[i - 1]['_OCCURRED_AT'] if i > 0 else None
            block_timestamp = current_block['_OCCURRED_AT']
            
            # Strategy: As a default, we'll use estimates made shortly before the current block
            matching_estimates = fee_estimates[
                (fee_estimates['estimated_at'] < block_timestamp - pd.Timedelta(minutes=1)) &
                (fee_estimates['estimated_at'] > block_timestamp - pd.Timedelta(minutes=2))
            ]

            # For any current blocks (except the last one) where the duration between consecutive blocks
            # is at least five minutes, then we'll prefer to override the default estimates
            # with those made halfway through an average mining interval
            if last_block_timestamp:
                target_timestamp = last_block_timestamp + pd.Timedelta(minutes=self.AVG_BLOCK_DURATION_MINS / 2)
                if target_timestamp < block_timestamp:
                    matching_estimates = fee_estimates[
                        (fee_estimates['estimated_at'] >= target_timestamp) &
                        (fee_estimates['estimated_at'] < target_timestamp + pd.Timedelta(minutes=1))
                    ]
            
            if matching_estimates.empty:
                continue
            
            # Process estimates by provider and target
            estimates_by_target = matching_estimates.groupby(['provider', 'block_target'])
            
            for (provider, block_target), target_estimates in estimates_by_target:
                if str(block_target) not in metrics[provider]:
                    continue
                
                target_blocks = int(block_target)
                
                # Check if we have enough future blocks
                if i + target_blocks > len(block_fees_list):
                    continue
                
                # Select blocks in the target confirmation window
                blocks_in_window = block_fees_list[i:i + target_blocks]

                # Extract confirmation p5 threshold fees
                # We'll replace 0's with the median fee rate to avoid edge cases
                # such as anchor outputs that have zero fee rates, which can cause p5 to be zero
                fees = [
                    block[self.UNDERESTIMATE_PERCENTILE] if block[self.UNDERESTIMATE_PERCENTILE] != 0 else block['median_fee_rate']
                    for block in blocks_in_window
                ]

                # Determine the minimum required fee (at least 1 sats/vB) for confirmation
                # We'll consider any fees below this threshold as too low to get confirmed
                min_required_fee = max(min(fees), self.MIN_SATS_PER_KB) if fees else self.MIN_SATS_PER_KB

                # Choose the block with the lowest p5 threshold and find the p75 threshold of this block.
                # We'll consider any fees above the p75 threshold as too high and overpaying
                min_fee_block = blocks_in_window[fees.index(min(fees))] if fees else blocks_in_window[0]
                min_overestimate_threshold = min_fee_block[self.OVERESTIMATE_PERCENTILE]
                
                # Use the most recent estimate
                latest_estimate = target_estimates.iloc[-1]
                estimated_fee = int(latest_estimate['fee_rate_sats_per_kb'])
                
                # Update metrics
                provider_metrics = metrics[provider][str(block_target)]
                provider_metrics['total_estimates'] += 1
                
                fee_difference = estimated_fee - min_required_fee
                is_underestimate = fee_difference < 0
                
                # Initialize difference tracking
                absolute_difference = 0
                percent_difference = 0
                
                # Calculate performance metrics
                if is_underestimate:
                    # Find actual confirmation time (consider a one day grace period past the target block as an upper bound)
                    blocks_to_confirm = target_blocks
                    for idx, block in enumerate(block_fees_list[i + target_blocks:i + target_blocks + self.AVG_BLOCK_COUNT_IN_DAY]):
                        block_fee = block[self.UNDERESTIMATE_PERCENTILE] or block['median_fee_rate']
                        blocks_to_confirm = target_blocks + idx + 1
                        
                        if estimated_fee >= block_fee:
                            break
                    
                    provider_metrics['missed_estimates'] += 1
                    
                    absolute_difference = min_required_fee - estimated_fee
                    percent_difference = absolute_difference / min_required_fee * 100
                    
                    provider_metrics['underestimates'].append({
                        'absolute': absolute_difference,
                        'percent': percent_difference,
                        'blocks_to_confirm': blocks_to_confirm
                    })
                else:
                    # Find actual confirmation time within window
                    blocks_to_confirm = 1
                    for idx, block in enumerate(blocks_in_window):
                        block_fee = block[self.UNDERESTIMATE_PERCENTILE] or block['median_fee_rate']
                        if estimated_fee >= block_fee:
                            blocks_to_confirm = idx + 1
                            break
                    
                    # Overestimation occurs when estimated_fee > min_overestimate_threshold
                    absolute_difference = max(estimated_fee - min_overestimate_threshold, 0)
                    percent_difference = (absolute_difference / min_overestimate_threshold * 100) if absolute_difference > 0 else 0

                    provider_metrics['overestimates'].append({
                        'absolute': absolute_difference,
                        'percent': percent_difference,
                        'blocks_to_confirm': blocks_to_confirm
                    })
                
                # Track total differences for aggregate statistics
                provider_metrics['total_difference'] = provider_metrics.get('total_difference', 0) + absolute_difference
                provider_metrics['total_difference_percent'] = provider_metrics.get('total_difference_percent', 0) + percent_difference
        
        return metrics
    
    def format_results(self, metrics: Dict) -> None:
        """
        Format and print results using the same style as the internal tool.
        
        Args:
            metrics: Performance metrics dictionary
        """
        def format_percentage(value: float, thresholds: Tuple[float, float] = (10.0, 20.0)) -> str:
            """Format percentage with color coding based on thresholds."""
            if value < thresholds[0]:
                return f"{Fore.GREEN}{value:.1f}%{Style.RESET_ALL}"
            elif value > thresholds[1]:
                return f"{Fore.RED}{value:.1f}%{Style.RESET_ALL}"
            return f"{Fore.YELLOW}{value:.1f}%{Style.RESET_ALL}"
        
        def format_fee_stat(pct: float, abs_value: float) -> str:
            """Format fee statistic with both percentage and absolute values."""
            return f"{format_percentage(pct)} ({abs_value:,.0f} sats/kb)"
        
        def calc_stats(estimates: List[Dict]) -> Tuple[float, float, float, float]:
            """Calculate statistics from estimate list."""
            if not estimates:
                return 0, 0, 0, 0
            
            abs_values = [e['absolute'] for e in estimates]
            pct_values = [e['percent'] for e in estimates]
            
            return (
                sum(abs_values) / len(abs_values),
                sum(pct_values) / len(pct_values),
                np.median(abs_values),
                np.median(pct_values)
            )
        
        print(f"\n{Fore.CYAN}═══════════════════ METRICS SUMMARY ═══════════════════{Style.RESET_ALL}")
        print(f"\nUsing {self.UNDERESTIMATE_PERCENTILE} percentile fee rate as confirmation threshold\n")
        print(f"Using {self.OVERESTIMATE_PERCENTILE} percentile fee rate as overestimate threshold\n")
        
        headers = ["Block Target", "Total Estimates", "Miss Rate", "Total Difference", "Avg Over-Est", "Med Over-Est",
                   "Avg Under-Est", "Med Under-Est", "Avg Blocks", "Median Blocks"]
        
        # Calculate and display metrics for each provider separately
        for provider, provider_metrics in metrics.items():
            print(f"\n{Fore.CYAN}═══════════════════ {provider} ═══════════════════{Style.RESET_ALL}")
            
            table_data = []
            
            for block_target, target_metrics in sorted(provider_metrics.items(), key=lambda x: int(x[0])):
                m = target_metrics
                # Skip if no estimates were collected for this target
                if m['total_estimates'] == 0:
                    continue
                
                # Calculate miss rate (% of estimates that wouldn't have confirmed in target time)
                miss_rate = m['missed_estimates'] / m['total_estimates']
                
                over_abs_avg, over_pct_avg, over_abs_med, over_pct_med = calc_stats(m['overestimates'])
                under_abs_avg, under_pct_avg, under_abs_med, under_pct_med = calc_stats(m['underestimates'])
                
                # Collect confirmation blocks data from all estimates
                blocks_to_confirm = [e['blocks_to_confirm'] for e in m['overestimates'] + m['underestimates']]
                
                # Calculate average and median confirmation blocks
                avg_blocks = sum(blocks_to_confirm) / len(blocks_to_confirm) if blocks_to_confirm else 0
                median_blocks = np.median(blocks_to_confirm) if blocks_to_confirm else 0
                
                # Calculate average total difference
                avg_total_diff_abs = m.get('total_difference', 0) / m['total_estimates'] if m['total_estimates'] > 0 else 0
                avg_total_diff_pct = m.get('total_difference_percent', 0) / m['total_estimates'] if m['total_estimates'] > 0 else 0
                
                # Create a row for this block target with all calculated statistics
                row = [
                    f"{Fore.CYAN}{block_target}{Style.RESET_ALL}",  # Block target
                    m['total_estimates'],  # Total estimates analyzed
                    format_percentage(miss_rate * 100, (10, 20)),  # Miss rate with thresholds
                    format_fee_stat(avg_total_diff_pct, avg_total_diff_abs),  # Total difference with % and abs
                    format_fee_stat(over_pct_avg, over_abs_avg),  # Average overestimation
                    format_fee_stat(over_pct_med, over_abs_med),  # Median overestimation
                    format_fee_stat(under_pct_avg, under_abs_avg),  # Average underestimation
                    format_fee_stat(under_pct_med, under_abs_med),  # Median underestimation
                    f"{avg_blocks:.1f}",  # Average blocks to confirm
                    f"{median_blocks:.1f}",  # Median blocks to confirm
                ]
                table_data.append(row)
            
            # Print a nicely formatted table of results for this provider
            if table_data:
                print(tabulate(table_data, headers=headers, tablefmt="grid"))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Bitcoin Fee Estimator Backtest Tool')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--providers', default='AUGUR,WHAT_THE_FEE,BITCOINER_LIVE,BLOCKSTREAM,MEMPOOL_SPACE', 
                       help='Comma-separated list of providers')
    parser.add_argument('--block-data', default='data/block_fees.csv',
                       help='Path to block fees CSV file')
    parser.add_argument('--provider-data', default='data/fee_provider_estimates.csv',
                       help='Path to provider estimates CSV file')
    
    args = parser.parse_args()
    
    providers = [p.strip() for p in args.providers.split(',')]
    
    print(f"Starting backtest from {args.start_date} to {args.end_date}")
    print(f"Testing providers: {', '.join(providers)}")
    print()
    
    try:
        backtest = FeeEstimatorBacktest(args.block_data, args.provider_data)
        metrics = backtest.run_backtest(args.start_date, args.end_date, providers)
        
        backtest.format_results(metrics)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()