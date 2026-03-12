"""
Enrich Transactions with Location and User Data
----------------------------------------------
This script connects each transaction to:
- Sender and recipient user information
- Last known biotag location for sender and recipient
- Timestamps of those locations
- Cities where each party was located
- All other relevant data in one comprehensive row
"""

import json
import csv
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


class TransactionLocationEnricher:
    """Enrich transactions with location data and user information."""
    
    def __init__(self):
        self.transactions = []
        self.locations = []
        self.users = {}
        self.biotag_to_user = {}
        self.iban_to_biotags = {}
        self.iban_to_user = {}
        self.city_to_biotags = {}
        self.load_data()
    
    def load_data(self):
        """Load all necessary data files."""
        print("Loading data files...")
        
        # Load complete dataset
        with open('unified_dataset_complete.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.transactions = data.get('transaction_records', [])
        self.locations = data.get('location_history', [])
        raw_users = data.get('users', [])
        
        print(f"✓ Loaded {len(self.transactions)} transactions")
        print(f"✓ Loaded {len(self.locations)} locations")
        print(f"✓ Loaded {len(raw_users)} users")
        
        # Build user lookup by IBAN
        for user in raw_users:
            iban = user.get('iban', '')
            city = user.get('residence', {}).get('city', '')
            name = f"{user.get('first_name', '')} {user.get('last_name', '')}".strip()
            
            self.iban_to_user[iban] = user
            self.users[iban] = {
                'name': name,
                'job': user.get('job', 'Unknown'),
                'city': city,
                'salary': user.get('salary', 0),
                'iban': iban
            }
            
            # Map city to biotags (users in same city likely have biotags from there)
            if city not in self.city_to_biotags:
                self.city_to_biotags[city] = set()
        
        # Build biotag to locations mapping and link by city
        for loc in self.locations:
            biotag = loc.get('biotag', '')
            city = loc.get('city', '')
            
            if biotag and city:
                self.biotag_to_user[biotag] = {
                    'city': city,
                    'lat': loc.get('lat'),
                    'lng': loc.get('lng')
                }
                
                # Initialize if city not yet in dict
                if city not in self.city_to_biotags:
                    self.city_to_biotags[city] = set()
                
                self.city_to_biotags[city].add(biotag)
        
        print(f"✓ Built user lookup with {len(self.users)} users")
        print(f"✓ Identified biotags in {len(self.city_to_biotags)} cities")
    
    def get_last_location_before_timestamp(self, biotag: str, timestamp: str) -> Optional[Dict]:
        """Get the last known location for a biotag before a given timestamp."""
        matching_locations = []
        
        for loc in self.locations:
            if loc.get('biotag') == biotag:
                loc_time = loc.get('timestamp', '')
                # Find locations before or at the transaction time
                if loc_time <= timestamp:
                    matching_locations.append(loc)
        
        if matching_locations:
            # Return the most recent one
            matching_locations.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return matching_locations[0]
        
        return None
    
    def extract_biotag_or_id(self, identifier: str) -> Optional[str]:
        """Try to extract biotag from sender/recipient ID."""
        # IDs like "WHTI-LZBT-7CE-WAS-0" might be biotags
        # or like "EMP96499" might be employee IDs
        if identifier and '-' in identifier and len(identifier) > 10:
            return identifier
        return None
    
    def get_user_info_by_identifier(self, identifier: str) -> Dict:
        """Get user info from various identifier types."""
        user_info = {
            'name': 'Unknown',
            'job': 'Unknown',
            'city': 'Unknown',
            'salary': 0
        }
        
        # Try to match by IBAN (in transactions it might be stored)
        if identifier in self.users:
            return self.users[identifier]
        
        # Try partial matches
        for iban, user in self.users.items():
            if identifier.lower() in iban.lower() or iban.lower() in identifier.lower():
                return user
        
        return user_info
    
    def get_location_info(self, sender_iban: str, recipient_iban: str, timestamp: str) -> Dict:
        """Get location information for sender and recipient at transaction time."""
        location_info = {
            'sender_biotag': None,
            'sender_location_city': 'Unknown',
            'sender_location_lat': None,
            'sender_location_lng': None,
            'sender_location_timestamp': None,
            'sender_location_distance_from_tx': None,
            
            'recipient_biotag': None,
            'recipient_location_city': 'Unknown',
            'recipient_location_lat': None,
            'recipient_location_lng': None,
            'recipient_location_timestamp': None,
            'recipient_location_distance_from_tx': None,
        }
        
        # Get sender location using IBAN
        if sender_iban in self.users:
            sender_city = self.users[sender_iban].get('city', '')
            sender_biotags = self.city_to_biotags.get(sender_city, set())
            
            if sender_biotags:
                # Get last location before timestamp from any sender biotag
                last_loc = None
                best_biotag = None
                for biotag in sender_biotags:
                    loc = self.get_last_location_before_timestamp(biotag, timestamp)
                    if loc:
                        if last_loc is None or loc.get('timestamp', '') > last_loc.get('timestamp', ''):
                            last_loc = loc
                            best_biotag = biotag
                
                if last_loc and best_biotag:
                    location_info['sender_biotag'] = best_biotag
                    location_info['sender_location_city'] = last_loc.get('city', 'Unknown')
                    location_info['sender_location_lat'] = last_loc.get('lat')
                    location_info['sender_location_lng'] = last_loc.get('lng')
                    location_info['sender_location_timestamp'] = last_loc.get('timestamp')
                    
                    # Calculate time difference
                    try:
                        loc_time = datetime.fromisoformat(last_loc.get('timestamp', '').replace('Z', '+00:00'))
                        tx_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        time_diff = (tx_time - loc_time).total_seconds() / 3600
                        location_info['sender_location_distance_from_tx'] = f"{time_diff:.1f}h"
                    except:
                        pass
        
        # Get recipient location using IBAN
        if recipient_iban in self.users:
            recipient_city = self.users[recipient_iban].get('city', '')
            recipient_biotags = self.city_to_biotags.get(recipient_city, set())
            
            if recipient_biotags:
                # Get last location before timestamp from any recipient biotag
                last_loc = None
                best_biotag = None
                for biotag in recipient_biotags:
                    loc = self.get_last_location_before_timestamp(biotag, timestamp)
                    if loc:
                        if last_loc is None or loc.get('timestamp', '') > last_loc.get('timestamp', ''):
                            last_loc = loc
                            best_biotag = biotag
                
                if last_loc and best_biotag:
                    location_info['recipient_biotag'] = best_biotag
                    location_info['recipient_location_city'] = last_loc.get('city', 'Unknown')
                    location_info['recipient_location_lat'] = last_loc.get('lat')
                    location_info['recipient_location_lng'] = last_loc.get('lng')
                    location_info['recipient_location_timestamp'] = last_loc.get('timestamp')
                    
                    # Calculate time difference
                    try:
                        loc_time = datetime.fromisoformat(last_loc.get('timestamp', '').replace('Z', '+00:00'))
                        tx_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        time_diff = (tx_time - loc_time).total_seconds() / 3600
                        location_info['recipient_location_distance_from_tx'] = f"{time_diff:.1f}h"
                    except:
                        pass
        
        return location_info
    
    def enrich_transaction(self, transaction: Dict, index: int) -> Dict:
        """Enrich a single transaction with all available data."""
        sender_id = transaction.get('sender_id', '')
        recipient_id = transaction.get('recipient_id', '')
        sender_iban = transaction.get('sender_iban', '')
        recipient_iban = transaction.get('recipient_iban', '')
        timestamp = transaction.get('timestamp', '')
        amount = float(transaction.get('amount', 0))
        
        # Get user information
        sender_info = self.get_user_info_by_identifier(sender_iban) or self.get_user_info_by_identifier(sender_id)
        recipient_info = self.get_user_info_by_identifier(recipient_iban) or self.get_user_info_by_identifier(recipient_id)
        
        # Get location information
        location_info = self.get_location_info(sender_iban, recipient_iban, timestamp)
        
        # Build enriched record
        enriched = {
            # Transaction ID and Type
            'transaction_id': transaction.get('transaction_id', ''),
            'transaction_type': transaction.get('transaction_type', ''),
            'timestamp': timestamp,
            'date_only': timestamp[:10] if timestamp else '',
            'time_only': timestamp[11:19] if len(timestamp) > 11 else '',
            
            # Amount and Balance
            'amount': amount,
            'currency': 'USD' if amount < 50000 else 'USD/EUR',  # Rough heuristic
            'balance_after': transaction.get('balance_after', ''),
            'description': transaction.get('description', ''),
            
            # Sender Information
            'sender_id': sender_id,
            'sender_iban': sender_iban,
            'sender_name': sender_info.get('name', 'Unknown'),
            'sender_job': sender_info.get('job', 'Unknown'),
            'sender_residence_city': sender_info.get('city', 'Unknown'),
            'sender_salary': sender_info.get('salary', 0),
            
            # Recipient Information
            'recipient_id': recipient_id,
            'recipient_iban': recipient_iban,
            'recipient_name': recipient_info.get('name', 'Unknown'),
            'recipient_job': recipient_info.get('job', 'Unknown'),
            'recipient_residence_city': recipient_info.get('city', 'Unknown'),
            'recipient_salary': recipient_info.get('salary', 0),
            
            # Sender Last Known Location
            'sender_biotag': location_info['sender_biotag'],
            'sender_last_city': location_info['sender_location_city'],
            'sender_last_lat': location_info['sender_location_lat'],
            'sender_last_lng': location_info['sender_location_lng'],
            'sender_last_location_time': location_info['sender_location_timestamp'],
            'sender_location_hours_before_tx': location_info['sender_location_distance_from_tx'],
            
            # Recipient Last Known Location
            'recipient_biotag': location_info['recipient_biotag'],
            'recipient_last_city': location_info['recipient_location_city'],
            'recipient_last_lat': location_info['recipient_location_lat'],
            'recipient_last_lng': location_info['recipient_location_lng'],
            'recipient_last_location_time': location_info['recipient_location_timestamp'],
            'recipient_location_hours_before_tx': location_info['recipient_location_distance_from_tx'],
            
            # Original fields
            'payment_method': transaction.get('payment_method', ''),
            'transaction_location': transaction.get('location', ''),
        }
        
        return enriched
    
    def enrich_all_transactions(self) -> List[Dict]:
        """Enrich all transactions."""
        print("\nEnriching transactions...")
        enriched_transactions = []
        
        for idx, tx in enumerate(self.transactions):
            if (idx + 1) % 25 == 0:
                print(f"  Processing transaction {idx + 1}/{len(self.transactions)}...")
            
            enriched = self.enrich_transaction(tx, idx)
            enriched_transactions.append(enriched)
        
        print(f"✓ Enriched {len(enriched_transactions)} transactions")
        return enriched_transactions
    
    def save_enriched_transactions(self, enriched_transactions: List[Dict], 
                                   output_file: str = "transactions_enriched_with_locations.csv"):
        """Save enriched transactions to CSV."""
        print(f"\nSaving to {output_file}...")
        
        if not enriched_transactions:
            print("No transactions to save")
            return
        
        # Define column order for better readability
        columns = [
            # Core transaction info
            'transaction_id',
            'timestamp',
            'date_only',
            'time_only',
            'transaction_type',
            'amount',
            'currency',
            'balance_after',
            'description',
            
            # Sender info
            'sender_id',
            'sender_iban',
            'sender_name',
            'sender_job',
            'sender_residence_city',
            'sender_salary',
            'sender_biotag',
            'sender_last_city',
            'sender_last_lat',
            'sender_last_lng',
            'sender_last_location_time',
            'sender_location_hours_before_tx',
            
            # Recipient info
            'recipient_id',
            'recipient_iban',
            'recipient_name',
            'recipient_job',
            'recipient_residence_city',
            'recipient_salary',
            'recipient_biotag',
            'recipient_last_city',
            'recipient_last_lat',
            'recipient_last_lng',
            'recipient_last_location_time',
            'recipient_location_hours_before_tx',
            
            # Other
            'payment_method',
            'transaction_location',
        ]
        
        df = pd.DataFrame(enriched_transactions)
        
        # Ensure all columns exist
        for col in columns:
            if col not in df.columns:
                df[col] = ''
        
        # Reorder columns
        df = df[columns]
        
        # Save to CSV
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"✓ Saved to {output_file}")
        print(f"✓ File size: {Path(output_file).stat().st_size / 1024:.1f} KB")
        print(f"✓ Rows: {len(df)}, Columns: {len(columns)}")
        
        return df
    
    def save_enriched_transactions_json(self, enriched_transactions: List[Dict],
                                       output_file: str = "transactions_enriched_with_locations.json"):
        """Also save as JSON for reference."""
        print(f"Saving JSON version to {output_file}...")
        
        output_data = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "total_transactions": len(enriched_transactions),
                "description": "Transactions enriched with user and location information"
            },
            "transactions": enriched_transactions
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved to {output_file}")


def main():
    """Main execution."""
    print("=" * 80)
    print("TRANSACTION & LOCATION ENRICHMENT")
    print("=" * 80)
    print()
    
    enricher = TransactionLocationEnricher()
    
    # Enrich all transactions
    enriched = enricher.enrich_all_transactions()
    
    # Save to CSV
    df = enricher.save_enriched_transactions(
        enriched,
        "transactions_enriched_with_locations.csv"
    )
    
    # Save to JSON
    enricher.save_enriched_transactions_json(
        enriched,
        "transactions_enriched_with_locations.json"
    )
    
    # Display sample
    print("\n" + "=" * 80)
    print("SAMPLE DATA (First Transaction)")
    print("=" * 80)
    
    if len(df) > 0:
        sample = df.iloc[0]
        print(f"\nTransaction ID: {sample['transaction_id']}")
        print(f"Date/Time: {sample['timestamp']}")
        print(f"Amount: {sample['amount']}")
        print(f"\nSender: {sample['sender_name']} ({sample['sender_job']})")
        print(f"  Residence: {sample['sender_residence_city']}")
        print(f"  Last Known Location: {sample['sender_last_city']} at {sample['sender_last_location_time']}")
        print(f"  Coordinates: ({sample['sender_last_lat']}, {sample['sender_last_lng']})")
        print(f"  Hours before transaction: {sample['sender_location_hours_before_tx']}")
        
        print(f"\nRecipient: {sample['recipient_name']} ({sample['recipient_job']})")
        print(f"  Residence: {sample['recipient_residence_city']}")
        print(f"  Last Known Location: {sample['recipient_last_city']} at {sample['recipient_last_location_time']}")
        print(f"  Coordinates: ({sample['recipient_last_lat']}, {sample['recipient_last_lng']})")
        print(f"  Hours before transaction: {sample['recipient_location_hours_before_tx']}")
        
        print(f"\nDescription: {sample['description']}")
        print(f"Balance After: {sample['balance_after']}")
    
    print("\n" + "=" * 80)
    print("✅ ENRICHMENT COMPLETE!")
    print("=" * 80)
    print("\nFiles created:")
    print("  1. transactions_enriched_with_locations.csv")
    print("  2. transactions_enriched_with_locations.json")


if __name__ == "__main__":
    main()
