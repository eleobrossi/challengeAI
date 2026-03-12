import json
import csv
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import os

# Path to the data directory
DATA_DIR = Path("The Truman Show_train/public")

class UnifiedDatasetBuilder:
    """Build a unified dataset from JSON and CSV files."""
    
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.users = []
        self.locations = []
        self.transactions = []
        self.mails = []
        self.sms = []
        self.unified_data = {}
    
    def load_json_file(self, filename: str) -> List[Dict]:
        """Load a JSON file."""
        filepath = self.data_dir / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def load_csv_file(self, filename: str) -> List[Dict]:
        """Load a CSV file as list of dictionaries."""
        filepath = self.data_dir / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                return list(reader)
        return []
    
    def load_all_data(self):
        """Load all data files."""
        print("Loading data files...")
        self.users = self.load_json_file("users.json")
        self.locations = self.load_json_file("locations.json")
        self.mails = self.load_json_file("mails.json")
        self.sms = self.load_json_file("sms.json")
        self.transactions = self.load_csv_file("transactions.csv")
        
        print(f"✓ Loaded {len(self.users)} users")
        print(f"✓ Loaded {len(self.locations)} location records")
        print(f"✓ Loaded {len(self.transactions)} transactions")
        print(f"✓ Loaded {len(self.mails)} emails")
        print(f"✓ Loaded {len(self.sms)} SMS messages")
    
    def extract_user_id_from_biodata(self, user: Dict) -> str:
        """Extract a user ID from user data."""
        # Create a unique ID from name and residence
        first = user.get('first_name', '').upper()[:3]
        last = user.get('last_name', '').upper()[:4]
        city = user.get('residence', {}).get('city', 'UNK')[:3].upper()
        birth = str(user.get('birth_year', 0000))[-2:]
        return f"{first}-{last}-{birth}-{city}"
    
    def match_biotags_to_users(self) -> Dict[str, Dict]:
        """Create a mapping of biotags/IDs to user information."""
        biotag_to_user = {}
        
        # Extract unique biotags from locations
        unique_biotags = set(loc.get('biotag') for loc in self.locations)
        
        # Try to match biotags to users based on city
        for user_idx, user in enumerate(self.users):
            user_id = self.extract_user_id_from_biodata(user)
            
            # Also add the full name as potential identifier
            email_prefix = f"{user.get('first_name', '').lower()}.{user.get('last_name', '').lower()}"
            iban = user.get('iban', '')
            
            # Store user with generated ID
            biotag_to_user[user_id] = user
            biotag_to_user[email_prefix] = user
            biotag_to_user[iban] = user
            
            # Add first name + last initial + city pattern
            pattern = f"{user.get('first_name', 'X').upper()[:1]}{user.get('last_name', 'X').upper()[:4]}-{user.get('residence', {}).get('city', 'X')}"
            biotag_to_user[pattern] = user
        
        return biotag_to_user
    
    def build_unified_dataset(self) -> Dict[str, Any]:
        """Build the unified dataset."""
        print("\nBuilding unified dataset...")
        
        biotag_map = self.match_biotags_to_users()
        
        # Create a comprehensive dataset structure
        unified = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "source": "The Truman Show_train/public",
                "data_types": ["users", "locations", "transactions", "communications"],
                "total_records": len(self.users)
            },
            "users": [],
            "user_profiles": [],
            "communication_records": [],
            "transaction_records": [],
            "location_history": []
        }
        
        # Build user profiles with all related data
        for user_idx, user in enumerate(self.users):
            user_id = self.extract_user_id_from_biodata(user)
            
            # Find related communications
            related_mails = []
            related_sms = []
            related_transactions_out = []
            related_transactions_in = []
            related_locations = []
            
            # Get user's IBAN and name for matching
            user_iban = user.get('iban', '')
            user_email = f"{user.get('first_name', '').lower()}.{user.get('last_name', '').lower()}"
            
            # Match communications
            for mail in self.mails:
                mail_text = mail.get('mail', '')
                if user_email in mail_text.lower() or f"{user.get('first_name', '').lower()}" in mail_text.lower():
                    related_mails.append(mail_text[:500])  # First 500 chars
            
            for sms in self.sms:
                sms_text = sms.get('sms', '')
                if user.get('first_name', '').lower() in sms_text.lower():
                    related_sms.append(sms_text)
            
            # Match transactions
            for transaction in self.transactions:
                if transaction.get('sender_iban') == user_iban:
                    related_transactions_out.append(transaction)
                if transaction.get('recipient_iban') == user_iban:
                    related_transactions_in.append(transaction)
            
            # Match locations
            for location in self.locations:
                if location.get('city') == user.get('residence', {}).get('city'):
                    related_locations.append(location)
            
            # Build user profile
            user_profile = {
                "user_id": user_id,
                "iban": user_iban,
                "name": f"{user.get('first_name')} {user.get('last_name')}",
                "first_name": user.get('first_name'),
                "last_name": user.get('last_name'),
                "birth_year": user.get('birth_year'),
                "age_2026": 2026 - user.get('birth_year', 2000),
                "job": user.get('job'),
                "salary": user.get('salary'),
                "residence": user.get('residence'),
                "description": user.get('description'),
                "email_pattern": user_email,
                "statistics": {
                    "emails_received": len(related_mails),
                    "sms_received": len(related_sms),
                    "transactions_sent": len(related_transactions_out),
                    "transactions_received": len(related_transactions_in),
                    "location_records": len(related_locations),
                    "total_sent": sum(float(t.get('amount', 0)) for t in related_transactions_out),
                    "total_received": sum(float(t.get('amount', 0)) for t in related_transactions_in)
                },
                "communications": {
                    "emails": related_mails[:3],  # First 3
                    "sms_count": len(related_sms),
                    "sms_samples": [s[:100] for s in related_sms[:2]]
                },
                "transactions": {
                    "outgoing_count": len(related_transactions_out),
                    "incoming_count": len(related_transactions_in),
                    "recent_outgoing": [
                        {
                            "id": t.get('transaction_id'),
                            "amount": t.get('amount'),
                            "recipient_iban": t.get('recipient_iban'),
                            "description": t.get('description'),
                            "date": t.get('timestamp')
                        }
                        for t in related_transactions_out[:3]
                    ],
                    "recent_incoming": [
                        {
                            "id": t.get('transaction_id'),
                            "amount": t.get('amount'),
                            "sender_iban": t.get('sender_iban'),
                            "description": t.get('description'),
                            "date": t.get('timestamp')
                        }
                        for t in related_transactions_in[:3]
                    ]
                },
                "locations": {
                    "residence": user.get('residence'),
                    "tracking_records": len(related_locations),
                    "sample_locations": [
                        {
                            "timestamp": l.get('timestamp'),
                            "lat": l.get('lat'),
                            "lng": l.get('lng'),
                            "city": l.get('city')
                        }
                        for l in related_locations[:3]
                    ] if related_locations else []
                }
            }
            
            unified["users"].append(user)
            unified["user_profiles"].append(user_profile)
        
        # Add raw records for detailed analysis
        unified["transaction_records"] = self.transactions
        unified["location_history"] = self.locations
        
        # Add a summary
        unified["summary"] = {
            "total_users": len(self.users),
            "total_transactions": len(self.transactions),
            "total_location_records": len(self.locations),
            "total_communications": len(self.mails) + len(self.sms),
            "cities": list(set(u.get('residence', {}).get('city') for u in self.users if u.get('residence'))),
            "job_distribution": self._get_job_distribution(),
            "date_range": self._get_date_range()
        }
        
        return unified
    
    def _get_job_distribution(self) -> Dict[str, int]:
        """Get distribution of jobs."""
        jobs = {}
        for user in self.users:
            job = user.get('job', 'Unknown')
            jobs[job] = jobs.get(job, 0) + 1
        return jobs
    
    def _get_date_range(self) -> Dict[str, str]:
        """Get date range of transactions."""
        if not self.transactions:
            return {}
        
        dates = [t.get('timestamp', '') for t in self.transactions if t.get('timestamp')]
        dates.sort()
        
        return {
            "earliest": dates[0] if dates else None,
            "latest": dates[-1] if dates else None
        }
    
    def save_datasets(self, output_dir: str = "."):
        """Save unified dataset in multiple formats."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("\nSaving unified dataset...")
        
        # Save as JSON (complete)
        json_file = output_path / "unified_dataset_complete.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.unified_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved complete dataset to {json_file}")
        
        # Save user profiles as JSON (for easy analysis)
        profiles_file = output_path / "user_profiles.json"
        with open(profiles_file, 'w', encoding='utf-8') as f:
            json.dump(self.unified_data["user_profiles"], f, indent=2, ensure_ascii=False)
        print(f"✓ Saved user profiles to {profiles_file}")
        
        # Save user profiles as CSV (for spreadsheet analysis)
        profiles_csv = output_path / "user_profiles.csv"
        if self.unified_data["user_profiles"]:
            df = pd.DataFrame(self.unified_data["user_profiles"])
            # Flatten nested dicts for CSV
            df_flat = pd.json_normalize(self.unified_data["user_profiles"])
            df_flat.to_csv(profiles_csv, index=False, encoding='utf-8')
            print(f"✓ Saved user profiles to {profiles_csv}")
        
        # Save summary report
        summary_file = output_path / "dataset_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(self.unified_data["summary"], f, indent=2)
        print(f"✓ Saved summary to {summary_file}")
        
        # Save transactions as CSV
        transactions_csv = output_path / "transactions_clean.csv"
        if self.transactions:
            df_trans = pd.DataFrame(self.transactions)
            df_trans.to_csv(transactions_csv, index=False, encoding='utf-8')
            print(f"✓ Saved transactions to {transactions_csv}")
        
        # Save locations as CSV
        locations_csv = output_path / "locations_clean.csv"
        if self.locations:
            df_locs = pd.DataFrame(self.locations)
            df_locs.to_csv(locations_csv, index=False, encoding='utf-8')
            print(f"✓ Saved locations to {locations_csv}")
        
        print("\n✅ All datasets saved successfully!")
    
    def generate_analysis_report(self) -> str:
        """Generate a summary analysis report."""
        report = []
        report.append("=" * 80)
        report.append("UNIFIED DATASET ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        summary = self.unified_data["summary"]
        report.append(f"Dataset Created: {self.unified_data['metadata']['created']}")
        report.append(f"Total Users: {summary['total_users']}")
        report.append(f"Total Transactions: {summary['total_transactions']}")
        report.append(f"Total Location Records: {summary['total_location_records']}")
        report.append(f"Total Communications: {summary['total_communications']}")
        report.append("")
        
        report.append("Cities:")
        for city in summary.get('cities', []):
            report.append(f"  - {city}")
        report.append("")
        
        report.append("Job Distribution:")
        for job, count in summary.get('job_distribution', {}).items():
            report.append(f"  - {job}: {count}")
        report.append("")
        
        report.append("Date Range:")
        date_range = summary.get('date_range', {})
        report.append(f"  - Earliest: {date_range.get('earliest')}")
        report.append(f"  - Latest: {date_range.get('latest')}")
        report.append("")
        
        report.append("Top Users by Activity:")
        profiles = sorted(
            self.unified_data["user_profiles"],
            key=lambda x: sum([
                x['statistics']['emails_received'],
                x['statistics']['sms_received'],
                x['statistics']['transactions_sent']
            ]),
            reverse=True
        )[:5]
        
        for profile in profiles:
            total_activity = (
                profile['statistics']['emails_received'] +
                profile['statistics']['sms_received'] +
                profile['statistics']['transactions_sent']
            )
            report.append(f"  - {profile['name']}: {total_activity} activities")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """Main execution function."""
    print("🚀 Building Unified Dataset from The Truman Show_train\n")
    
    builder = UnifiedDatasetBuilder()
    builder.load_all_data()
    builder.unified_data = builder.build_unified_dataset()
    
    # Save all formats
    builder.save_datasets()
    
    # Print analysis report
    report = builder.generate_analysis_report()
    print("\n" + report)
    
    # Save report to file
    with open("dataset_analysis_report.txt", "w", encoding='utf-8') as f:
        f.write(report)
    print("\n✓ Analysis report saved to dataset_analysis_report.txt")


if __name__ == "__main__":
    main()
