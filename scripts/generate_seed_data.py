#!/usr/bin/env python3
"""
Memory System Seed Data CLI

Command-line interface for generating, validating, and managing seed data
for the memory system testing.

Usage:
    python scripts/generate_seed_data.py --generate --sessions 5 --memories 10
    python scripts/generate_seed_data.py --validate
    python scripts/generate_seed_data.py --clear
    python scripts/generate_seed_data.py --help
"""

import argparse
import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from api.memory_system.seed_data import MemorySystemSeedData
from api.models import get_db


def print_banner():
    """Print application banner"""
    print("=" * 60)
    print("🌱 MEMORY SYSTEM SEED DATA GENERATOR")
    print("=" * 60)
    print("Generate realistic test data for memory system testing")
    print()


async def generate_seed_data(args):
    """Generate seed data with specified parameters"""
    print_banner()
    
    db = next(get_db())
    try:
        seed_generator = MemorySystemSeedData(db)
        
        if args.clear_first:
            print("🧹 Clearing existing data first...")
            await seed_generator.clear_seed_data()
            print()
        
        print(f"🌱 Generating seed data:")
        print(f"  - Sessions: {args.sessions}")
        print(f"  - Memories per session: {args.memories}")
        print()
        
        results = await seed_generator.generate_seed_data(
            num_sessions=args.sessions,
            num_memories_per_session=args.memories
        )
        
        if args.validate:
            print()
            validation = await seed_generator.validate_seed_data()
            
            if validation['validation_passed']:
                print("✅ Validation passed - seed data is valid!")
            else:
                print("❌ Validation failed - please check the errors above")
                return False
        
        print()
        print("🎯 Generation Summary:")
        print(f"  ✅ Sessions created: {results['sessions_created']}")
        print(f"  ✅ Memories created: {results['memories_created']}")
        print(f"  ✅ Contexts created: {results['contexts_created']}")
        print()
        print("✅ Seed data generation complete!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating seed data: {e}")
        return False
    finally:
        db.close()


async def validate_seed_data(args):
    """Validate existing seed data"""
    print_banner()
    
    db = next(get_db())
    try:
        seed_generator = MemorySystemSeedData(db)
        
        print("🔍 Validating existing seed data...")
        print()
        
        validation = await seed_generator.validate_seed_data()
        
        print()
        print("📊 Validation Results:")
        print(f"  Sessions: {validation['sessions']['count']}")
        print(f"    - With metadata: {validation['sessions']['has_metadata']}")
        print(f"    - With tags: {validation['sessions']['has_tags']}")
        print(f"    - With duration: {validation['sessions']['duration_set']}")
        print()
        print(f"  Memories: {validation['memories']['count']}")
        print(f"    - With entities: {validation['memories']['with_entities']}")
        print(f"    - With facts: {validation['memories']['with_facts']}")
        print(f"    - Avg importance: {validation['memories']['avg_importance']:.2f}")
        print()
        print("  Memory types:")
        for mem_type, count in validation['memories']['by_type'].items():
            print(f"    - {mem_type}: {count}")
        print()
        print(f"  Contexts: {validation['contexts']['count']}")
        print()
        
        if validation['validation_passed']:
            print("✅ Validation PASSED - seed data is valid!")
            return True
        else:
            print("❌ Validation FAILED:")
            for error in validation['errors']:
                print(f"  - {error}")
            return False
            
    except Exception as e:
        print(f"❌ Error validating seed data: {e}")
        return False
    finally:
        db.close()


async def clear_seed_data(args):
    """Clear all seed data"""
    print_banner()
    
    if not args.force:
        confirm = input("⚠️ This will delete ALL seed data. Are you sure? (y/N): ")
        if confirm.lower() != 'y':
            print("❌ Operation cancelled")
            return False
    
    db = next(get_db())
    try:
        seed_generator = MemorySystemSeedData(db)
        
        print("🧹 Clearing all seed data...")
        print()
        
        results = await seed_generator.clear_seed_data()
        
        print()
        print("📊 Clearing Summary:")
        print(f"  ✅ Sessions deleted: {results['sessions_deleted']}")
        print(f"  ✅ Memories deleted: {results['memories_deleted']}")
        print(f"  ✅ Contexts deleted: {results['contexts_deleted']}")
        print()
        print("✅ Seed data cleared successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error clearing seed data: {e}")
        return False
    finally:
        db.close()


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Memory System Seed Data Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate seed data with default settings
  python scripts/generate_seed_data.py --generate
  
  # Generate with custom parameters
  python scripts/generate_seed_data.py --generate --sessions 10 --memories 5
  
  # Generate and validate
  python scripts/generate_seed_data.py --generate --validate
  
  # Clear existing data first
  python scripts/generate_seed_data.py --generate --clear-first
  
  # Just validate existing data
  python scripts/generate_seed_data.py --validate
  
  # Clear all seed data
  python scripts/generate_seed_data.py --clear --force
        """
    )
    
    # Main actions
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--generate", "-g",
        action="store_true",
        help="Generate seed data"
    )
    action_group.add_argument(
        "--validate", "-v",
        action="store_true",
        help="Validate existing seed data"
    )
    action_group.add_argument(
        "--clear", "-c",
        action="store_true",
        help="Clear all seed data"
    )
    
    # Generation options
    parser.add_argument(
        "--sessions", "-s",
        type=int,
        default=5,
        help="Number of sessions to generate (default: 5)"
    )
    parser.add_argument(
        "--memories", "-m",
        type=int,
        default=3,
        help="Number of memories per session (default: 3)"
    )
    parser.add_argument(
        "--clear-first",
        action="store_true",
        help="Clear existing data before generating new data"
    )
    
    # Validation options
    parser.add_argument(
        "--validate-after",
        action="store_true",
        dest="validate",
        help="Validate data after generation"
    )
    
    # Clearing options
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force clear without confirmation"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.sessions < 1 or args.sessions > 20:
        print("❌ Error: Number of sessions must be between 1 and 20")
        sys.exit(1)
    
    if args.memories < 1 or args.memories > 50:
        print("❌ Error: Number of memories must be between 1 and 50")
        sys.exit(1)
    
    # Run the appropriate action
    try:
        if args.generate:
            success = asyncio.run(generate_seed_data(args))
        elif args.validate:
            success = asyncio.run(validate_seed_data(args))
        elif args.clear:
            success = asyncio.run(clear_seed_data(args))
        else:
            parser.print_help()
            sys.exit(1)
        
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()