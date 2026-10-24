import sys
from exp.data.split import generate_all_splits


def main():
    profile = "smoke" if "--smoke" in sys.argv else "full"
    force = "--force" in sys.argv or "-f" in sys.argv
    print(f"Generating scenarios and log bank with profile='{profile}'...")
    splits = generate_all_splits(profile=profile, force=force)
    for k, v in splits.items():
        print(f"  {k}: {len(v)} scenarios")


if __name__ == "__main__":
    main()
