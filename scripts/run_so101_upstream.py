"""Separate upstream SO-101 benchmark; joint baseline or Cartesian-to-upstream control."""
import sys
from run_replica_evaluation import main


if __name__ == "__main__":
    main(["--benchmark", "so101-upstream", *sys.argv[1:]])
