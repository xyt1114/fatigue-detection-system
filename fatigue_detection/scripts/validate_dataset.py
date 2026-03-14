from argparse import ArgumentParser
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from detection.utils.dataset_loader import DatasetLoader


def main():
    parser = ArgumentParser()
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--dataset-root", type=str, default=str(BASE_DIR / "dataset"))
    args = parser.parse_args()

    loader = DatasetLoader(dataset_root=args.dataset_root)
    report = loader.validate_dataset(min_samples_per_class=args.min_samples)
    print(f"数据集路径: {report['dataset_root']}")
    print(f"类别分布: {report['distribution']}")
    print(f"最少样本阈值: {report['min_samples_per_class']}")
    if report["is_valid"]:
        print("数据集校验通过")
    else:
        print(f"缺失类别: {report['missing_classes']}")
        print(f"样本不足类别: {report['insufficient_classes']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
