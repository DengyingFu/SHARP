import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(
        description="Unified entry point for running evaluations.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # 必须的参数
    parser.add_argument(
        "benchmark_folder",
        type=str,
        help="Benchmark folder name under data/, e.g. Ambiguity-200"
    )
    
    parser.add_argument(
        "--eval",
        type=str,
        required=True,
        choices=["end2end", "no_scenegraph", "no_vlm", "sharp"],
        help="Which evaluation pipeline to run:\n"
             "  end2end:       Run evaluate_benchmark_end2end.py\n"
             "  no_scenegraph: Run evaluate_benchmark_no_SceneGraph.py\n"
             "  no_vlm:        Run evaluate_benchmark_no_VLM.py\n"
             "  sharp:         Run evaluate_benchmark_SHARP.py"
    )

    #这允许我们接受 --runs, --eval-metric 等其他参数并将它们传递给子脚本
    args, unknown_args = parser.parse_known_args()

    # 确定脚本路径
    # 假设此脚本位于项目根目录，评估脚本在 evaluate/ 目录下
    base_dir = os.path.dirname(os.path.abspath(__file__))
    evaluate_dir = os.path.join(base_dir, "evaluate")
    
    script_map = {
        "end2end": "evaluate_benchmark_end2end.py",
        "no_scenegraph": "evaluate_benchmark_no_SceneGraph.py",
        "no_vlm": "evaluate_benchmark_no_VLM.py",
        "sharp": "evaluate_benchmark_SHARP.py"
    }
    
    script_name = script_map[args.eval]
    script_path = os.path.join(evaluate_dir, script_name)
    
    if not os.path.exists(script_path):
        print(f"Error: Target script not found at {script_path}")
        sys.exit(1)
        
    # 构建命令
    # 格式: python <script_path> <benchmark_folder> [other_args]
    cmd = [sys.executable, script_path, args.benchmark_folder] + unknown_args
    
    print("=" * 60)
    print(f"Starting Evaluation: {args.eval}")
    print(f"Script: {script_path}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    print("")
    
    try:
        # 使用 subprocess 调用，保持实时输出
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"\nEvaluation failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
        sys.exit(1)

if __name__ == "__main__":
    main()
