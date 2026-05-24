import time
import subprocess
import re
import argparse

def get_gpu_memory(pid):
    """Get the GPU memory usage of a specific process."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            print(f"Error running nvidia-smi: {result.stderr}")
            return None

        memory_usage = 0
        for line in result.stdout.strip().split("\n"):
            match = re.match(r"(\d+),\s*(\d+)", line)
            if match:
                current_pid, mem = map(int, match.groups())
                if current_pid == pid:
                    memory_usage = max(memory_usage, mem)
        return memory_usage
    except Exception as e:
        print(f"Failed to query GPU memory: {e}")
        return None

def monitor_gpu_memory(pid, interval=1):
    """Monitor the GPU memory usage of a specific process."""
    peak_memory = 0
    print(f"Monitoring GPU memory usage for PID {pid}...")
    try:
        while True:
            memory = get_gpu_memory(pid)
            if memory is not None:
                peak_memory = max(peak_memory, memory)
                print(f"Current: {memory} MB, Peak: {peak_memory} MB")
            else:
                print("Process not found or no GPU memory usage.")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
        print(f"Final Peak Memory Usage: {peak_memory} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor GPU memory usage of a specific process.")
    parser.add_argument("pid", type=int, help="Process ID to monitor.")
    parser.add_argument("--interval", type=float, default=1, help="Interval in seconds between checks (default: 1s).")
    args = parser.parse_args()

    monitor_gpu_memory(args.pid, args.interval)