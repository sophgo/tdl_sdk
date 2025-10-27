#!/usr/bin/env python3
"""
简化版 clang-tidy 代码分析工具
===============================

专注于代码分析功能，支持交叉编译环境和项目头文件依赖分析。

用法:
    python3 run_clang_tidy_simplified.py -p build/BM1688 [选项] [files...]
    
示例:
    # 分析所有 git 跟踪的代码文件（包括头文件依赖）
    python3 run_clang_tidy_simplified.py -p build/BM1688
    
    # 分析特定文件
    python3 run_clang_tidy_simplified.py -p build/BM1688 src/main.cpp
    
    # 使用自定义检查规则分析所有 git 跟踪的文件
    python3 run_clang_tidy_simplified.py -p build/BM1688 -checks="readability-*,modernize-*"
    
    # 快速模式（不分析头文件依赖）
    python3 run_clang_tidy_simplified.py -p build/BM1688 --no-headers
"""

import argparse
import json
import multiprocessing
import os
import re
import subprocess
import sys
import threading
import queue
import time
import datetime


class ClangTidyAnalyzer:
    """clang-tidy 分析器"""

    def __init__(self, build_path, clang_tidy_binary="clang-tidy"):
        self.build_path = build_path
        self.clang_tidy_binary = clang_tidy_binary
        self.cross_compile_args = []
        self._load_cross_compile_args()

    def _load_cross_compile_args(self):
        """从 build_info.txt 加载交叉编译参数"""
        build_info_path = os.path.join(
            os.path.dirname(self.build_path), "build_info.txt"
        )
        if not os.path.exists(build_info_path):
            return

        print(f"Loading cross-compilation settings from {build_info_path}")

        with open(build_info_path, "r", encoding="utf-8") as f:
            build_info = {}
            for line in f:
                line = line.strip()
                if "=" in line:
                    key, value = line.split("=", 1)
                    build_info[key] = value.strip().strip("'\"")

        # 构造交叉编译参数
        target_triple = build_info.get("TARGET_TRIPLE", "")
        sysroot = build_info.get("SYSROOT", "")
        cxx_compiler = build_info.get("CXX", "")

        if target_triple:
            self.cross_compile_args.append(f"--target={target_triple}")

        if sysroot and os.path.isdir(sysroot):
            self.cross_compile_args.append(f"--sysroot={sysroot}")

        if cxx_compiler and os.path.isfile(cxx_compiler):
            toolchain = os.path.dirname(os.path.dirname(cxx_compiler))
            if os.path.isdir(os.path.join(toolchain, "lib", "gcc")):
                self.cross_compile_args.append(f"--gcc-toolchain={toolchain}")

                # 添加 C++ 标准库头文件路径
                self._add_cpp_include_paths(toolchain, target_triple)

        if self.cross_compile_args:
            print(f"Cross-compilation args: {self.cross_compile_args}")

    def _add_cpp_include_paths(self, toolchain, target_triple):
        """添加 C++ 标准库头文件路径"""
        gcc_lib_dir = os.path.join(toolchain, "lib", "gcc")
        if not os.path.isdir(gcc_lib_dir):
            return

        try:
            # 查找 GCC 版本
            arch_dirs = [
                d
                for d in os.listdir(gcc_lib_dir)
                if os.path.isdir(os.path.join(gcc_lib_dir, d))
            ]
            if not arch_dirs:
                return

            arch_dir = os.path.join(gcc_lib_dir, arch_dirs[0])
            version_dirs = [
                d
                for d in os.listdir(arch_dir)
                if os.path.isdir(os.path.join(arch_dir, d))
            ]
            if not version_dirs:
                return

            version = version_dirs[0]

            # 添加可能的 C++ 头文件路径
            include_paths = [
                os.path.join(toolchain, target_triple, "include", "c++", version),
                os.path.join(
                    toolchain, target_triple, "include", "c++", version, target_triple
                ),
                os.path.join(toolchain, "include", "c++", version),
                os.path.join(toolchain, "include", "c++", version, target_triple),
                os.path.join(toolchain, target_triple, "include"),
            ]

            for path in include_paths:
                if os.path.isdir(path):
                    self.cross_compile_args.append(f"-isystem{path}")

        except (OSError, IndexError):
            pass

    def _build_tidy_command(self, file_path, checks=None, quiet=True):
        """构建 clang-tidy 命令"""
        cmd = [self.clang_tidy_binary, f"-p={self.build_path}"]

        if checks:
            cmd.append(f"-checks={checks}")

        if quiet:
            cmd.append("-quiet")

        # 添加交叉编译参数
        for arg in self.cross_compile_args:
            cmd.append(f"--extra-arg-before={arg}")

        cmd.append(file_path)
        return cmd

    def analyze_file(self, file_path, checks=None, quiet=True):
        """分析单个文件"""
        cmd = self._build_tidy_command(file_path, checks, quiet)

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=os.getcwd(), check=False
            )
            return result.stdout, result.stderr, result.returncode
        except FileNotFoundError:
            print(f"Error: {self.clang_tidy_binary} not found")
            return "", "", 1
        except (OSError, subprocess.SubprocessError) as e:
            print(f"Error running clang-tidy: {e}")
            return "", "", 1


class OutputFilter:
    """输出过滤器"""

    @staticmethod
    def filter_warnings_only(output):
        """只保留警告和错误行"""
        lines = []
        for line in output.split("\n"):
            if re.search(r"\b(warning|error):", line):
                lines.append(line)
        return "\n".join(lines) if lines else ""

    @staticmethod
    def format_output(file_path, stdout, stderr, show_all=False):
        """格式化输出"""
        output = stdout + stderr
        if not output.strip():
            return ""

        if show_all:
            formatted_output = output
        else:
            formatted_output = OutputFilter.filter_warnings_only(output)

        if formatted_output.strip():
            result = f"=== {os.path.relpath(file_path)} ===\n"
            result += formatted_output + "\n"
            return result

        return ""


class FileCollector:
    """文件收集器"""

    def __init__(self, build_path):
        self.build_path = build_path
        # 项目根目录应该是构建目录的父目录的父目录
        # 因为 build_path 通常是 'build/BM1688'，我们需要回到项目根目录
        build_parent = os.path.dirname(build_path)  # 'build'
        self.project_root = (
            os.path.dirname(build_parent) if build_parent != "." else "."
        )
        if self.project_root == "":
            self.project_root = "."

    def get_cpp_files_from_database(self):
        """从编译数据库获取 C++ 文件列表"""
        db_path = os.path.join(self.build_path, "compile_commands.json")
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Compilation database not found: {db_path}")

        with open(db_path, "r", encoding="utf-8") as f:
            database = json.load(f)

        cpp_files = set()
        for entry in database:
            file_path = entry["file"]
            if not os.path.isabs(file_path):
                file_path = os.path.join(entry["directory"], file_path)
            cpp_files.add(os.path.abspath(file_path))

        return cpp_files

    def get_git_tracked_cpp_files(self):
        """获取 git 跟踪的 C++ 源文件"""
        git_files = self._get_git_tracked_files()

        cpp_files = set()

        for file_path in git_files:
            if file_path.endswith((".c", ".cpp", ".cc", ".cxx", ".c++")):
                full_path = os.path.join(self.project_root, file_path)
                if os.path.isfile(full_path):
                    cpp_files.add(os.path.abspath(full_path))

        return cpp_files

    def _get_git_tracked_files(self):
        """获取 git 跟踪的所有文件"""
        try:
            result = subprocess.run(
                ["git", "ls-files"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode == 0:
                files = [
                    line.strip()
                    for line in result.stdout.strip().split("\n")
                    if line.strip()
                ]

                return files
            else:
                print(
                    "Warning: Failed to get git tracked files, falling back to directory scan"
                )
                return self._fallback_scan_files()
        except (OSError, subprocess.SubprocessError):
            print("Warning: Git not available, falling back to directory scan")
            return self._fallback_scan_files()

    def _fallback_scan_files(self):
        """回退到目录扫描（排除常见的构建和第三方目录）"""
        exclude_dirs = {
            "build",
            "_deps",
            "node_modules",
            ".git",
            "third_party",
            "dependency",
        }
        files = []

        for root, dirs, file_list in os.walk(self.project_root):
            # 跳过排除的目录
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in file_list:
                rel_path = os.path.relpath(os.path.join(root, file), self.project_root)
                files.append(rel_path)

        return files

    def find_project_headers(self, source_files):
        """查找项目内头文件依赖"""
        print("Searching for project header dependencies...")

        project_headers = set()
        all_headers = self._collect_all_headers()
        processed_files = set()
        files_to_process = list(source_files)

        while files_to_process:
            current_file = files_to_process.pop(0)
            if current_file in processed_files:
                continue

            processed_files.add(current_file)
            includes = self._extract_includes(current_file)

            for include in includes:
                header_path = self._resolve_header_path(include, all_headers)
                if header_path and header_path not in processed_files:
                    project_headers.add(header_path)
                    files_to_process.append(header_path)

        return project_headers

    def _collect_all_headers(self):
        """收集项目内所有头文件（仅限 git 跟踪的头文件）"""
        headers = {}  # basename -> full_path 映射

        # 获取 git 跟踪的头文件
        git_files = self._get_git_tracked_files()

        for file_path in git_files:
            if file_path.endswith((".h", ".hpp", ".hxx", ".h++", ".hh")):
                full_path = os.path.join(self.project_root, file_path)
                if os.path.isfile(full_path):
                    headers[os.path.basename(file_path)] = full_path
                    headers[file_path] = full_path

        return headers

    def _extract_includes(self, file_path):
        """从文件中提取 #include 指令"""
        includes = set()
        if not os.path.isfile(file_path):
            return includes

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    match = re.match(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', line)
                    if match:
                        includes.add(match.group(1))
        except (OSError, UnicodeDecodeError):
            pass

        return includes

    def _resolve_header_path(self, include, all_headers):
        """解析头文件的完整路径"""
        # 直接匹配
        if include in all_headers:
            return all_headers[include]

        # 只匹配文件名
        basename = os.path.basename(include)
        if basename in all_headers:
            return all_headers[basename]

        return None


def run_parallel_analysis(
    analyzer, files, checks, quiet, max_workers, show_all=False, log_file=None
):
    """并行运行分析"""
    task_queue = queue.Queue()
    results = []
    lock = threading.Lock()
    progress_lock = threading.Lock()

    # 进度跟踪
    total_files = len(files)
    completed_files = [0]  # 使用列表以便在闭包中修改
    start_time = time.time()

    def worker():
        while True:
            try:
                file_path = task_queue.get(timeout=1)

                # 分析文件
                stdout, stderr, returncode = analyzer.analyze_file(
                    file_path, checks, quiet
                )

                with lock:
                    formatted = OutputFilter.format_output(
                        file_path, stdout, stderr, show_all
                    )
                    if formatted:
                        results.append((file_path, formatted, returncode))

                    # 如果有日志文件，写入结果
                    if log_file and formatted:
                        try:
                            log_file.write(formatted)
                            log_file.write("\n")
                            log_file.flush()
                        except (OSError, IOError) as e:
                            print(f"Warning: Failed to write to log file: {e}")

                # 更新进度
                with progress_lock:
                    completed_files[0] += 1
                    progress = (completed_files[0] / total_files) * 100
                    elapsed = time.time() - start_time
                    files_per_sec = completed_files[0] / elapsed if elapsed > 0 else 0
                    eta = (
                        (total_files - completed_files[0]) / files_per_sec
                        if files_per_sec > 0
                        else 0
                    )

                    # 使用 \r 覆盖当前行显示进度
                    print(
                        f"\rProgress: {completed_files[0]}/{total_files} ({progress:.1f}%) | "
                        f"Speed: {files_per_sec:.1f} files/s | ETA: {eta:.0f}s",
                        end="",
                        flush=True,
                    )

                task_queue.task_done()
            except queue.Empty:
                break
            except (OSError, subprocess.SubprocessError) as e:
                print(f"\nError processing file: {e}")
                task_queue.task_done()

    # 启动工作线程
    threads = []
    for _ in range(max_workers):
        t = threading.Thread(target=worker)
        t.daemon = True
        t.start()
        threads.append(t)

    # 添加任务
    for file_path in files:
        task_queue.put(file_path)

    # 等待完成
    task_queue.join()

    # 完成时清空进度行
    print("\r" + " " * 80 + "\r", end="")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="简化版 clang-tidy 代码分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "-p",
        "--build-path",
        required=True,
        help="编译数据库路径（包含 compile_commands.json 的目录）",
    )
    parser.add_argument(
        "-checks",
        default=None,
        help="检查规则过滤器，例如: 'readability-*,modernize-*'",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=0,
        help="并行分析的线程数，0=自动检测CPU核心数",
    )
    parser.add_argument(
        "--clang-tidy-binary", default="clang-tidy", help="clang-tidy 二进制文件路径"
    )
    parser.add_argument(
        "--no-headers", action="store_true", help="不分析项目头文件依赖"
    )
    parser.add_argument(
        "--show-all", action="store_true", help="显示所有输出（包括非警告信息）"
    )
    parser.add_argument(
        "--quiet", action="store_true", default=True, help="静默模式，减少输出"
    )

    parser.add_argument(
        "--log-file",
        default="clang_check.log",
        help="日志文件路径（默认: clang_check.log）",
    )
    parser.add_argument(
        "files", nargs="*", help="要分析的文件（支持正则表达式），不指定则分析所有文件"
    )

    args = parser.parse_args()

    # 检查构建路径
    if not os.path.exists(args.build_path):
        print(f"Error: Build path does not exist: {args.build_path}")
        sys.exit(1)

    # 初始化分析器
    analyzer = ClangTidyAnalyzer(args.build_path, args.clang_tidy_binary)

    # 收集要分析的文件
    collector = FileCollector(args.build_path)

    try:
        cpp_files = collector.get_cpp_files_from_database()
        print(f"Found {len(cpp_files)} source files in compilation database")

        if args.files:
            # 用户指定了特定文件，使用编译数据库中的文件进行过滤
            file_patterns = [re.compile(pattern) for pattern in args.files]
            filtered_files = set()
            for cpp_file in cpp_files:
                for pattern in file_patterns:
                    if pattern.search(cpp_file):
                        filtered_files.add(cpp_file)
            files_to_analyze = filtered_files
            print(f"Filtering to {len(files_to_analyze)} files matching patterns")

            if not args.no_headers:
                # 添加头文件依赖
                project_headers = collector.find_project_headers(files_to_analyze)
                files_to_analyze = files_to_analyze.union(project_headers)
                print(f"Added {len(project_headers)} project header files")
        else:
            # 没有指定文件，分析所有 git 跟踪的代码
            # git_cpp_files = collector.get_git_tracked_cpp_files()
            # print(f"Found {len(git_cpp_files)} git-tracked C++ files")

            files_to_analyze = cpp_files

            if not args.no_headers:
                # 添加 git 跟踪的头文件
                project_headers = collector.find_project_headers(cpp_files)
                files_to_analyze = files_to_analyze.union(project_headers)
                print(f"Added {len(project_headers)} project header files")

        if not files_to_analyze:
            print("No files to analyze")
            sys.exit(0)

        print(f"Total files to analyze: {len(files_to_analyze)}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # 设置并行度
    max_workers = args.jobs if args.jobs > 0 else multiprocessing.cpu_count()
    print(f"Using {max_workers} parallel workers")

    # 准备日志文件
    log_file = None
    if args.log_file:
        try:
            log_file = open(args.log_file, "w", encoding="utf-8")
            log_file.write("# clang-tidy Analysis Log\n")
            log_file.write(
                f"# Generated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            log_file.write(f"# Total files: {len(files_to_analyze)}\n")
            log_file.write(f"# Build path: {args.build_path}\n")
            if args.checks:
                log_file.write(f"# Checks: {args.checks}\n")
            log_file.write("\n" + "=" * 80 + "\n\n")
            log_file.flush()
            print(f"Logging results to: {args.log_file}")
        except (OSError, IOError) as e:
            print(f"Warning: Failed to open log file {args.log_file}: {e}")
            log_file = None

    print("=" * 80)

    # 运行分析
    print("Starting analysis...")
    results = run_parallel_analysis(
        analyzer,
        files_to_analyze,
        args.checks,
        args.quiet,
        max_workers,
        args.show_all,
        log_file,
    )

    # 显示结果
    error_count = 0
    for _, output, returncode in sorted(results):
        print(output)
        if returncode != 0:
            error_count += 1

    # 总结
    print("=" * 80)
    analysis_summary = f"Analysis completed. {len(results)} files with issues, {error_count} files with errors."
    print(analysis_summary)

    # 写入总结到日志文件
    if log_file:
        try:
            log_file.write("\n" + "=" * 80 + "\n")
            log_file.write(f"# {analysis_summary}\n")
            log_file.write(
                f"# Analysis completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            log_file.close()
            print(f"Results written to: {args.log_file}")
        except (OSError, IOError) as e:
            print(f"Warning: Failed to finalize log file: {e}")

    sys.exit(1 if error_count > 0 else 0)


if __name__ == "__main__":
    main()
