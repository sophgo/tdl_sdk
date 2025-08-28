#!/bin/sh

# 改进的日常回归测试脚本 v2.0
# Daily Regression Test Script v2.0

# set -euo pipefail

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_INFO_PATH="${SCRIPT_DIR}/../build_info.txt"
CONFIG_DIR="${SCRIPT_DIR}/config"
DEFAULT_CONFIG="${CONFIG_DIR}/api_reg_config.yaml"

# 全局变量
CONFIG_model_dir=""
CONFIG_dataset_dir=""
CONFIG_asset_dir=""
CONFIG_log_dir=""
CONFIG_test_flag=""
CONFIG_chip_arch=""
FAILED_TESTS=""
TOTAL_TESTS=0
FAILED_COUNT=0
MODE="TEST" # 默认为测试模式

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {  
    echo -e "${BLUE}[INFO]${NC} $*" >&2
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*" >&2
}

# 模式相关日志
log_mode_info() {
    if [ "$MODE" = "GENERATE" ]; then
        echo -e "${BLUE}[GENERATE]${NC} $*" >&2
    else
        echo -e "${BLUE}[TEST]${NC} $*" >&2
    fi
}

log_mode_success() {
    if [ "$MODE" = "GENERATE" ]; then
        echo -e "${GREEN}[GENERATE SUCCESS]${NC} $*" >&2
    else
        echo -e "${GREEN}[TEST SUCCESS]${NC} $*" >&2
    fi
}

log_mode_error() {
    if [ "$MODE" = "GENERATE" ]; then
        echo -e "${RED}[GENERATE ERROR]${NC} $*" >&2
    else
        echo -e "${RED}[TEST ERROR]${NC} $*" >&2
    fi
}

# 显示帮助信息
print_usage() {
    cat << EOF
改进的日常回归测试脚本

用法: $0 [选项]

选项:
    -c, --config FILE       配置文件路径 (默认: ${DEFAULT_CONFIG})
    -m, --models DIR        模型目录路径
    -d, --dataset DIR       数据集目录路径  ,
    -f, --flag              测试标志 (function|generate_function_res)
    -h, --help              显示此帮助信息

示例:
    $0 -c configs/api_reg_config.yaml -m /path/to/tdl_models -d /path/to/aisdk_daily_regression #功能回归测试
    $0 -c configs/api_reg_config.yaml -m /path/to/tdl_models -d /path/to/aisdk_daily_regression -f generate_function_res #生成功能测试json数据


EOF
}

# 解析YAML配置文件（纯sh/awk，支持 map、嵌套map、列表）
parse_yaml() {
    local file="$1"
    local prefix="$2"
    if [ ! -f "$file" ]; then
        log_error "配置文件不存在: $file"
        exit 1
    fi
    awk -v prefix="$2" '
    function ltrim(s){ sub(/^[ \t\r\n]+/, "", s); return s }
    function rtrim(s){ sub(/[ \t\r\n]+$/, "", s); return s }
    function trim(s){ return rtrim(ltrim(s)) }
    function dequote(s){ s=trim(s); if (s ~ /^".*"$/) return substr(s,2,length(s)-2); if (s ~ /^\x27.*\x27$/) return substr(s,2,length(s)-2); return s }
    function join_path(n){ s=""; for(i=1;i<=n;i++){ if(i>1) s=s"_"; s=s key_stack[i] } return s }
    BEGIN{ stack_len=0 }
    /^[ \t]*#/ { next }
    /^[ \t]*$/ { next }
    {
      indent=match($0,/[^ ]/)-1; if (indent<0) indent=0; level=int(indent/2)
      line=substr($0, indent+1)
      while (stack_len>level) { delete key_stack[stack_len]; stack_len-- }
      if (line ~ /^- /) {
        item=trim(substr(line,3)); item=dequote(item)
        path=join_path(stack_len); idx=counts[path]
        printf("%s%s_%d=\"%s\"\n", prefix, path, idx, item)
        counts[path]=idx+1; next
      }
      split(line, kv, ":")
      key=trim(kv[1]); val=line; sub(/^[^:]*:/, "", val); val=trim(val)
      if (val=="") { stack_len++; key_stack[stack_len]=key }
      else { val=dequote(val); path=join_path(stack_len); if (path!="") path=path"_"; printf("%s%s%s=\"%s\"\n", prefix, path, key, val) }
    }
    ' "$file"
}

# 检测芯片架构
detect_chip_arch() {

  # 获取当前脚本的绝对路径并提取平台信息
  local CURRENT_DIR="$(cd "$(dirname "$0")" && pwd)"
  local platform_name=$(basename "$(dirname "$CURRENT_DIR")")
  
  # 映射平台名称（特殊处理 SOPHON -> CV186X）
  case "$platform_name" in
    "SOPHON") echo "CV186X"; return 0 ;;
    "CV181X"|"CV184X"|"BM1688"|"BM1684X") 
      echo "$platform_name"; return 0 ;;
  esac
  
  # 后备方案：硬件检测
  local chip_info
  chip_info=$(devmem 0x300008c 2>/dev/null || true)
  if echo "$chip_info" | grep -q "181"; then echo "CV181X"; return 0; fi
  if echo "$chip_info" | grep -q "184"; then echo "CV184X"; return 0; fi
  if [ -f "/sys/kernel/debug/ion/cvi_npu_heap_dump/total_mem" ]; then echo "CV186X"; return 0; fi
  if [ -f "/proc/soph/vpss" ]; then echo "BM1688"; return 0; fi
  echo "UNKNOWN"; return 1
}

# 加载配置
load_config() {
    local config_file="${1:-$DEFAULT_CONFIG}"
    
    log_info "加载配置文件: $config_file"
    
    # 解析YAML配置
    local config_vars
    config_vars=$(parse_yaml "$config_file" "CONFIG_")
    eval "$config_vars"
    
    # 设置默认值
    CONFIG_model_dir="${MODEL_DIR:-${CONFIG_paths_model_dir:-/mnt/data/cvimodel}}"
    CONFIG_dataset_dir="${DATASET_DIR:-${CONFIG_paths_dataset_dir:-/mnt/data/dataset}}"
    CONFIG_asset_dir="${ASSET_DIR:-${CONFIG_paths_asset_dir:-/mnt/data/asset}}"
    CONFIG_log_dir="${CONFIG_paths_log_dir:-/tmp/regression_logs}"
    CONFIG_test_flag="${TEST_FLAG:-}"
    
    # 检测芯片架构
    CONFIG_chip_arch=$(detect_chip_arch)
    
    log_info "配置加载完成"
    log_info "芯片架构: ${CONFIG_chip_arch}"
    log_info "模型目录: ${CONFIG_model_dir}"
    log_info "数据集目录: ${CONFIG_dataset_dir}"
    log_info "资源目录: ${CONFIG_asset_dir}"
    log_info "测试标志: ${CONFIG_test_flag}"
}

# 获取测试文件列表
get_test_files() {
    local suite="$1"
    local arch="${CONFIG_chip_arch}"
    local files=""

    # 获取需要排除的文件列表
    local exclude_files=""
    local i=0
    while true; do
        local exclude_varname="CONFIG_test_files_exclude_${arch}_${suite}_${i}"
        local exclude_value=$(eval echo "\${$exclude_varname:-}")
        if [ -z "$exclude_value" ]; then
            break
        fi
        exclude_files="$exclude_files $exclude_value"
        i=$((i + 1))
    done

    # 聚合 common 文件
    i=0
    while true; do
        local varname="CONFIG_test_files_common_${suite}_${i}"
        local value=$(eval echo "\${$varname:-}")
        if [ -z "$value" ]; then
            break
        fi
        
        # 检查当前文件是否需要排除
        local should_exclude=false
        for exclude_file in $exclude_files; do
            if [ "$value" = "$exclude_file" ]; then
                should_exclude=true
                break
            fi
        done
        
        # 如果不需要排除，则添加到文件列表
        if [ "$should_exclude" = "false" ]; then
            files="$files $value"
        fi
        i=$((i + 1))
    done

    # 输出文件列表
    echo "$files" | tr ' ' '\n' | grep -v '^$'
}

# 收集所有 test_suites 名称
get_all_test_suites() {
    # 使用set查看所有变量，然后过滤出我们需要的
    set | grep '^CONFIG_test_suites_' | grep '_name=' | while read line; do
        local var_name="${line%%=*}"
        local suite="${var_name#CONFIG_test_suites_}"
        suite="${suite%_name}"
        echo "$suite"
    done | sort -u
}

# 执行单个回归测试
run_test_suite() {
    local suite="$1"
    local files=""
    
    if [ "$MODE" = "GENERATE" ]; then
        log_mode_info "准备为回归测试生成json数据: $suite"
    else
        log_mode_info "准备执行回归测试: $suite"
    fi
    
    # 获取测试文件列表
    files=$(get_test_files "$suite")
    
    if [ -z "$files" ]; then
        log_warn "回归测试 $suite 没有找到测试文件"
        return 0
    fi
    
    # 计算文件数量
    local file_count=$(echo "$files" | wc -l)
    
    if [ "$MODE" = "GENERATE" ]; then
        log_info "回归测试 $suite 包含 $file_count 个需要生成json数据的测试文件"
    else
        log_info "回归测试 $suite 包含 $file_count 个测试文件"
    fi
    
    if [ "${VERBOSE:-false}" = "true" ]; then
        echo "$files" | while read -r file; do
            log_info "  - $file"
        done
    fi
    
    # get suite name for filter
    local filter_var="CONFIG_test_suites_${suite}_name"
    local suite_name=$(eval echo "\${$filter_var:-}")
    
    if [ -z "$suite_name" ]; then
        log_warn "未找到回归测试 $suite 的过滤器配置"
        return 0
    fi
    
  # 执行测试
    while IFS= read -r file; do
        [ -z "$file" ] && continue # 跳过空行
        
        # 构建测试命令
        if [ "$MODE" = "GENERATE" ]; then
            local cmd="./test_main \"${CONFIG_model_dir}\" \"${CONFIG_dataset_dir}\" \"${file}\" generate_function_res --gtest_filter=\"$suite_name\""
        else
            local cmd="./test_main \"${CONFIG_model_dir}\" \"${CONFIG_dataset_dir}\" \"${file}\" --gtest_filter=\"$suite_name\""
        fi
        
        if [ "$MODE" = "GENERATE" ]; then
            log_mode_info "生成json数据: $file"
        else
            log_mode_info "执行测试: $file"
        fi
        log_info "命令: $cmd"
        
        # 执行测试
        if [ "${DRY_RUN:-false}" = "true" ]; then
            echo "DRY-RUN: $cmd"
            TOTAL_TESTS=$((TOTAL_TESTS + 1))
            continue
        fi
        
        # 执行实际测试命令
        if eval "$cmd"; then
            if [ "$MODE" = "GENERATE" ]; then
                log_mode_success "json数据生成成功: $file"
            else
                log_mode_success "测试通过: $file"
            fi
        else
            if [ "$MODE" = "GENERATE" ]; then
                log_mode_error "json数据生成失败: $file"
            else
                log_mode_error "测试失败: $file"
            fi
            FAILED_TESTS="$FAILED_TESTS $file"
            FAILED_COUNT=$((FAILED_COUNT + 1))
        fi
        
        # 更新测试计数
        TOTAL_TESTS=$((TOTAL_TESTS + 1))
        
    done <<EOF
$files
EOF
}

# 生成测试报告
generate_report() {
    local format="${1:-console}"
    
    if [ "$MODE" = "GENERATE" ]; then
        log_info "=== json数据生成报告 ==="
        log_info "总处理文件数: $TOTAL_TESTS"
        log_info "失败处理数: $FAILED_COUNT"
        log_info "成功处理数: $((TOTAL_TESTS - FAILED_COUNT))"
        
        if [ $FAILED_COUNT -gt 0 ]; then
            log_error "json数据生成失败的文件:"
            for test in $FAILED_TESTS; do
                log_error "  - $test"
            done
        else
            log_success "所有json数据都已成功生成!"
        fi
    else
        log_info "=== 回归测试报告 ==="
        log_info "总测试数: $TOTAL_TESTS"
        log_info "失败测试数: $FAILED_COUNT"
        log_info "成功测试数: $((TOTAL_TESTS - FAILED_COUNT))"
        
        if [ $FAILED_COUNT -gt 0 ]; then
            log_error "失败的测试:"
            for test in $FAILED_TESTS; do
                log_error "  - $test"
            done
        else
            log_success "所有测试都通过了!"
        fi
    fi
}

# 运行所有测试
run_tests() {
    local suites="$1"
    local SUITE_LIST=""

    # 确保测试可执行文件存在
    if [ ! -x "./test_main" ]; then
        log_error "测试可执行文件 ./test_main 不存在或无执行权限"
        exit 1
    fi

    if [ -z "$suites" ]; then
        SUITE_LIST=$(get_all_test_suites | tr '\n' ' ')
    else
        SUITE_LIST="$suites"
    fi

    if [ "$MODE" = "GENERATE" ]; then
        log_info "开始生成json数据"
    else
        log_info "开始执行回归测试"
    fi
    log_info "回归测试: $SUITE_LIST"

    for suite in $SUITE_LIST; do
        run_test_suite "$suite" 
    done
}

# 主函数
main() {
    local config_file="$DEFAULT_CONFIG"
    local test_suites=""
    local output_format="console"
    
    # 解析命令行参数
    while [ $# -gt 0 ]; do
        case $1 in
            -c|--config)
                config_file="$2"
                shift 2
                ;;
            -m|--models)
                MODEL_DIR="$2"
                shift 2
                ;;
            -d|--dataset)
                DATASET_DIR="$2"
                shift 2
                ;;
            -f|--flag)
                TEST_FLAG="$2"
                shift 2
                ;;
            -n|--dry-run)
                DRY_RUN="true"
                shift
                ;;
            -h|--help)
                print_usage
                exit 0
                ;;
            *)
                log_error "未知参数: $1"
                print_usage
                exit 1
                ;;
        esac
    done
    
    # 如果没有指定配置文件、模型目录或数据集目录，打印帮助信息
    if [ -z "$config_file" ] || [ -z "$MODEL_DIR" ] || [ -z "$DATASET_DIR" ]; then
        print_usage
        exit 0
    fi
    
    # 根据标志确定模式
    if [ "${TEST_FLAG}" = "generate_function_res" ]; then
        MODE="GENERATE"
    fi

    # 加载配置
    load_config "$config_file"

    # 运行测试
    run_tests "$test_suites"
    
    # 生成报告
    generate_report "$output_format"
    
    # 返回适当的退出码
    if [ $FAILED_COUNT -gt 0 ]; then
        if [ "$MODE" = "GENERATE" ]; then
            log_error "json数据生成存在失败项"
        else
            log_error "回归测试存在失败项"
        fi
        exit 1
    else
        if [ "$MODE" = "GENERATE" ]; then
            log_success "所有json数据已成功生成"
        else
            log_success "所有回归测试已成功通过"
        fi
        exit 0
    fi
}

# 如果直接执行此脚本
if [ "$(basename "$0")" = "daily_regression_v2.sh" ]; then
    main "$@"
fi