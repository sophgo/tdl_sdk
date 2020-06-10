debugfs=/sys/kernel/debug
echo nop > $debugfs/tracing/current_tracer
echo 0 > $debugfs/tracing/tracing_on
# Due to limited memory, the events are off
# echo 0 > $debugfs/tracing/events/irq/enable
# echo 0 > $debugfs/tracing/events/sched/sched_wakeup/enable

echo 1 > $debugfs/tracing/events/cpuhp/enable
echo 1 > $debugfs/tracing/events/sched/enable
echo 1 > $debugfs/tracing/events/task/enable

echo 30000 > $debugfs/tracing/buffer_size_kb # 30mb
echo > $debugfs/tracing/trace
echo 1 > $debugfs/tracing/tracing_on
exec "$@"

# tracer: nop
#
#                              _-----=> irqs-off
#                             / _----=> need-resched
#                            | / _---=> hardirq/softirq
#                            || / _--=> preempt-depth
#                            ||| /     delay
#           TASK-PID   CPU#  ||||    TIMESTAMP  FUNCTION
#              | |       |   ||||       |         |
#          <...>-12390 [007] ....  5891.897055: tracing_mark_write: B|12390|SIMD
#          <...>-12390 [007] ....  5891.897603: tracing_mark_write: E# tracer: nop
#
#                              _-----=> irqs-off
#                             / _----=> need-resched
#                            | / _---=> hardirq/softirq
#                            || / _--=> preempt-depth
#                            ||| /     delay
#           TASK-PID   CPU#  ||||    TIMESTAMP  FUNCTION
#              | |       |   ||||       |         |
#          <...>-12390 [007] ....  5891.897055: tracing_mark_write: B|12390|SIMD
#          <...>-12390 [007] ....  5891.897603: tracing_mark_write: E