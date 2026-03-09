核心模式（默认，推荐）：
bash /data/250010187/yeziyang1/lingbot-va/evaluation/robotwin/distributed_eval.sh \
  /data/share/lijiang/yeziyang-exp/results/chunk-aware-full_eval_0304 \
  100 \
  10000
完整明细模式（必须两个变量都开）：
TIMING_LOG_MODE=full KEEP_CALL_DETAILS=1 \
bash /data/250010187/yeziyang1/lingbot-va/evaluation/robotwin/distributed_eval.sh \
  /data/share/lijiang/yeziyang-exp/results/chunk-aware-full_eval_0304 \
  100 \
  10000
强制回到核心：
TIMING_LOG_MODE=core KEEP_CALL_DETAILS=0 \
bash /data/250010187/yeziyang1/lingbot-va/evaluation/robotwin/distributed_eval.sh ...
这两个变量在 distributed_eval.sh (line 258) 注入到客户端进程。

直接运行客户端（不经过 distributed 脚本）
核心：
LINGBOT_TIMING_LOG_MODE=core LINGBOT_KEEP_CALL_DETAILS=0 python -m evaluation.robotwin.eval_with_logging ...
完整：
LINGBOT_TIMING_LOG_MODE=full LINGBOT_KEEP_