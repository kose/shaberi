#!/usr/bin/env bash

##
## nohup ./00benchmark.sh > output.log 2>&1 &
## pkill -f ./00benchmark.sh
##

export OLLAMA_HOST="localhost"
export OLLAMA_CONTEXT_LENGTH=32768
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
LOGFILE=${SCRIPT_DIR}/data/elapsed_time.jsonl
cd "${SCRIPT_DIR}"

# 処理時間計測・保存
elapsed_time()
{
    COMMAND=$1

    start_time=$(date +%s)

    DATE=`date +'%Y-%m-%d %H:%M:%S'`
    echo -e "\033[32m${COMMAND}\033[m"
    ${COMMAND}

    end_time=$(date +%s)

    elapsed_time=$((end_time - start_time))

    echo "{\"timestamp\": \"${DATE}\", \"command\": \"${COMMAND}\", \"処理時間\": ${elapsed_time}}"  | tee -a $LOGFILE
}

## ベンチマーク試験

JUDGE_MODEL="gpt-oss:20b"
# JUDGE_MODEL="qwen3.8:27b"

# JUDGE_MODEL="gemini-3.5-flash"
# ADD_OPTION="--gemini-judge"

for M in gemma4:e4b gpt-oss:20b \
	     gemma4:26b gemma4:12b gemma2:9b gemma4:e4b gemma4:e2b \
	     qwen3.8:27b qwen3.5:27b qwen3.5:9b qwen3.5:4b qwen3.5:2b \
	     yuiseki/sarashina2.2:3b yuiseki/sarashina2.2:1b \
	     phi4:14b phi4-mini:3.8b \
	     dsasai/llama3-elyza-jp-8b:latest \
	     digitsflow/bonsai-8b:latest; do

    for T in elyza/ELYZA-tasks-100 lightblue/tengu_bench shisa-ai/ja-mt-bench-1shot yuzuai/rakuda-questions; do

	TT=$(echo $T | sed 's/\//__/')
	MM=$(echo $M | sed 's/\//__/')

	ANSWER_FILE=data/model_answers/${TT}/${MM}.json
	JUDGE_FILE=data/judgements/judge_${JUDGE_MODEL}/${TT}/${MM}.json

	if [ ! -s ${ANSWER_FILE} ]; then
	    echo "${ANSWER_FILE} が存在しません。"
	    
	    elapsed_time "uv run generate_answers.py --model_name "${M}" -fp 0.5 --eval_dataset_name "${T}" --num_proc 1"
	fi

	if [ ! -s ${JUDGE_FILE} ]; then
	    echo "${JUDGE_FILE} が存在しません。"

	    elapsed_time "uv run judge_answers.py -m "${M}" --judge-model ${JUDGE_MODEL} --eval_dataset_name "${T}" --num_proc 0 ${ADD_OPTION}"
	fi

	exit 0
    done
done

echo "End:"

exit 0

# end
