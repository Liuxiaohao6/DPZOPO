TASK=${TASK:-SST-2}
K=${K:-512}
SEED=${SEED:-42}
BS=${BS:-64}
LR=${LR:-2e-5}
WD=${WD:-0}
EVAL_STEP=${EVAL_STEP:-10}
MODEL=${MODEL:-"roberta-large"}
OPT=${OPT:-"sgd"}
NUM_GPU=${NUM_GPU:-1}

LOGITS=$(jq -n '{"SNLI": 3, "MNLI": 3, "trec": 6, "sst-5": 5}["'$TASK'"] // 2')

echo "TASK: $TASK"
echo "K: $K"
echo "Seed: $SEED"
echo "BS: $BS"
echo "LR: $LR"

TASK_EXTRA=""
case $TASK in
    SST-2)
        TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
        MAPPING="{'0':'terrible','1':'great'}"
        ;;
    sst-5)
        TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
        MAPPING="{0:'terrible',1:'bad',2:'okay',3:'good',4:'great'}"
        TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 20 --double_demo"
        ;;
    QQP)
        TEMPLATE=*cls**sent_0**mask*,*+sentl_1**sep+*
        MAPPING="{'0':'No','1':'Yes'}"
        ;;
    QNLI)
        TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
        MAPPING="{'not_entailment':'No','entailment':'Yes'}"
        ;;
    MNLI)
        TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
        MAPPING="{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}"
        TASK_EXTRA="--max_seq_len 256 --first_sent_limit 240"
        ;;
    SNLI)
        TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
        MAPPING="{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}"
        TASK_EXTRA="--max_seq_len 256 --num_sample 4"
        ;;
    trec)
        TEMPLATE="*cls**mask*:*+sent_0**sep+*"
        MAPPING="{0:'Description',1:'Entity',2:'Expression',3:'Human',4:'Location',5:'Number'}"
        TASK_EXTRA="--first_sent_limit 110"
        ;;
    mr)
        TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
        MAPPING="{0:'terrible',1:'great'}"
        TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 50"
        ;;
    cr)
        TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
        MAPPING="{0:'terrible',1:'great'}"
        TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 50"
        ;;
    mpqa)
        TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
        MAPPING="{0:'terrible',1:'great'}"
        TASK_EXTRA="--first_sent_limit 110"
        ;;
    CoLA)
        TEMPLATE=*cls**sent_0*_This_is*mask*.*sep+*
        MAPPING="{'0':'incorrect','1':'correct'}"
        ;;
    subj)
        TEMPLATE=*cls**sent_0*_This_is*mask*.*sep+*
        MAPPING="{0:'subjective',1:'objective'}"
        TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 50"
        ;;
    MRPC)
        TEMPLATE=*cls**sent_0**mask*,*+sentl_1**sep+*
        MAPPING="{'0':'No','1':'Yes'}"
        ;;
    RTE)
        TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
        MAPPING="{'not_entailment':'No','entailment':'Yes'}"
        TASK_EXTRA="--max_seq_len 256 --first_sent_limit 240"
        ;;
esac

GR_TAG=seed$SEED-bs$BS-lr$LR-wd$WD

ALL_ARGS_TOGETHER="
    --model_name_or_path $MODEL
    --task_name $TASK --template $TEMPLATE --mapping $MAPPING
    --data_dir data/k-data/$TASK/$K-$SEED
    --overwrite_output_dir --output_dir result/$TASK-$MODEL-$GR_TAG/$K-$SEED
    --num_k $K
    --max_seq_length 128
    --seed $SEED
    --do_eval --do_predict --do_train
    --learning_rate $LR
    --optimizer $OPT
    --lr_scheduler_type "constant"
    --weight_decay $WD
    --eval_steps $EVAL_STEP
    --logging_steps 10
    --per_device_train_batch_size $BS
    --per_device_eval_batch_size $BS
    --evaluate_during_training
    --zero_order_optim
    $TASK_EXTRA
"
echo "$ALL_ARGS_TOGETHER"

if [[ $NUM_GPU > 1 ]]; then
    PORT_ID=$(expr $RANDOM + 1000)

    export OMP_NUM_THREADS=8

    python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID run.py \
        $ALL_ARGS_TOGETHER
else
    python run.py \
        $ALL_ARGS_TOGETHER
    # python -m debugpy --wait-for-client --listen 5678 run.py \
    #     $ALL_ARGS_TOGETHER
fi