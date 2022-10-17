for model in bert-base-uncased roberta-base roberta-large
do
    python -u bert.py --model $model > log-$model.log
done
