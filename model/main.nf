nextflow.enable.dsl=2

process PREDICT {
    publishDir params.output_dir, mode: 'copy'

    input:
    path input_dir

    output:
    path 'predictions.csv'

    script:
    """
    python /opt/merit/predict.py \
        --input-dir ${input_dir} \
        --output predictions.csv \
        --bundle /opt/merit/model_bundle.pkl
    """
}

workflow {
    PREDICT(Channel.fromPath(params.input_dir, type: 'dir'))
}
