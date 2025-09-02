# Full Stack Deep Learning Spring 2021 Labs

Welcome!

As part of Full Stack Deep Learning 2021, we will incrementally develop a complete deep learning codebase to understand the content of handwritten paragraphs.

We will use the modern stack of PyTorch and PyTorch-Ligtning

We will use the main workhorses of DL today: CNNs, RNNs, and Transformers

We will manage our experiments using what we believe to be the best tool for the job: Weights & Biases

We will set up continuous integration system for our codebase using CircleCI

We will package up the prediction system as a REST API using FastAPI, and deploy it as a Docker container on AWS Lambda.

We will set up monitoring that alerts us when the incoming data distribution changes.

Sequence:

- [Lab Setup](setup/readme.md): Set up our computing environment.
- [Lab 1: Intro](lab1/readme.md): Formulate problem, structure codebase, train an MLP for MNIST.
- [Lab 2: CNNs](lab2/readme.md): Introduce EMNIST, generate synthetic handwritten lines, and train CNNs.
- [Lab 3: RNNs](lab3/readme.md): Using CNN + LSTM with CTC loss for line text recognition.
- [Lab 4: Transformers](lab4/readme.md): Using Transformers for line text recognition.
- [Lab 5: Experiment Management](lab5/readme.md): Real handwriting data, Weights & Biases, and hyperparameter sweeps.
- [Lab 6: Data Labeling](lab6/readme.md): Label our own handwriting data and properly store it.
- [Lab 7: Paragraph Recognition](lab7/readme.md): Train and evaluate whole-paragraph recognition.
- [Lab 8: Continuous Integration](lab8/readme.md): Add continuous linting and testing of our code.
- [Lab 9: Deployment](lab9/readme.md): Run as a REST API locally, then in Docker, then put in production using AWS Lambda.
- [Lab 10: Monitoring](lab10/readme.md): Set up monitoring that alerts us when the incoming data distribution changes.

lab3 start on Windows
# 1) make sure the env is active
conda activate fsdl2021

# 2) make the repo importable (Windows uses ';' as the path separator)
$env:PYTHONPATH = "$PWD\lab3;$PWD"

# 3) (optional) quiet the TF32 warning by setting the new knobs *inside* Python
#    You can ignore this if you want — it’s just a warning.
# python -c "import torch; torch.backends.cuda.matmul.fp32_precision='tf32'; torch.backends.cudnn.conv.fp32_precision='tf32'"

# 4) train on the GPU with mixed precision
python -m lab3.training.run_experiment `
  --max_epochs=40 `
  --data_class=sentence_generator.SentenceGenerator `
  --model_class=LineCNNSimple `
  --batch_size=256 `
  --num_workers=16 `
  --prefetch_factor=4 `
  --accelerator=gpu --devices=1 `
  --precision=16-mixed `
  --loss=ctc_loss `
  --line_image_height=28 `
  --sentence_max_length=24 `
  --output_timesteps=64 `
  --conv_dim=128 `
  --fc_dim=256

# Check GPU usage and other stats
while ($true) {
    # Get GPU usage, memory, and temperature
    $data = nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv | Select-Object -Skip 1
    $fields = $data -split ", "
    $gpuUsage = $fields[0]
    $memoryUsed = $fields[1]
    $memoryTotal = $fields[2]
    $temperature = $fields[3]
    # Clear the console
    Clear-Host
    # Display formatted output
    Write-Host "GPU Usage: $gpuUsage"
    Write-Host "Memory: $memoryUsed / $memoryTotal"
    Write-Host "Temperature: $temperature C"
    # Wait for 1 second
    Start-Sleep -Seconds 1
}

# Run preview_decode.py
$env:PYTHONPATH = "$PWD\lab3;$PWD"
python preview_decode.py --ckpt "lightning_logs\version_17\checkpoints\epoch=29-step=1200.ckpt" `
  --samples 8 --img_height 28 --output_timesteps 64 --device cuda
