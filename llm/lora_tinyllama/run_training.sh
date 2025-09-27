#!/bin/bash

# TinyLlama Fine-tuning Runner Script
# Usage: ./run_training.sh [lora|qlora] [--sample N]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    print_error "Please run this script from the lora_tinyllama directory"
    exit 1
fi

# Parse arguments
METHOD=${1:-lora}
SAMPLE_SIZE=""

if [ "$2" == "--sample" ]; then
    SAMPLE_SIZE="--sample $3"
    print_info "Running with sample size: $3"
fi

# Check Python installation
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    exit 1
fi

# Check CUDA availability
print_info "Checking CUDA availability..."
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
python3 -c "import torch; print(f'CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB' if torch.cuda.is_available() else '')"

# Create necessary directories
print_info "Creating output directories..."
mkdir -p results logs

# Function to run training
run_training() {
    local method=$1
    local config_file="../configs/${method}_config.yaml"
    local script_file="train_${method}.py"
    
    print_info "Starting $method training..."
    print_info "Config: $config_file"
    print_info "Script: $script_file"
    
    # Change to scripts directory
    cd scripts
    
    # Run the training script
    if [ -n "$SAMPLE_SIZE" ]; then
        python3 $script_file --config $config_file $SAMPLE_SIZE
    else
        python3 $script_file --config $config_file
    fi
    
    # Return to main directory
    cd ..
}

# Main execution
case $METHOD in
    lora)
        print_info "Running LoRA fine-tuning..."
        print_warning "This will use approximately 10GB of VRAM"
        run_training "lora"
        ;;
    
    qlora)
        print_info "Running QLoRA fine-tuning..."
        print_warning "This will use approximately 6-8GB of VRAM"
        run_training "qlora"
        ;;
    
    test)
        print_info "Running quick test with 100 samples..."
        run_training "lora" "--sample 100"
        ;;
    
    explore)
        print_info "Exploring the dataset..."
        cd data
        python3 prepare_data.py explore iamtarun/python_code_instructions_18k_alpaca
        cd ..
        ;;
    
    sample_data)
        print_info "Creating sample dataset..."
        cd data
        python3 prepare_data.py sample --output sample_dataset.json --num-examples 100
        print_info "Sample dataset created at data/sample_dataset.json"
        cd ..
        ;;
    
    tensorboard)
        print_info "Starting TensorBoard..."
        tensorboard --logdir ./logs
        ;;
    
    clean)
        print_warning "Cleaning output directories..."
        read -p "Are you sure? This will delete results and logs. (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf results logs
            print_info "Cleaned successfully"
        else
            print_info "Cancelled"
        fi
        ;;
    
    *)
        echo "TinyLlama Fine-tuning Runner"
        echo ""
        echo "Usage: ./run_training.sh [command] [options]"
        echo ""
        echo "Commands:"
        echo "  lora              Run LoRA fine-tuning"
        echo "  qlora             Run QLoRA fine-tuning (lower memory)"
        echo "  test              Run quick test with 100 samples"
        echo "  explore           Explore the dataset"
        echo "  sample_data       Create a sample dataset"
        echo "  tensorboard       Start TensorBoard monitoring"
        echo "  clean             Clean output directories"
        echo ""
        echo "Options:"
        echo "  --sample N        Use only N samples for testing"
        echo ""
        echo "Examples:"
        echo "  ./run_training.sh lora                  # Full LoRA training"
        echo "  ./run_training.sh qlora --sample 1000   # QLoRA with 1000 samples"
        echo "  ./run_training.sh test                  # Quick test run"
        exit 0
        ;;
esac

print_info "Done!"