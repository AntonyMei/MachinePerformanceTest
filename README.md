# Machine Performance Test
This test suite aims to test the performance of deep learning servers with 4/8 x A100/A40.
The test suite covers CPU, GPU and multi-GPU DDP tests. This is a step-by-step guide that
sets up the environment and runs the tests from scratch.

## 1. Environment Setup
### Machine Requirement:
The server to be tested should have 4 or 8 GPUs in order to run the whole test set. The GPUs
recommended are A100, A10 and A40. To start with, the machine should have `ubuntu 18.04` or 
`ubuntu 20.04`. It should also be installed with `nvidia-driver-510` and `cuda 11.6`.

### Docker Setup
1. Install docker and docker's Nvidia GPU support.
2. Pull docker image from docker hub.
   
        docker pull antonymei/distributed_muzero:torch181
3. Launch container with the following command. The port mapping here can be used to upload
the test files into the container.
   
        docker run -p 1234:22 --privileged --gpus all -itd --shm-size="450g" --name test 73c74281b161 /bin/bash

4. Start docker and attach to it.

        docker start test
        docker exec -it test /bin/bash

5. Upgrade shared memory communication package.
   
         pip install --upgrade SMOS_antony


## 2. EfficientZero Batch Making Test
In this test, the test program runs reanalyze part of EfficientZero. It evaluates the performance
of CPU and GPU.
1. Install MCTS tree from cpp source code.
   
         cd test_1_EfficientZero_batch_making/core/ctree/
         bash make.sh

2. Launch test.
         
         cd ../..
         bash launch_test.sh
   The final output should be something like this. (ignoring all logs)
   
         ************* Begin Testing *************
         [CPU Result] Batch size 1024, Avg. CPU=3.80, Lst. CPU=3.80
         [GPU Result] Batch size 1024, Avg. GPU=6.02, Lst. GPU=5.66
         ************* Test Finished *************
 
## 3. EfficientZero Single Card Training Test 
In this test, the test program runs training part of EfficientZero with only one trainer.
It evaluates the performance of single GPU.
1. Install MCTS tree from cpp source code.
   
         cd test_2_EfficientZero_single_card_training/core/ctree/
         bash make.sh
   
2. Unzip dataset.
         
         cd ../..
         unzip batch.zip
   The output should be something like this.
   
         Archive:  batch.zip
            creating: batch/
           inflating: batch/batch1024.part0
           inflating: batch/batch1024.part1
           inflating: batch/batch1024.part2
2. Launch test.

         bash launch_test.sh
   The final output should be something like this. (ignoring all logs)

         ************* Begin Testing *************
         [Trainer 0] batch size=1024, loop=50, Avg. Tloop=0.58, Lst. Tloop=0.56
         ************* Test Finished *************

## 4. EfficientZero Multi Card DDP Test 
In this test, the test program runs training part of EfficientZero with multiple DDP workers.
It evaluates the performance of single GPU as well as communication speed between multiple GPUs.
Note that torch amp is enabled in this test.

1. Install MCTS tree from cpp source code.
   
         cd test_3_EfficientZero_multi_card_ddp/core/ctree/
         bash make.sh
   
2. Unzip dataset.
         
         cd ../..
         unzip batch.zip
   The output should be something like this.
   
         Archive:  batch.zip
            creating: batch/
           inflating: batch/batch1024.part0
           inflating: batch/batch1024.part1
           inflating: batch/batch1024.part2
2. Launch test. If the machine has four cards, use

         bash launch_test_4card.sh
   
   If the machine has eight cards, use

         bash launch_test_8card.sh

   The final output should be something like this. (ignoring all logs)

         ************* Begin Testing *************
         [Trainer 0] batch size=1024, loop=50, Avg. Tloop=0.65, Lst. Tloop=0.66
         [Trainer 1] batch size=1024, loop=50, Avg. Tloop=0.65, Lst. Tloop=0.66
         [Trainer 2] batch size=1024, loop=50, Avg. Tloop=0.65, Lst. Tloop=0.66
         [Trainer 3] batch size=1024, loop=50, Avg. Tloop=0.65, Lst. Tloop=0.66
         ************* Test Finished *************