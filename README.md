# end_2_end_model

Whole image classigier proposed in "Deep Learning to Improve Breast Cancer Detection on Screening Mammography". This paper can be found [here](https://arxiv.org/pdf/1708.09427.pdf).
This code uses torchrun to paralleize the training stage in multiple nodes.  

##Usage 

Run the model in a single node with multiple GPUs:

torchrun --standalone --nnodes=1 --nproc_per_node=2 main.py -b 1 -w 1
