# ESA (Edge Set Attention)

### UNOFFICIAL implementation of the paper:
  - **An end-to-end attention-based approach for learning on graphs**
  - https://arxiv.org/abs/2402.10793
  - Authors : David Buterez, Jon Paul Janet, Dino Oglic, Pietro Lio

I found this paper really great and wanted to try implementing it in Pytorch.

> NOTE : Everything will not correspond exactelly to the paper since all information is not available, and I already made some changes that I found interesting (For example using the `torch.nn.scaled_dot_product_attention` with the edge adjacency matrix).
