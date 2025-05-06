
import torch.nn.parallel
import torch.distributed as dist
import oneccl_bindings_for_pytorch
import os

# os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', '198.18.18.254') 
# os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))

backend = 'ccl'
dist.init_process_group(backend, init_method='env://')
my_rank = dist.get_rank()
my_size = dist.get_world_size()
print("my rank = %d  my size = %d" % (my_rank, my_size))

n = 2
c = 3
x = torch.ones([n, n, c])
y = torch.ones([n *2 , n*2 ,c])
with torch.autograd.profiler.profile(record_shapes=True) as prof:
    for _ in range(10):
        dist.all_reduce(x)
        dist.all_reduce(y)
dist.barrier()
print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total"))
