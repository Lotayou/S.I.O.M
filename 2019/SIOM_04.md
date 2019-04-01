### 20190401: Memory leakage

When writing batched code, if the output dimension is determined, it's best to pre-alloc memory blocks
instead of list append and stacking.

For example, the following code is bad and prone to memory leakage

```
result = []
for i in range(10000):
    temp = function(input[i])
    result.append(temp)
    
result = torch.stack(result, dim=0)
```

Instead, you should consider this:

```
out_dim = function(input[0]).shape[0]
result = torch.zeros((10000, out_dim))
for i in range(10000):
    result[i] = function(input[i])
```

Of course, it's even better if you implement function in batched manner. But for some options it's not always possible (for example torch.svd), then it's crucial that you manage your memory correctly.