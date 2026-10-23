import torch
import aiter
from aiter.ops.shuffle import shuffle_weight


torch.set_default_device("cuda")

def find_non_zero_offset(x: torch.Tensor):
    offsets = torch.arange(x.numel()).view(x.shape)
    return offsets.masked_select(x != 0)


def show_shuffle_coord(gx: int, gy: int):
    M, N, K = 1024, 17408, 5120
    dtype = torch.bfloat16
    x = torch.zeros((M, K), dtype=dtype)
    w = torch.zeros((N, K), dtype=dtype)

    w[gx, gy * 32 : (gy + 1) * 32] = 2

    quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    x, x_scales_shuffle = quant_func(x, shuffle=True)
    w, w_scales_shuffle = quant_func(w, shuffle=True)
    wshuffle = shuffle_weight(w, layout=(16, 16))

    w_scale_shuffle_offsets = find_non_zero_offset(w_scales_shuffle.view(torch.uint8)).cpu()
    wshuffle_offsets = find_non_zero_offset(wshuffle.view(torch.uint8)).cpu()
    return {
        "scale_coord": [(i.item() // w_scales_shuffle.size(1), i.item() % w_scales_shuffle.size(1)) for i in w_scale_shuffle_offsets],
        "weight_coord": [(i.item() // wshuffle.size(1), i.item() % wshuffle.size(1)) for i in wshuffle_offsets]
    }


OM = 32
ON = 32
scale_matrix = [[None for _ in range(ON)] for _ in range(OM)]

for x in range(32):
    tmp = []
    for y in range(32):
        ret = show_shuffle_coord(x, y)
        scale_matrix[ret["scale_coord"][0][0]][ret["scale_coord"][0][1]] = (x, y)

print(scale_matrix)
