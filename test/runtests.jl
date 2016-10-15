using TensorBasic
using Base.Test

# write your own tests here
@eval x = rand(2,3,4);
@eval y = tensor_permute(x, [1,2,3], [2,1,3]);
@test y == permutedims(x, [2,1,3]);
