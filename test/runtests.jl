using TensorBasic
using Base.Test

# write your own tests here
@test x = rand(2,3,4); y = tensor_permute(x, [1,2,3], [2,1,3]);
