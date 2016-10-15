# --------------module TensorBasic-----------
# 1. Tensor index is generally 1d int array
# 2. Merge two 1d array [a;b]
# 3. [1 2] size == (1,2) 2d array
# 4. [1,2] 1d array
#---------------------------------------------------



module TensorBasic

# package code goes here
export tensor_product,tensor_permute,tensor_reshape

# Sveral tensor product
# By Huang Ruizhen, 2016/7/8
function tensor_product(tensor_tup, indx_tup)
  tup_no = length(tensor_tup);
  (outT,outInd) = tensor_product(tensor_tup[1], indx_tup[1], tensor_tup[2], indx_tup[2]);
  for i=3:1:tup_no
    (outT,outInd) = tensor_product(outT, outInd, tensor_tup[i], indx_tup[i]);
  end
  return outT,outInd;
end


function tensor_product(A, aindex, B, bindex)
  a_length = length( aindex );
  b_length = length( bindex );

  size_a = size(A);
  size_b = size(B);
  comab = intersect(aindex,bindex);
  if length(comab)==0
    cindex = [aindex;bindex];
    A = reshape(A,length(A));
    B = reshape(B,1,length(B));
    return reshape(A*B,size_a...,size_b...),cindex;
  else
    com_in_a = [findfirst(aindex,xa) for xa in comab];
    com_in_b = [findfirst(bindex,xb) for xb in comab];

    if size_a[com_in_a] != size_b[com_in_b]
      error("Error: Dimentions don't match!");
    end


    diff_in_a = setdiff(collect(1:1:a_length), com_in_a);
    diff_in_b = setdiff(collect(1:1:b_length), com_in_b);
    cindex = [ aindex[diff_in_a]; bindex[diff_in_b] ];

    # mutiply
    if [ com_in_a; diff_in_a ] != collect(1:1:a_length)
        A = permutedims( A, [ com_in_a; diff_in_a ] );
    end
    if [ com_in_b;diff_in_b ] != collect(1:1:b_length)
        B = permutedims( B, [ com_in_b; diff_in_b ] );
    end

    sda = prod(size_a[diff_in_a]);
    sc = prod(size_a[com_in_a]);
    sdb = prod(size_b[diff_in_b]);
    A = reshape(A,sc,sda);
    B = reshape(B,sc,sdb);

    C = A.' * B ;
    return reshape(C,size_a[diff_in_a]..., size_b[diff_in_b]...),cindex;
  end
end


function tensor_product(cindex, A, aindex, B, bindex)
  (C,indc) = tensor_product(A,aindex,B,bindex);
  return tensor_permute(C,indc,cindex);
end



# Tensor permute
# By Huang Ruizhen, 2016.10.15
function tensor_permute(T, oldidx, newidx)
  exchidx = indexin(oldidx, newidx);
  return permutedims(T,exchidx);
end


# hrz, 2016/10/15
function tensor_reshape(T, indx_all, inds)
  size_t = size(T);
  dim_t = ndims(T);
  dim_all = length(indx_all);
  T_ = T;
  if dim_all > dim_t
    size_t = tuple(size_t..., ones(Int64, dim_all-dim_t)...);
    T_ = reshape(T, size_t);
  end

  ind_new = vcat(inds...);
  if indx_all != ind_new
    T_ = tensor_permute(T_, indx_all, ind_new);
  end
  size_t = size(T_);
  n_inds = length(inds);
  dims_i = zeros(Int64, n_inds);
  for i in 1:1:n_inds
    dims_i[i] = prod(size_t[indexin(inds[i],ind_new)]);
  end
  return reshape(T_, dims_i...);
end


end # module
