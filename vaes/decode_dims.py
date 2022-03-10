

def calc_dims(shape, strides, k_size,padding,output_padding):
    new_depth = ((shape[0] - 1) * strides[0] + k_size[0] - 2 * padding[0] + output_padding[0])

    new_rows = ((shape[1] - 1) * strides[1] + k_size[1] - 2 * padding[1] + output_padding[1])

    new_cols = ((shape[2] - 1) * strides[2] + k_size[2] - 2 * padding[2] + output_padding[2])

    return (new_depth,new_rows,new_cols)

def main():
    shape = (2,62,62)
    k_size = (3,3,3)
    strides = (2,2,2)
    padding = (0,0,0)
    output_padding = padding

    new_shape = calc_dims(shape,strides,k_size,padding,output_padding)
    print(new_shape)

    k_size = (1,1,1)
    strides = (2,2,2)
    new_shape = calc_dims(new_shape,strides,k_size,padding,output_padding)
    print(new_shape)

    k_size = (1,4,4)
    strides = (1,2,2)
    new_shape = calc_dims(new_shape,strides,k_size,padding,output_padding)
    print(new_shape)

    k_size = (1,13,13)
    strides = (1,1,1)
    new_shape = calc_dims(new_shape,strides,k_size,padding,output_padding)
    print(new_shape)


    

if __name__ == '__main__':
    main()