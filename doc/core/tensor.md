- Simple 2D tensor implementation, currently only consists of number of rows, columns and the shared pointer as the 'data'
- The point of using `std::shared_ptr` at the moment is just to support the upcoming implementation of a transposed view, meaning transposed tensor will still technically use the same pointer

