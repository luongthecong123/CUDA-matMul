#include <vector>
#include <random>

template <typename T>
std::vector<T> create_rand_vector(size_t n)
/*
Generate a random square matrix
Used 1-D array for coalesced memory access and easier memory transfer
Params:
    - N : The dimension of square matrix

Returns:
    std::vector<T>
 */ 
{
    std::random_device r;
    std::default_random_engine e(r());
    std::uniform_int_distribution<int> uniform_dist(-32, 32);

    std::vector<T> vec(n);
    for (size_t i{0}; i < n; ++i)
    {
        vec.at(i) = static_cast<T>(uniform_dist(e));
    }

    return vec;
}

template <typename T>
void printVec(const std::vector<T> &vec, size_t m, size_t n)
{
    std::cout<<"np.array ([";
    for (int i=0; i < m; i++)
    {   std::cout<<"[";
        for (int j=0; j < n; j++)
        {
            std::cout<<vec[i*n + j]<<"   ";
            if (j != n-1){std::cout<<",";}
        }
        if (i != m-1){std::cout<<"],";}
        else{std::cout<<"]";}
        std::cout<<"\n";
    }
    std::cout<<"])\n";
}