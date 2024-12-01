#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
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
    std::uniform_int_distribution<int> uniform_dist(-256, 256);

    std::vector<T> vec(n);
    for (size_t i{0}; i < n; ++i)
    {
        vec.at(i) = static_cast<T>(uniform_dist(e));
    }

    return vec;
}

template <typename T>
void printVec(const std::vector<T>& vec, size_t m, size_t n) {
    // std::cout << "np.array([\n";
    for (size_t i = 0; i < m; i++) {
        std::cout << "  [";
        for (size_t j = 0; j < n; j++) {

            std::cout << std::setw(8) << std::setprecision(4) << std::fixed << vec[i * n + j];
            if (j != n - 1) {
                std::cout << ", ";
            }
        }
        if (i != m - 1) {
            std::cout << "],\n";
        }
        else {
            std::cout << "]";
        }
    }
    // std::cout << "\n])\n";
    std::cout << "\n";
}

template <typename T>
bool allclose(std::vector<T> const& vec_1, std::vector<T> const& vec_2, T const& abs_tol)
/*
Compare 2 vectors to check if they're close to each other
*/
{
    if (vec_1.size() != vec_2.size())
    {
        return false;
    }
    for (size_t i{0}; i < vec_1.size(); ++i)
    {
        if (std::abs(vec_1.at(i) - vec_2.at(i)) > abs_tol)
        {
            std::cout << vec_1.at(i) << " " << vec_2.at(i) << std::endl;
            return false;
        }
    }
    return true;
}