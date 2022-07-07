#ifndef PARALLEL_UTILS_HPP
#define PARALLEL_UTILS_HPP

#include <algorithm>
#include <thread>
#include <functional>
#include <vector>


namespace parallel_utils
{
    /// @param[in] nb_elements : size of your for loop
    /// @param[in] functor(start, end) :
    /// your function processing a sub chunk of the for loop.
    /// "start" is the first index to process (included) until the index "end"
    /// (excluded)
    /// @code
    ///     for(int i = start; i < end; ++i)
    ///         computation(i);
    /// @endcode
    /// @param use_threads : enable / disable threads.
    ///
    ///
    template <typename Callback>
    static void RawForEach(size_t nb_elements, Callback functor, bool use_threads = true)
    {
        // -------
        size_t nb_threads_hint = std::thread::hardware_concurrency();
        size_t nb_threads = nb_threads_hint == 0 ? 8 : (nb_threads_hint);

        size_t batch_size = nb_elements / nb_threads;
        size_t batch_remainder = nb_elements % nb_threads;

        std::vector< std::thread > my_threads(nb_threads);

        if( use_threads )
        {
            // Multithread execution
            for(size_t i = 0; i < nb_threads; ++i)
            {
                size_t start = i * batch_size;
                my_threads[i] = std::thread(functor, start, start+batch_size);
            }
        }
        else
        {
            // Single thread execution (for easy debugging)
            for(size_t i = 0; i < nb_threads; ++i){
                size_t start = i * batch_size;
                functor( start, start+batch_size );
            }
        }

        // Deform the elements left
        size_t start = nb_threads * batch_size;
        functor( start, start+batch_remainder);

        // Wait for the other thread to finish their task
        if( use_threads )
            std::for_each(my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));
    }

/// @param[in] nb_elements : size of your for loop
    /// @param[in] functor(start, end) :
    /// your function processing a sub chunk of the for loop.
    /// "start" is the first index to process (included) until the index "end"
    /// (excluded)
    /// @code
    ///     for(int i = start; i < end; ++i)
    ///         computation(i);
    /// @endcode
    /// @param use_threads : enable / disable threads.
    ///
    ///
    template <typename Callback>
    static void ConstRawForEach(size_t nb_elements, const Callback functor, bool use_threads = true)
    {
        // -------
        size_t nb_threads_hint = std::thread::hardware_concurrency();
        size_t nb_threads = nb_threads_hint == 0 ? 8 : (nb_threads_hint);

        size_t batch_size = nb_elements / nb_threads;
        size_t batch_remainder = nb_elements % nb_threads;

        std::vector< std::thread > my_threads(nb_threads);

        if( use_threads )
        {
            // Multithread execution
            for(size_t i = 0; i < nb_threads; ++i)
            {
                size_t start = i * batch_size;
                my_threads[i] = std::thread(functor, start, start+batch_size);
            }
        }
        else
        {
            // Single thread execution (for easy debugging)
            for(size_t i = 0; i < nb_threads; ++i){
                size_t start = i * batch_size;
                functor( start, start+batch_size );
            }
        }

        // Deform the elements left
        size_t start = nb_threads * batch_size;
        functor( start, start+batch_remainder);

        // Wait for the other thread to finish their task
        if( use_threads )
            std::for_each(my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));
    }

    template <typename Callback>
    static void ForEach(size_t size, Callback& functor) 
    {
        RawForEach(size, [&](size_t start, size_t end)
        { 
            for(size_t i = start; i < end; i++)
            {
                functor(i);
            }
        });
    }

    template <typename Callback>
    static void ConstForEach(size_t size, const Callback& functor) 
    {
        ConstRawForEach(size, [&](size_t start, size_t end)
        { 
            for(size_t i = start; i < end; i++)
            {
                functor(i);
            }
        });
    }

    template <typename Callback>
    static void ForEach3(size_t sizeX, size_t sizeY, size_t sizeZ, Callback& functor) 
    {
        RawForEach(sizeX, [&](size_t start, size_t end)
        { 
            for(size_t i = start; i < end; i++)
            {
                for(size_t j = 0; j < sizeY; j++)
                {
                    for(size_t k = 0; k < sizeZ; k++)
                    {
                        functor(i, j, k);
                    }
                }
            }
        });
    }

    template <typename Callback>
    static void ConstForEach3(size_t sizeX, size_t sizeY, size_t sizeZ, const Callback& functor) 
    {
        ConstRawForEach(sizeX, [&](size_t start, size_t end)
        { 
            for(size_t i = start; i < end; i++)
            {
                for(size_t j = 0; j < sizeY; j++)
                {
                    for(size_t k = 0; k < sizeZ; k++)
                    {
                        functor(i, j, k);
                    }
                }
            }
        });
    }
}

#endif // PARALLEL_UTILS_HPP